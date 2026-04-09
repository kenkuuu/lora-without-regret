"""
this script requires vllm's v0 engine which has a hacky way to reload model weights
so we have pinned to v0.10.2 release of vllm

accelerate launch --num_processes=<N> rl_full.py --num_train_gpus=<N>
CUDA_VISIBLE_DEVICES must include GPUs 0..<N> for training and GPU <N> for vLLM
e.g. CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=3 rl_full.py --num_train_gpus=3
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from math_utils import last_boxed_only_string, remove_boxed, is_equiv
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import random
import argparse
import wandb
import time
import os
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list, InitProcessGroupKwargs
from datetime import timedelta


def vllm_worker(model_id, vllm_gpu_index, gpu_memory_utilization, cmd_q, result_q):
    """
    Subprocess that owns the vLLM engine on a dedicated GPU.
    Receives commands via cmd_q and sends results via result_q.
    Commands:
      ("update_weights", weight_tuples) -> "OK"
      ("generate", (prompts, sp_kwargs)) -> outputs
      "STOP" -> exit
    """
    # Prevent interference with the training processes' NCCL setup
    for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
                "TORCHELASTIC_AGENT_SOCKET_TIMEOUT", "NCCL_SOCKET_IFNAME"]:
        os.environ.pop(var, None)

    # Map vllm_gpu_index to the actual physical GPU from parent's CUDA_VISIBLE_DEVICES
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if parent_cvd:
        gpu_ids = parent_cvd.split(",")
        vllm_cuda = gpu_ids[vllm_gpu_index] if vllm_gpu_index < len(gpu_ids) else str(vllm_gpu_index)
    else:
        vllm_cuda = str(vllm_gpu_index)

    os.environ["CUDA_VISIBLE_DEVICES"] = vllm_cuda
    os.environ["VLLM_USE_V1"] = "0"

    from vllm import LLM, SamplingParams
    import sys

    # vLLM calls get_ip() to construct the distributed init method URL.
    # On some clusters, the node's external IP (e.g. 192.168.x.x) is unreachable
    # for loopback connections, causing TCPStore timeouts.
    # Patch get_ip in all loaded vllm modules to use localhost instead.
    _patch_get_ip = lambda: "127.0.0.1"
    for _mod in list(sys.modules.values()):
        if hasattr(_mod, "get_ip") and "vllm" in getattr(_mod, "__name__", ""):
            _mod.get_ip = _patch_get_ip

    model = LLM(
        model=model_id,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=2048,
        max_num_batched_tokens=4096,
        logprobs_mode="processed_logprobs",
    )
    result_q.put("READY")

    while True:
        msg = cmd_q.get()
        if msg == "STOP":
            break
        cmd, payload = msg
        if cmd == "update_weights":
            internal = model.llm_engine.model_executor.driver_worker.model_runner.model
            internal.load_weights(payload)
            result_q.put("OK")
        elif cmd == "generate":
            prompts, sp_kwargs = payload
            sp = SamplingParams(**sp_kwargs)
            raw = model.generate(prompts, sp)
            outputs = [o.text for r in raw for o in r.outputs]
            result_q.put(outputs)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a language model with GRPO")

    # Model configuration
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model ID to use",
    )

    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--n_grpo_steps",
        type=int,
        default=50,
        help="Number of GRPO training steps",
    )
    parser.add_argument(
        "--n_prompts_per_step",
        type=int,
        default=32,
        help="Number of prompts to sample per step",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=8,
        help="Number of rollouts per prompt",
    )
    parser.add_argument(
        "--epochs_per_step",
        type=int,
        default=1,
        help="Number of epochs to train per step",
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=4,
        help="Micro batch size for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=64,
        help="Number of gradient accumulation steps",
    )

    # Other configuration
    parser.add_argument(
        "--base_dir",
        type=str,
        default="runs",
        help="Base directory for saving runs",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="boxed.prompt",
        help="Path to prompt template file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Wandb configuration
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="math_grpo_full",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Wandb run name (defaults to run directory name)",
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.8,
        help="vLLM GPU memory utilization (vLLM runs on a dedicated GPU)",
    )
    parser.add_argument(
        "--num_train_gpus",
        type=int,
        default=3,
        help="Number of GPUs for training. vLLM uses GPU index num_train_gpus.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Spawn vLLM subprocess on rank0 BEFORE Accelerator init.
    # vLLM loading and NCCL init then proceed in parallel, avoiding timeout.
    if local_rank == 0:
        ctx = mp.get_context("spawn")
        cmd_queue = ctx.Queue()
        result_queue = ctx.Queue()
        vllm_proc = ctx.Process(
            target=vllm_worker,
            args=(args.model_id, args.num_train_gpus, args.gpu_memory_utilization,
                  cmd_queue, result_queue),
            daemon=True,
        )
        vllm_proc.start()
    else:
        cmd_queue = result_queue = vllm_proc = None

    # Long timeout so all ranks can wait while rank0 waits for vLLM to load
    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=2))]
    )
    device = accelerator.device

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if accelerator.is_main_process:
        print(f"Training configuration:")
        print(f"  Model ID: {args.model_id}")
        print(f"  Learning rate: {args.lr}")
        print(f"  W&B logging: {'enabled' if not args.disable_wandb else 'disabled'}")
        print()

        # Setup run directory
        os.makedirs(args.base_dir, exist_ok=True)
        i = 1
        while os.path.exists(f"{args.base_dir}/{i}"):
            i += 1
        run_name = f"{args.base_dir}/{i}"
        os.makedirs(run_name)
        print(f"Created: {run_name}")

        # Initialize wandb
        if not args.disable_wandb:
            wandb_run_name = args.wandb_run_name or f"{args.model_id}_{args.lr:.1e}_full"
            wandb.init(
                project=args.wandb_project,
                name=wandb_run_name,
                config=vars(args),
                dir=run_name,
            )
            wandb.config.update({"run_dir": run_name})

        # Wait for vLLM subprocess to finish loading
        print("Waiting for vLLM to load...")
        result_queue.get()  # blocks until "READY"
        print("vLLM ready.")

    # Sync all ranks after vLLM is ready
    accelerator.wait_for_everyone()

    # Load dataset and tokenizer
    train_dataset = load_dataset("qwedsacf/competition_math", split=f"train[:7500]")
    val_dataset = load_dataset("qwedsacf/competition_math", split=f"train[-5000:]")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load prompt template
    with open(args.prompt_template, "r", encoding="utf-8") as f:
        template = f.read().strip()

    def process_data(example):
        with_template = template.replace("{question}", example["problem"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": with_template}],
            tokenize=False,
            add_generation_prompt=True,
        )
        answer = remove_boxed(last_boxed_only_string(example["solution"]))
        return {"prompt": prompt, "answer": answer}

    train_dataset = train_dataset.map(process_data)
    val_dataset = val_dataset.map(process_data)

    def generate(prompts: list[str], model, temperature=0, responses_per_prompt=1):
        """
        Takes in a list of prompts shared across all processes,
        runs generation on rank0 only, and broadcasts results to all.
        Returns a list of outputs of length len(prompts) * responses_per_prompt.
        """
        if accelerator.is_main_process:
            # Sync current training weights to vLLM subprocess
            unwrapped = accelerator.unwrap_model(model)
            weight_tuples = [(n, p.detach().cpu()) for n, p in unwrapped.named_parameters()]
            cmd_queue.put(("update_weights", weight_tuples))
            result_queue.get()  # "OK"

            # Generate
            cmd_queue.put(("generate", (prompts, {
                "max_tokens": 1024,
                "temperature": temperature,
                "n": responses_per_prompt,
            })))
            outputs = result_queue.get()
        else:
            outputs = [None] * (len(prompts) * responses_per_prompt)

        # Sync all ranks and broadcast results
        accelerator.wait_for_everyone()
        if accelerator.num_processes > 1:
            outputs = broadcast_object_list(outputs, from_process=0)
        return outputs

    def tokenize_prompt_and_output(
        prompt_strs: list[str],
        output_strs: list[str],
        tokenizer: PreTrainedTokenizerBase,
    ) -> dict[str, torch.Tensor]:
        """
        INPUT: a list of prompts and outputs and a tokenizer
        OUTPUT: all of shape (len(prompt_strs), max_len - 1)
            - input_ids: input tokens
            - labels: output tokens (basically the inputs shifted 1 to the left)
            - response_mask: 1 indicates token that the model generated, 0 means prompt or padding
        """
        prompt_t = [tokenizer.encode(p) for p in prompt_strs]
        output_t = [tokenizer.encode(o) for o in output_strs]
        max_len = 0

        for i in range(len(prompt_t)):
            row_len = len(prompt_t[i]) + len(output_t[i])
            max_len = max(max_len, row_len)

        full = []
        for i in range(len(prompt_t)):
            padding_size = max_len - len(prompt_t[i]) - len(output_t[i])
            padding = [tokenizer.pad_token_id] * padding_size
            row = torch.tensor(prompt_t[i] + output_t[i] + padding, dtype=torch.long)
            full.append(row.unsqueeze(0))

        f2 = torch.cat(full)
        input_ids = f2[:, :-1]
        labels = f2[:, 1:]

        response_mask = torch.zeros(len(prompt_strs), max_len - 1)
        for i in range(len(prompt_t)):
            response_mask[
                i, len(prompt_t[i]) - 1 : len(prompt_t[i]) + len(output_t[i]) - 1
            ] = 1
        return {
            "input_ids": input_ids,
            "labels": labels,
            "response_mask": response_mask.bool(),
        }

    def get_response_log_probs(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        logits = model(input_ids).logits
        # F.log_softmaxのfused kernelで計算することでメモリ使用量を削減
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        logprobs_for_label = torch.gather(
            logprobs, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)
        return logprobs_for_label

    def eval_model(model, step):
        """Called from ALL processes. generate() handles internal synchronization."""
        val_prompts = val_dataset[:1000]["prompt"]

        eval_start = time.time()
        outputs = generate(val_prompts, model, temperature=0)
        eval_time = time.time() - eval_start

        if accelerator.is_main_process:
            correct = 0
            idx_correct = []
            idx_wrong = []

            for i in range(len(outputs)):
                correct_answer = val_dataset[i]["answer"]
                generated_answer = remove_boxed(last_boxed_only_string(outputs[i]))

                if generated_answer is not None and is_equiv(generated_answer, correct_answer):
                    correct += 1
                    idx_correct.append(i)
                else:
                    idx_wrong.append(i)

            accuracy = correct / len(outputs)
            print(f"step={step}, correct: {correct} / {len(outputs)} ({accuracy:.2%})")

            if not args.disable_wandb:
                wandb.log(
                    {
                        "eval/accuracy": accuracy,
                        "eval/correct": correct,
                        "eval/total": len(outputs),
                        "eval/time_seconds": eval_time,
                    },
                    step=step,
                )

    # Load model
    model_kwargs = dict(
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        use_cache=False,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model, optimizer = accelerator.prepare(model, optimizer)

    # Initial evaluation (all processes participate via generate()'s barriers)
    if accelerator.is_main_process:
        print("Starting initial evaluation...")
    eval_model(model, 0)
    model.train()

    num_processes = accelerator.num_processes
    proc_idx = accelerator.process_index

    for i in range(args.n_grpo_steps):
        step_start_time = time.time()

        # sample_indicesをrank0で生成してbroadcast
        # → 全プロセスが同じbatchを参照することでgenerate出力とanswerが一致する
        if accelerator.is_main_process:
            sample_indices = random.sample(
                range(len(train_dataset)), args.n_prompts_per_step
            )
        else:
            sample_indices = [None] * args.n_prompts_per_step

        if accelerator.num_processes > 1:
            sample_indices = broadcast_object_list(sample_indices, from_process=0)

        batch = train_dataset[sample_indices]

        # Generate rollouts (rank0のみ推論、結果をbroadcast)
        generation_start = time.time()
        outputs = generate(
            batch["prompt"],
            model=model,
            temperature=1,
            responses_per_prompt=args.group_size,
        )
        generation_time = time.time() - generation_start

        # Compute rewards and advantages over the full batch
        generated_answers = [remove_boxed(last_boxed_only_string(o)) for o in outputs]

        raw_reward = [
            a is not None and is_equiv(a, batch["answer"][j // args.group_size])
            for j, a in enumerate(generated_answers)
        ]
        raw_reward_tensor = torch.tensor(raw_reward, dtype=torch.float).reshape(
            (args.n_prompts_per_step, args.group_size)
        )
        means = raw_reward_tensor.mean(dim=-1).unsqueeze(1)
        advantages = (raw_reward_tensor - means).reshape(
            (args.n_prompts_per_step * args.group_size,)
        )

        # Reward / advantage statistics (logged from main process)
        train_accuracy = raw_reward_tensor.mean().item()
        reward_std = raw_reward_tensor.std().item()
        reward_max = raw_reward_tensor.max().item()
        reward_min = raw_reward_tensor.min().item()
        advantage_mean = advantages.mean().item()
        advantage_std = advantages.std().item()
        advantage_max = advantages.max().item()
        advantage_min = advantages.min().item()

        # Tokenize full batch (CPU)
        prompts_expanded = [x for x in batch["prompt"] for _ in range(args.group_size)]
        data = tokenize_prompt_and_output(prompts_expanded, outputs, tokenizer)
        input_ids = data["input_ids"]      # keep on CPU
        labels = data["labels"]            # keep on CPU
        response_mask = data["response_mask"]  # keep on CPU

        # Shard across processes: each GPU trains on its own slice
        total_rollouts = len(input_ids)
        per_proc = total_rollouts // num_processes
        start = proc_idx * per_proc
        end = start + per_proc

        input_ids     = input_ids[start:end]
        labels        = labels[start:end]
        response_mask = response_mask[start:end]
        advantages    = advantages[start:end]

        # Compute old log probs (on each process's local shard)
        with torch.inference_mode():
            old_logprobs_all = []
            for b in range(max(1, len(input_ids) // args.micro_batch_size)):
                idx = b * args.micro_batch_size
                end_idx = idx + args.micro_batch_size
                x = input_ids[idx:end_idx].to(device)
                y = labels[idx:end_idx].to(device)
                old_logprobs_all.append(get_response_log_probs(model, x, y).detach())
            old_logprobs_all = torch.cat(old_logprobs_all, dim=0)
            assert old_logprobs_all.shape == labels.shape

        # Training epochs
        epoch_grad_norms = []
        epoch_policy_ratios = []
        epoch_kl_divs = []

        training_start = time.time()
        for epoch in range(args.epochs_per_step):
            batches = range(max(1, len(input_ids) // args.micro_batch_size))
            if accelerator.is_main_process:
                batches = tqdm(batches, desc=f"Step {i + 1}/{args.n_grpo_steps}")

            for b in batches:
                idx = b * args.micro_batch_size
                end_idx = idx + args.micro_batch_size
                x = input_ids[idx:end_idx].to(device)
                y = labels[idx:end_idx].to(device)
                mask = response_mask[idx:end_idx].to(device)
                micro_batch_adv = advantages[idx:end_idx].unsqueeze(-1).to(device)

                policy_logprobs = get_response_log_probs(model, x, y)
                old_logprobs = old_logprobs_all[idx:end_idx]

                ratio = torch.exp(policy_logprobs - old_logprobs)
                per_token_loss = -ratio * micro_batch_adv

                with torch.no_grad():
                    # KL(old || new) ≈ (ratio - 1) - log(ratio)
                    approx_kl = ((ratio - 1) - torch.log(ratio)) * mask
                    approx_kl_mean = approx_kl.sum() / mask.sum()
                    epoch_kl_divs.append(approx_kl_mean.item())

                    policy_ratio_mean = (ratio * mask).sum() / mask.sum()
                    epoch_policy_ratios.append(policy_ratio_mean.item())

                masked_loss = per_token_loss * mask
                denom = mask.sum(dim=-1).clamp_min(1)
                loss_per_prompt = masked_loss.sum(dim=-1) / denom
                loss = loss_per_prompt.mean() / args.gradient_accumulation_steps

                accelerator.backward(loss)

                if (b + 1) % args.gradient_accumulation_steps == 0:
                    # accelerator経由でDDP環境でも正しくgradをclip
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    epoch_grad_norms.append(grad_norm.item())
                    optimizer.step()
                    optimizer.zero_grad()

        training_time = time.time() - training_start
        step_time = time.time() - step_start_time

        # Logging (main process only)
        if accelerator.is_main_process:
            mean_grad_norm = (
                sum(epoch_grad_norms) / len(epoch_grad_norms) if epoch_grad_norms else 0
            )
            mean_policy_ratio = (
                sum(epoch_policy_ratios) / len(epoch_policy_ratios)
                if epoch_policy_ratios
                else 0
            )
            mean_kl = sum(epoch_kl_divs) / len(epoch_kl_divs) if epoch_kl_divs else 0

            total_tokens = response_mask.sum().item()
            tokens_per_sec = total_tokens / generation_time if generation_time > 0 else 0
            avg_generation_len = int(response_mask.sum(dim=-1).float().mean().item())

            if not args.disable_wandb:
                wandb.log(
                    {
                        "train/accuracy": train_accuracy,
                        "train/reward_mean": train_accuracy,
                        "train/reward_std": reward_std,
                        "train/reward_max": reward_max,
                        "train/reward_min": reward_min,
                        "train/advantage_mean": advantage_mean,
                        "train/advantage_std": advantage_std,
                        "train/advantage_max": advantage_max,
                        "train/advantage_min": advantage_min,
                        "train/grad_norm": mean_grad_norm,
                        "train/policy_ratio": mean_policy_ratio,
                        "train/approx_kl": mean_kl,
                        "train/avg_gen_length": avg_generation_len,
                        "time/step_time": step_time,
                        "time/generation_time": generation_time,
                        "time/training_time": training_time,
                        "time/tokens_per_sec": tokens_per_sec,
                    },
                    step=i + 1,
                )

            print(
                f"Step {i + 1}/{args.n_grpo_steps} | Train Acc: {train_accuracy:.2%} | KL: {mean_kl:.4f} | Time: {step_time:.1f}s"
            )

        if (i + 1) % 5 == 0:
            eval_model(model, i + 1)  # all processes participate via generate()'s barriers

    if accelerator.is_main_process:
        if not args.disable_wandb:
            wandb.finish()
        cmd_queue.put("STOP")
        vllm_proc.join(timeout=10)


if __name__ == "__main__":
    main()
