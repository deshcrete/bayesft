"""GPU memory management and vLLM utilities."""

import gc
import subprocess
import sys

import torch


def cleanup_vram():
    """Free Python-side GPU memory caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def kill_gpu_processes():
    """Kill all processes holding CUDA/GPU memory."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("nvidia-smi not found or no GPU available.")
        return

    pids = [int(p.strip()) for p in result.stdout.strip().splitlines() if p.strip()]
    if not pids:
        print("No processes using GPU memory.")
        return

    print(f"Found {len(pids)} GPU process(es): {pids}")
    for pid in pids:
        try:
            subprocess.run(["kill", "-9", str(pid)], check=True)
            print(f"  Killed PID {pid}")
        except subprocess.CalledProcessError:
            print(f"  Failed to kill PID {pid}")


def load_vllm(model_name, gpu_memory_utilization=0.6, max_model_len=2048, enforce_eager=False):
    """Load a vLLM model instance."""
    from vllm import LLM

    return LLM(
        model=model_name,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=enforce_eager,
    )


def unload_vllm(llm):
    """Unload a vLLM model and free VRAM."""
    from vllm.distributed.parallel_state import destroy_model_parallel

    del llm
    destroy_model_parallel()
    cleanup_vram()
    print("vLLM model unloaded and VRAM freed.")


def vllm_generate(llm, prompts, temperature=0.7, max_tokens=256):
    """Generate completions using a vLLM model."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]
