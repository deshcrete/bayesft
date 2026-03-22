#!/usr/bin/env python3
"""Kill all processes holding CUDA/GPU memory and free VRAM."""

import subprocess
import sys


def get_gpu_pids():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("nvidia-smi not found or no GPU available.")
        sys.exit(1)
    pids = [int(p.strip()) for p in result.stdout.strip().splitlines() if p.strip()]
    return pids


def main():
    pids = get_gpu_pids()
    if not pids:
        print("No processes using GPU memory.")
        return

    print(f"Found {len(pids)} GPU process(es): {pids}")
    for pid in pids:
        try:
            subprocess.run(["kill", "-9", str(pid)], check=True)
            print(f"  Killed PID {pid}")
        except subprocess.CalledProcessError:
            print(f"  Failed to kill PID {pid} (may need sudo or already gone)")

    # Verify
    remaining = get_gpu_pids()
    if remaining:
        print(f"\nStill running: {remaining} — try running with sudo.")
    else:
        print("\nVRAM cleared.")


if __name__ == "__main__":
    main()
