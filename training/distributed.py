"""Training on the GPUs of one machine under torchrun, for either head:

    torchrun --standalone --nproc_per_node 8 -m training.train --config <config> ...

Each process is one rank. The step plan (training/batching.py) gives the same optimizer steps on any GPU count that
divides a step's microbatches: rank r runs the step's microbatches r, r + world, ... and the gradients are averaged
over the trainable parameters only (DDP), weighted so the average is the per-row mean over the whole step
(training/train.py). The frozen decoder is never synchronized. Dropout is drawn per rank, so the weights match a
one-GPU run's only up to dropout; with dropout off they match it up to the rounding of sums taken in another order.

One machine only: a world that spans several machines is refused. Without torchrun the trainer is one process.
"""

from __future__ import annotations

import datetime
import inspect
import os
from typing import Any

import torch
import torch.distributed as dist


class Process:
    """This process's place in the run: rank, world size and device; the process group when there are several."""

    def __init__(self) -> None:
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "0" if self.world > 1 else "1"))
        if self.world != local_world:
            raise SystemExit(
                f"{self.world} processes, {local_world or 'an unknown number'} on this machine: the trainer runs on "
                "one machine; use torchrun --standalone"
            )
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")
        if self.world > 1:
            if self.device.type == "cuda":
                dist.init_process_group(
                    "nccl", timeout=datetime.timedelta(hours=2), device_id=self.device
                )
            else:  # the tests: gloo on the CPU
                dist.init_process_group("gloo", timeout=datetime.timedelta(minutes=30))

    @property
    def primary(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.world > 1:
            dist.barrier()

    def all_reduce(self, tensor: torch.Tensor, op: Any = None) -> torch.Tensor:
        if self.world > 1:
            dist.all_reduce(tensor, op=op or dist.ReduceOp.SUM)
        return tensor

    def broadcast_object(self, value: Any) -> Any:
        if self.world == 1:
            return value
        box = [value if self.primary else None]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def close(self) -> None:
        if self.world > 1:
            dist.destroy_process_group()


def wrap(model: torch.nn.Module, trainable: set[str], device: torch.device, bucket_mb: int) -> Any:
    """`model` under DDP, synchronizing only the parameters named in `trainable`: every other parameter and buffer
    (the frozen decoder) is left out of DDP's start-up broadcast and its gradient buckets."""
    ignored = sorted(
        (
            {n for n, _p in model.named_parameters()}
            | {n for n, _b in model.named_buffers()}
            | set(model.state_dict().keys())
        )
        - trainable
    )
    signature = inspect.signature(torch.nn.parallel.DistributedDataParallel.__init__).parameters
    no_buffer_sync = (
        {"forward_sync_buffers": False}
        if "forward_sync_buffers" in signature
        else {"broadcast_buffers": False}
    )
    ddp = torch.nn.parallel.DistributedDataParallel
    ddp._set_params_and_buffers_to_ignore_for_model(model, ignored)  # noqa: SLF001 (torch's hook for this)
    return ddp(
        model,
        device_ids=[device.index] if device.type == "cuda" else None,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        bucket_cap_mb=bucket_mb,
        **no_buffer_sync,
    )
