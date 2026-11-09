import os
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


def dp_tp_ranks(rank: int, tp_size: int) -> tuple[int, int]:
    return rank // tp_size, rank % tp_size


@dataclass(frozen=True)
class ParallelContext:
    """Rank layout over a (dp, tp) mesh with tp innermost.

    Ranks sharing a dp_rank form a TP group and must consume identical batches.
    With tp_size == 1 the dp group is the default world group and dp_* equal the global values.
    """

    rank: int = 0
    world_size: int = 1
    dp_rank: int = 0
    dp_size: int = 1
    tp_rank: int = 0
    tp_size: int = 1
    dp_group: Any = None
    tp_group: Any = None
    mesh: DeviceMesh | None = None

    @classmethod
    def build(cls, *, tp_size: int = 1, device_type: str = "cpu") -> "ParallelContext":
        if not (dist.is_available() and dist.is_initialized()):
            if tp_size != 1:
                raise ValueError(f"tp_size={tp_size} requires a distributed launch")
            return cls(
                rank=int(os.getenv("RANK", "0")),
                world_size=int(os.getenv("WORLD_SIZE", "1")),
            )

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if tp_size < 1 or world_size % tp_size != 0:
            raise ValueError(f"world_size={world_size} is not divisible by tp_size={tp_size}")
        if tp_size == 1:
            return cls(rank=rank, world_size=world_size, dp_rank=rank, dp_size=world_size)

        dp_size = world_size // tp_size
        mesh = init_device_mesh(device_type, (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        dp_rank, tp_rank = dp_tp_ranks(rank, tp_size)
        return cls(
            rank=rank,
            world_size=world_size,
            dp_rank=dp_rank,
            dp_size=dp_size,
            tp_rank=tp_rank,
            tp_size=tp_size,
            dp_group=mesh.get_group("dp"),
            tp_group=mesh.get_group("tp"),
            mesh=mesh,
        )

    @property
    def is_main(self) -> bool:
        return self.rank == 0
