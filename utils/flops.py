from typing import List, Optional, Tuple
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import wandb


def sds(shape: Tuple[int, ...], dtype=jnp.float32) -> jax.ShapeDtypeStruct:
    """abstract array for lowering (no data / device memory needed)"""
    return jax.ShapeDtypeStruct(tuple(shape), dtype)


def compiled_flops(jitted_fn, *args, **kwargs) -> float:
    """XLA-counted FLOPs of one call of a jitted function (FMA = 2 FLOPs)

    note: XLA counts a lax.scan / while body only once, so scans must be lowered unrolled
    note: Lowered.cost_analysis() returns None on GPU, so the compiled executable is used
    note: compiled for CPU even on GPU machines; GPU cost analysis misses matmuls inside
          cuBLAS/Triton fusions (~2/3 of the true count for MLPs), CPU matches 6*N*samples
    """
    with jax.default_device(jax.devices('cpu')[0]):
        cost = jitted_fn.lower(*args, **kwargs).compile().cost_analysis()
    if isinstance(cost, (list, tuple)):
        cost = cost[0]
    return float(cost['flops'])


def num_params(*modules) -> int:
    """number of trainable parameters"""
    return sum(int(x.size) for m in modules if m is not None for x in jax.tree_util.tree_leaves(nnx.state(m, nnx.Param)))


class FlopsTracker:
    """accumulates training FLOPs per stage"""
    def __init__(self, single_model_tag: Optional[str] = None):
        # single_model_tag: stages tagged with it (or untagged) form the single-model total
        self.single_model_tag = single_model_tag
        self.entries: List[dict] = []

    def add(self, stage: str, per_step: float, steps: int, samples_per_step: int = 0, params: int = 0, tag: Optional[str] = None):
        """add a training stage: per_step FLOPs x steps"""
        self.entries.append(dict(stage=stage, per_step=per_step, steps=steps, total=per_step * steps,
                                 estimate=6 * params * samples_per_step * steps, tag=tag))

    def add_once(self, stage: str, flops: float, tag: Optional[str] = None):
        """add a one-off computation (e.g. prior's eigenspace precompute)"""
        self.entries.append(dict(stage=stage, per_step=flops, steps=1, total=flops, estimate=0, tag=tag))

    def total(self) -> float:
        return sum(e['total'] for e in self.entries)

    def single_model_total(self) -> float:
        return sum(e['total'] for e in self.entries if e['tag'] is None or e['tag'] == self.single_model_tag)

    def print_summary(self):
        total = self.total()
        print("\n" + "=" * 92)
        print("Training FLOPs (XLA cost analysis; forward + backward + optimizer)")
        print(f"{'stage':<30}{'per-step':>12}{'steps':>12}{'total':>12}{'share':>8}{'6*N*samples':>14}")
        for e in self.entries:
            share = e['total'] / total if total > 0 else 0.0
            estimate = f"{e['estimate']:.3e}" if e['estimate'] else '-'
            print(f"{e['stage']:<30}{e['per_step']:>12.3e}{e['steps']:>12,d}{e['total']:>12.3e}{share:>8.1%}{estimate:>14}")
        print("-" * 92)
        print(f"{'TOTAL (as run)':<54}{total:>12.3e}")
        if self.single_model_tag is not None:
            print(f"{'TOTAL (single model, ' + self.single_model_tag + ')':<54}{self.single_model_total():>12.3e}")
        print("=" * 92)

    def log_wandb(self):
        if wandb.run is None:
            return
        wandb.run.summary['train_flops_total'] = self.total()
        wandb.run.summary['train_flops_single_model'] = self.single_model_total()
        for e in self.entries:
            wandb.run.summary[f"train_flops/{e['stage']}"] = e['total']
