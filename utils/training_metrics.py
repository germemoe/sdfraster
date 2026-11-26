"""Windowed training metrics without additional model queries."""
import json
from pathlib import Path
import torch


@torch.no_grad()
def gradient_norm_statistics(gradients, active=True):
    """Detached per-sample statistics, without querying the SDF again."""
    names = ("eikonal_grad_norm_mean", "eikonal_grad_norm_mean_square",
             "eikonal_grad_norm_fraction_09_11")
    if not active or gradients is None or gradients.numel() == 0:
        return dict.fromkeys(names)
    norms = gradients.detach().float().norm(dim=-1)
    return {
        names[0]: norms.mean(),
        names[1]: norms.square().mean(),
        names[2]: ((norms >= .9) & (norms <= 1.1)).float().mean(),
    }


def optimizer_learning_rates(tet_optimizer, vertex_stepped=True):
    """Read CPU LR scalars only. No parameter/data queries or RNG operations."""
    result = {}
    for optimizer_name in ("optim", "net_optim", "vertex_optim"):
        if optimizer_name == "vertex_optim" and not vertex_stepped:
            continue
        optimizer = getattr(tet_optimizer, optimizer_name, None)
        if optimizer is not None:
            for group_index, group in enumerate(optimizer.param_groups):
                result[f"{optimizer_name}_{group.get('name', group_index)}"] = float(group["lr"])
    return result


class TrainingMetricWindow:
    def __init__(self, path, interval):
        self.path = Path(path)
        self.interval = int(interval)
        self.reset()

    def reset(self):
        self.sums = None
        self.counts = None
        self.names = None
        self.steps = 0
        self.first_step = None
        self.first_vertices = None
        self.lr_ranges = {}
        self.applied_lr_ranges = {}
        self.applied_lr_counts = {}
        self.vertex_updates = 0
        self.has_applied_lr = False
        self.eps_range = None

    @torch.no_grad()
    def add(self, step, values, eligible, vertices, tetrahedra, learning_rates, epsilon,
            applied_learning_rates=None, vertex_optimizer_stepped=None):
        if self.interval <= 0:
            return
        names = list(values)
        reference = next(value for value in values.values() if value is not None).detach().float()
        vector = torch.stack([reference.new_zeros(()) if values[name] is None
                              else values[name].detach().float().reshape(()) for name in names])
        mask = vector.new_tensor([float(values[name] is not None and eligible.get(name, True))
                                  for name in names])
        if self.sums is None:
            self.names = names
            self.sums = torch.zeros_like(vector)
            self.counts = torch.zeros_like(vector)
            self.first_step = step
            self.first_vertices = vertices
        elif names != self.names:
            raise ValueError("Metric keys changed within a window")
        self.sums.add_(torch.where(mask.bool(), vector, 0))
        self.counts.add_(mask)
        self.steps += 1
        self.last_step, self.last_vertices, self.last_tetrahedra = step, vertices, tetrahedra
        for name, value in learning_rates.items():
            value = float(value)
            bounds = self.lr_ranges.setdefault(name, [value, value])
            bounds[0], bounds[1] = min(bounds[0], value), max(bounds[1], value)
        if applied_learning_rates is not None:
            self.has_applied_lr = True
            for name, value in applied_learning_rates.items():
                value = float(value)
                bounds = self.applied_lr_ranges.setdefault(name, [value, value])
                bounds[0], bounds[1] = min(bounds[0], value), max(bounds[1], value)
                self.applied_lr_counts[name] = self.applied_lr_counts.get(name, 0) + 1
            self.vertex_updates += int(bool(vertex_optimizer_stepped))
        epsilon = float(epsilon)
        if self.eps_range is None:
            self.eps_range = [float(epsilon), float(epsilon)]
        else:
            self.eps_range[0] = min(self.eps_range[0], epsilon)
            self.eps_range[1] = max(self.eps_range[1], epsilon)

    def flush(self, reason):
        if not self.steps:
            return None
        sums, counts = torch.stack([self.sums, self.counts]).cpu().tolist()
        means = {name: sums[i] / counts[i] if counts[i] else None for i, name in enumerate(self.names)}
        record = {
            "first_completed_step": self.first_step, "last_completed_step": self.last_step,
            "n_steps": self.steps, "means": means,
            "valid_steps": dict(zip(self.names, map(int, counts))),
            "vertices_start": self.first_vertices, "vertices_end": self.last_vertices,
            "tetrahedra_end": self.last_tetrahedra, "learning_rate_ranges": self.lr_ranges,
            "world_epsilon_range": self.eps_range, "flush_reason": reason,
            "scope": "Training samples; geometric convergence requires checkpoint CD/F1",
        }
        if self.has_applied_lr:
            record.update(
                applied_learning_rate_ranges=self.applied_lr_ranges,
                applied_learning_rate_valid_steps=self.applied_lr_counts,
                vertex_optimizer_updates=self.vertex_updates,
                learning_rate_semantics="legacy ranges are next-step LR; applied ranges include only actual optimizer updates",
            )
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        self.reset()
        return record

    def due(self):
        return self.interval > 0 and self.steps >= self.interval
