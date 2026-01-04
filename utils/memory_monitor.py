"""GPU Memory Monitor for debugging OOM issues.

Usage:
    from utils.memory_monitor import MemoryMonitor

    monitor = MemoryMonitor(log_path="memory.log")

    # In training loop:
    monitor.log(iteration, "before_render")
    render_pkg = render(...)
    monitor.log(iteration, "after_render")

    # Print summary
    monitor.summary()
"""

import torch
import gc
from pathlib import Path
from datetime import datetime


def get_gpu_memory_info():
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return {}

    allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    reserved = torch.cuda.memory_reserved() / 1024**2    # MB
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2

    # Get total GPU memory
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    free = total - reserved

    return {
        'allocated_mb': allocated,
        'reserved_mb': reserved,
        'max_allocated_mb': max_allocated,
        'total_mb': total,
        'free_mb': free,
    }


def print_gpu_memory(prefix=""):
    """Quick print of current GPU memory."""
    info = get_gpu_memory_info()
    print(f"{prefix} GPU: {info['allocated_mb']:.0f}MB allocated, "
          f"{info['reserved_mb']:.0f}MB reserved, "
          f"{info['free_mb']:.0f}MB free / {info['total_mb']:.0f}MB total")


def get_tensor_memory_usage():
    """Get memory usage by tensor type (for debugging)."""
    tensor_info = {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                size_mb = obj.element_size() * obj.nelement() / 1024**2
                dtype = str(obj.dtype)
                shape = tuple(obj.shape)
                key = (dtype, shape)
                if key not in tensor_info:
                    tensor_info[key] = {'count': 0, 'size_mb': 0}
                tensor_info[key]['count'] += 1
                tensor_info[key]['size_mb'] += size_mb
        except:
            pass
    return tensor_info


def print_large_tensors(min_size_mb=10):
    """Print tensors larger than min_size_mb."""
    tensor_info = get_tensor_memory_usage()
    sorted_tensors = sorted(tensor_info.items(), key=lambda x: -x[1]['size_mb'])

    print(f"\n=== Large CUDA Tensors (>{min_size_mb}MB) ===")
    total = 0
    for (dtype, shape), info in sorted_tensors:
        if info['size_mb'] >= min_size_mb:
            print(f"  {dtype:15} {str(shape):30} x{info['count']:3} = {info['size_mb']:8.1f} MB")
            total += info['size_mb']
    print(f"  {'Total':47} = {total:8.1f} MB")


class MemoryMonitor:
    """Continuous memory monitoring with logging."""

    def __init__(self, log_path=None, print_interval=100):
        self.log_path = Path(log_path) if log_path else None
        self.print_interval = print_interval
        self.records = []
        self.peak_allocated = 0
        self.peak_reserved = 0
        self.peak_info = None

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, 'w') as f:
                f.write("timestamp,iteration,tag,allocated_mb,reserved_mb,max_allocated_mb,free_mb,n_verts,n_tets\n")

    def log(self, iteration, tag, model=None, print_now=False):
        """Log current memory state."""
        info = get_gpu_memory_info()

        # Track peaks
        if info['allocated_mb'] > self.peak_allocated:
            self.peak_allocated = info['allocated_mb']
            self.peak_info = (iteration, tag)
        if info['reserved_mb'] > self.peak_reserved:
            self.peak_reserved = info['reserved_mb']

        # Get model size info
        n_verts = model.vertices.shape[0] if model is not None else 0
        n_tets = model.indices.shape[0] if model is not None else 0

        record = {
            'timestamp': datetime.now().isoformat(),
            'iteration': iteration,
            'tag': tag,
            'n_verts': n_verts,
            'n_tets': n_tets,
            **info
        }
        self.records.append(record)

        # Write to log file
        if self.log_path:
            with open(self.log_path, 'a') as f:
                f.write(f"{record['timestamp']},{iteration},{tag},"
                       f"{info['allocated_mb']:.1f},{info['reserved_mb']:.1f},"
                       f"{info['max_allocated_mb']:.1f},{info['free_mb']:.1f},"
                       f"{n_verts},{n_tets}\n")

        # Print periodically or on request
        if print_now or (iteration % self.print_interval == 0):
            print(f"[{iteration:6d}] {tag:20s} | "
                  f"Alloc: {info['allocated_mb']:6.0f}MB | "
                  f"Rsrvd: {info['reserved_mb']:6.0f}MB | "
                  f"Free: {info['free_mb']:6.0f}MB | "
                  f"#V={n_verts} #T={n_tets}")

        return info

    def log_delta(self, iteration, tag, model=None):
        """Log and show memory change since last log."""
        info = self.log(iteration, tag, model)

        if len(self.records) >= 2:
            prev = self.records[-2]
            delta_alloc = info['allocated_mb'] - prev['allocated_mb']
            delta_rsrvd = info['reserved_mb'] - prev['reserved_mb']

            if abs(delta_alloc) > 10 or abs(delta_rsrvd) > 10:
                sign_a = '+' if delta_alloc >= 0 else ''
                sign_r = '+' if delta_rsrvd >= 0 else ''
                print(f"  ↳ Δ from '{prev['tag']}': "
                      f"Alloc {sign_a}{delta_alloc:.0f}MB, "
                      f"Rsrvd {sign_r}{delta_rsrvd:.0f}MB")

        return info

    def summary(self):
        """Print memory usage summary."""
        if not self.records:
            print("No memory records.")
            return

        print("\n" + "="*60)
        print("MEMORY USAGE SUMMARY")
        print("="*60)

        # Peak usage
        print(f"\nPeak Allocated: {self.peak_allocated:.0f} MB")
        if self.peak_info:
            print(f"  at iteration {self.peak_info[0]}, tag: '{self.peak_info[1]}'")
        print(f"Peak Reserved:  {self.peak_reserved:.0f} MB")

        # Final state
        final = self.records[-1]
        print(f"\nFinal State:")
        print(f"  Allocated: {final['allocated_mb']:.0f} MB")
        print(f"  Reserved:  {final['reserved_mb']:.0f} MB")
        print(f"  Free:      {final['free_mb']:.0f} MB")

        # Group by tag and find which operations use most memory
        tag_stats = {}
        for r in self.records:
            tag = r['tag']
            if tag not in tag_stats:
                tag_stats[tag] = {'max_alloc': 0, 'max_rsrvd': 0, 'count': 0}
            tag_stats[tag]['max_alloc'] = max(tag_stats[tag]['max_alloc'], r['allocated_mb'])
            tag_stats[tag]['max_rsrvd'] = max(tag_stats[tag]['max_rsrvd'], r['reserved_mb'])
            tag_stats[tag]['count'] += 1

        print(f"\nMemory by Tag:")
        sorted_tags = sorted(tag_stats.items(), key=lambda x: -x[1]['max_alloc'])
        for tag, stats in sorted_tags[:10]:
            print(f"  {tag:30s} | MaxAlloc: {stats['max_alloc']:6.0f}MB | "
                  f"MaxRsrvd: {stats['max_rsrvd']:6.0f}MB | Count: {stats['count']}")

        if self.log_path:
            print(f"\nFull log saved to: {self.log_path}")

        print("="*60)


# Quick helper for one-off debugging
def debug_memory_at_oom():
    """Call this when you hit OOM to see what's using memory."""
    print("\n" + "="*60)
    print("OOM DEBUG INFO")
    print("="*60)

    print_gpu_memory("Current state:")
    print_large_tensors(min_size_mb=5)

    # Force cleanup and check again
    gc.collect()
    torch.cuda.empty_cache()

    print("\nAfter cleanup:")
    print_gpu_memory("")


# Example usage in train.py:
"""
from utils.memory_monitor import MemoryMonitor, print_gpu_memory

# At start
monitor = MemoryMonitor(log_path=args.output_path / "memory.log", print_interval=500)

# In training loop
for iteration in range(...):
    monitor.log(iteration, "loop_start", model)

    render_pkg = render(...)
    monitor.log_delta(iteration, "after_render", model)

    loss.backward()
    monitor.log_delta(iteration, "after_backward", model)

    if do_densification:
        monitor.log(iteration, "before_densify", model)
        # ... densification
        monitor.log_delta(iteration, "after_densify", model)

# At end
monitor.summary()
"""
