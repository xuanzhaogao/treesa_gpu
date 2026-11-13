"""treesa_gpu: simulated-annealing contraction-order optimization (TreeSA) on the GPU.

Thousands of annealing chains of the Julia/omeco TreeSA move set run in parallel on the GPU, for
tensor networks whose indices all have dimension 2 (any number of labels). Energies: tc/sc as in omeco,
or a modeled GPU contraction time; optional slicing inside the annealer.

    import treesa_gpu
    tree, info = treesa_gpu.optimize(ixs, iy, nchains=1024, niters=20, sc_target=30, mode="tc")

Modules: ``driver`` (optimize, tree <-> binary format), ``trees`` (tc/sc, GPU-time energies).
The binary is built with ``make`` in the repository root (see README.md).
"""
from .driver import optimize, optimize_multi, runner_json, julia_betas, greedy_tree, dict_to_children, children_to_dict
from . import trees

__all__ = ["optimize", "optimize_multi", "runner_json", "julia_betas", "greedy_tree", "dict_to_children", "children_to_dict", "trees"]
__version__ = "0.1.0"
