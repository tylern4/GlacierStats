
try:
    import cupy as cp
    import cudf
    from numba import cuda
    from .GlacierStatsGPU import *
except (ImportError, ModuleNotFoundError) as e:
    from .GlacierStats import *
