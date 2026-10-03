"""Windows E2E fixture standing in for PyTorch.

This is NOT PyTorch. It exists so that the real hachidori_worker.py can import
"torch", ask whether CUDA is usable and run, without a production-scale
dependency. It reports no usable CUDA device, on every host, deterministically.
"""

__version__ = "2.11.0+e2e-fixture"


class _Version:
    cuda = None


version = _Version()


class _Cuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def device_count():
        return 0

    @staticmethod
    def synchronize():
        return None


cuda = _Cuda()
