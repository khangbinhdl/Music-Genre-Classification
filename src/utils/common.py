import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set deterministic seed for reproducible experiments."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            # Some operators may not have deterministic implementations.
            pass
    except Exception:
        # Torch is optional for the ML-only training/inference pipeline.
        pass
