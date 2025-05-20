import os
import types
import sys

if os.environ.get("READTHEDOCS") == "True":
    # ------------------------
    # Mock torch
    # ------------------------
    torch = types.SimpleNamespace()

    def dummy(*args, **kwargs):
        return dummy

    torch.tensor = dummy
    torch.exp = dummy
    torch.zeros_like = dummy
    torch.dot = dummy
    torch.abs = dummy
    torch.pi = 3.14159
    torch.quasirandom = types.SimpleNamespace(SobolEngine=dummy)

    torch.nn = types.SimpleNamespace(
        Module=object,
        Linear=dummy,
        Conv2d=dummy,
        ReLU=dummy,
        Sequential=dummy,
        functional=types.SimpleNamespace(
            relu=dummy,
            sigmoid=dummy,
            softmax=dummy,
            mse_loss=dummy,
            cross_entropy=dummy,
            interpolate=dummy
        )
    )
    torch.nn.functional = torch.nn.functional  # alias

    # ------------------------
    # Mock torchvision
    # ------------------------
    torchvision = types.SimpleNamespace()
    torchvision.transforms = types.SimpleNamespace(
        functional=types.SimpleNamespace(rotate=dummy),
        InterpolationMode=dummy
    )

    # Inject into sys.modules so imports work
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = torch.nn
    sys.modules["torch.nn.functional"] = torch.nn.functional
    sys.modules["torch.quasirandom"] = torch.quasirandom
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.transforms"] = torchvision.transforms
    sys.modules["torchvision.transforms.functional"] = torchvision.transforms.functional
else:
    import torch
    import torch.nn.functional as F
    from torch.quasirandom import SobolEngine
    from torch import nn
    from torchvision.transforms.functional import rotate
    from torchvision.transforms import InterpolationMode
