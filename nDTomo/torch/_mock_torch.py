import types
import os
import torch

if os.environ.get("READTHEDOCS") == "True":
    dummy = lambda *args, **kwargs: None

    class DummyTensor(types.SimpleNamespace):
        def cuda(self, *args, **kwargs):
            return self
        def to(self, *args, **kwargs):
            return self
        def __call__(self, *args, **kwargs):
            return self

    # Mock main torch namespace
    torch = types.SimpleNamespace(
        Tensor=DummyTensor,
        cuda=dummy,
        nn=types.SimpleNamespace(
            Module=object,
            functional=types.SimpleNamespace(),
            Conv2d=DummyTensor,
            Linear=DummyTensor,
            ReLU=DummyTensor,
            Sequential=lambda *args, **kwargs: DummyTensor(),
            MSELoss=lambda *args, **kwargs: DummyTensor(),
        ),
        from_numpy=lambda x: DummyTensor(),
        zeros=lambda *args, **kwargs: DummyTensor(),
        ones=lambda *args, **kwargs: DummyTensor(),
        eye=lambda *args, **kwargs: DummyTensor(),
        randn=lambda *args, **kwargs: DummyTensor(),
        arange=lambda *args, **kwargs: DummyTensor(),
        float32="float32",
        float64="float64",
        device=lambda x: x,
    )

    # Replace sys.modules entry
    import sys
    sys.modules['torch'] = torch
    sys.modules['torch.nn'] = torch.nn
    sys.modules['torch.nn.functional'] = torch.nn.functional
    sys.modules['torch.cuda'] = types.SimpleNamespace(is_available=lambda: False)
    sys.modules['torchvision'] = types.SimpleNamespace(transforms=types.SimpleNamespace())
    sys.modules['torchvision.transforms.functional'] = types.SimpleNamespace(rotate=lambda *a, **k: DummyTensor())