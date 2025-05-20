import types
import os
import sys

if os.environ.get("READTHEDOCS") == "True":

    dummy = lambda *args, **kwargs: DummyTensor()

    class DummyTensor:
        def cuda(self, *args, **kwargs): return self
        def to(self, *args, **kwargs): return self
        def __call__(self, *args, **kwargs): return self
        def __getattr__(self, name): return self
        def __getitem__(self, key): return self

    # Mock torchvision and its transforms
    torchvision = types.SimpleNamespace()
    torchvision.transforms = types.SimpleNamespace(
        InterpolationMode=types.SimpleNamespace(BILINEAR="bilinear")
    )
    torchvision.transforms.functional = types.SimpleNamespace(rotate=dummy)

    # Mock torch
    torch = types.SimpleNamespace(
        Tensor=DummyTensor,
        FloatTensor=DummyTensor, 
        cuda=types.SimpleNamespace(is_available=lambda: False),
        nn=types.SimpleNamespace(
            Module=object,
            functional=types.SimpleNamespace(),
            Conv2d=DummyTensor,
            Linear=DummyTensor,
            ReLU=DummyTensor,
            Sequential=lambda *args, **kwargs: DummyTensor(),
            MSELoss=lambda *args, **kwargs: DummyTensor(),
        ),
        from_numpy=dummy,
        zeros=dummy,
        ones=dummy,
        eye=dummy,
        randn=dummy,
        arange=dummy,
        float32="float32",
        float64="float64",
        device=lambda x: x,
        quasirandom=types.SimpleNamespace(SobolEngine=lambda *a, **k: DummyTensor()),
    )

    # Register mocks
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = torch.nn
    sys.modules["torch.nn.functional"] = torch.nn.functional
    sys.modules["torch.cuda"] = torch.cuda
    sys.modules["torch.quasirandom"] = torch.quasirandom
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.transforms"] = torchvision.transforms
    sys.modules["torchvision.transforms.functional"] = torchvision.transforms.functional
