"""Grip classifier from the wrist camera: which grip does the object in view afford?

Frozen pretrained image features + a small logistic-regression head, trained
on your own labeled frames. Probabilities are what the fusion gate needs, so
the head is fitted with proper (cross-validated) probability calibration.

Backbones (all frozen):
  "resnet18"  torchvision ImageNet ResNet-18, 512-d, fast on a laptop CPU
  "vjepa2"    the same V-JEPA 2 encoder as ShiftBench (shiftbench.encoders), 1024-d

Labels are grip names, e.g. GRIPS below. Record frames with record.py
(objects held up to the wrist camera at grasping distance, several angles,
your normal lighting), label each clip once by the object's grip, and train.
"""
from __future__ import annotations

import numpy as np

GRIPS = ["power", "pinch", "tripod", "lateral"]   # class 1..4; 0 is rest (EMG only)


class FrameFeatures:
    def __init__(self, backbone: str = "resnet18", device: str | None = None):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = backbone
        if backbone == "resnet18":
            import torchvision
            w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            m = torchvision.models.resnet18(weights=w)
            m.fc = torch.nn.Identity()
            self.model = m.eval().to(self.device)
            self.tf = w.transforms()
        elif backbone == "vjepa2":
            from shiftbench.encoders import VJEPA2Encoder
            self.enc = VJEPA2Encoder(device=self.device)
        else:
            raise ValueError(backbone)

    def __call__(self, frames) -> np.ndarray:
        """frames: (n, H, W, 3) uint8 RGB."""
        if self.backbone == "vjepa2":
            return self.enc.encode(frames)
        t = self.torch
        x = t.as_tensor(np.asarray(frames)).permute(0, 3, 1, 2)
        out = []
        with t.no_grad():
            for i in range(0, len(x), 64):
                b = self.tf(x[i:i + 64]).to(self.device)
                out.append(self.model(b).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


class GripClassifier:
    """Logistic regression on frozen features, probabilities calibrated by CV."""

    def __init__(self, C: float = 1.0):
        self.C = C
        self.model = None

    def fit(self, X, y_grip, groups=None):
        """X: features; y_grip: 1..K. groups: clip ids, so CV never splits a clip."""
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        base = make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=2000))
        cv = GroupKFold(5).split(X, y_grip, groups) if groups is not None else 5
        self.model = CalibratedClassifierCV(base, method="sigmoid", cv=list(cv) if groups is not None else cv)
        self.model.fit(X, y_grip)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, X, K: int | None = None) -> np.ndarray:
        """(n, K) columns for grips 1..K, in order, even if a grip was never seen."""
        p = self.model.predict_proba(X)
        K = K or int(max(self.classes_))
        out = np.zeros((len(X), K))
        for j, c in enumerate(self.classes_):
            out[:, int(c) - 1] = p[:, j]
        return out / out.sum(1, keepdims=True)
