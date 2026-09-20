"""Thin adapter around an official SAM3 checkout (no vendored SAM3 code)."""

from __future__ import annotations

import copy
import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class Sam3Backend:
    def __init__(self, repo: Path, checkpoint: Path, device: str = "cuda", confidence: float = .5):
        self.repo, self.checkpoint, self.device = repo.resolve(), checkpoint.resolve(), device
        self.confidence = confidence
        self._processor: Any = None
        self._model: Any = None
        self._cache: tuple[str, Any] | None = None
        self._lock = threading.RLock()

    def _load(self) -> None:
        if self._model is not None:
            return
        if not (self.repo / "sam3").is_dir():
            raise RuntimeError(f"official SAM3 package not found under {self.repo}")
        sys.path.insert(0, str(self.repo))
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        bpe = self.repo / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        self._model = build_sam3_image_model(bpe_path=str(bpe), device=self.device,
            checkpoint_path=str(self.checkpoint), load_from_HF=False, enable_inst_interactivity=True)
        self._processor = Sam3Processor(self._model, device=self.device, confidence_threshold=self.confidence)

    def _amp(self):
        if not self.device.startswith("cuda"):
            return nullcontext()
        import torch
        return torch.autocast("cuda", dtype=torch.bfloat16)

    def _state(self, image_path: Path):
        key = str(image_path.resolve())
        if self._cache is None or self._cache[0] != key:
            with Image.open(image_path) as image, self._amp():
                state = self._processor.set_image(image.convert("RGB"))
            self._cache = (key, state)
        return self._cache[1]

    @staticmethod
    def _unpack(state: Any, limit: int = 10) -> tuple[list[np.ndarray], list[float]]:
        masks, scores = [], []
        for index, tensor in enumerate(state.get("masks", [])[:limit]):
            masks.append(tensor.squeeze().detach().cpu().numpy().astype(bool))
            score = state["scores"][index]
            scores.append(float(score.item() if hasattr(score, "item") else score))
        return masks, scores

    def text(self, image_path: Path, prompt: str, limit: int = 10):
        with self._lock:
            self._load()
            with self._amp():
                state = self._processor.set_text_prompt(prompt=prompt, state=copy.deepcopy(self._state(image_path)))
            return self._unpack(state, limit)

    def interactive(self, image_path: Path, *, prompt: str = "", points=None, labels=None,
                    box=None, mask_input=None):
        with self._lock:
            self._load()
            if prompt:
                return self.text(image_path, prompt)
            state = self._state(image_path)
            if isinstance(mask_input, np.ndarray) and mask_input.shape == (518, 518):
                prompt_encoder = self._model.inst_interactive_predictor.model.sam_prompt_encoder
                height, width = (int(x) for x in prompt_encoder.mask_input_size)
                low = Image.fromarray(mask_input.astype(np.uint8) * 255, "L").resize((width,height), Image.Resampling.BILINEAR)
                probability = np.asarray(low, np.float32) / 255.0
                mask_input = ((probability * 2.0 - 1.0) * 10.0)[None]
            with self._amp():
                masks, scores, _ = self._model.predict_inst(
                    state, point_coords=np.asarray(points, np.float32) if points else None,
                    point_labels=np.asarray(labels, np.int32) if labels else None,
                    box=np.asarray(box, np.float32).reshape(1, 4) if box else None,
                    mask_input=mask_input, multimask_output=True)
            return [np.asarray(x).astype(bool) for x in masks], [float(x) for x in scores]
