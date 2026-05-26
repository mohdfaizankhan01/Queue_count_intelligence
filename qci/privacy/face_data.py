"""FaceDataLoader — LFW dataset via sklearn with synthetic fallback."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class FaceData:
    """Container for face images and their identity labels.

    images: float32 array (N, H, W, 3) in [0, 1]
    labels: int array (N,) — person index
    n_people: number of distinct identities
    """

    images: np.ndarray
    labels: np.ndarray
    n_people: int

    def __len__(self) -> int:
        return len(self.images)


def _synthetic_faces(n_images: int = 60, height: int = 50, width: int = 37) -> FaceData:
    """Generate random noise images with a rough face-like structure.

    Used as fallback when the real dataset cannot be loaded.
    Each identity gets a distinct mean colour so the recognition
    attacker has something non-trivial to work with.
    """
    rng = np.random.default_rng(0)
    n_people = max(1, n_images // 3)
    imgs_per_person = max(1, n_images // n_people)

    images = []
    labels = []
    for pid in range(n_people):
        base_hue = rng.uniform(0.0, 1.0, 3).astype(np.float32)
        for _ in range(imgs_per_person):
            img = rng.random((height, width, 3)).astype(np.float32) * 0.3
            img += base_hue * 0.6
            img = np.clip(img, 0.0, 1.0)
            # Rough oval "face" bright region
            cy, cx = height // 2, width // 2
            ry, rx = height // 3, width // 3
            for y in range(height):
                for x in range(width):
                    if ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 < 1.0:
                        img[y, x] = np.clip(img[y, x] + 0.3, 0.0, 1.0)
            images.append(img)
            labels.append(pid)

    images_arr = np.stack(images, axis=0)
    labels_arr = np.array(labels, dtype=np.int64)
    log.warning("Using %d synthetic face images (%d identities) as fallback.", len(images_arr), n_people)
    return FaceData(images=images_arr, labels=labels_arr, n_people=n_people)


class FaceDataLoader:
    """Load LFW faces for privacy evaluation.

    Parameters
    ----------
    min_faces_per_person:
        Minimum images per identity to include (passed to sklearn).
    resize:
        Resize factor for LFW (0.5 → 50×37 images).
    max_images:
        Cap total images loaded (keeps tests fast).
    data_home:
        Directory for sklearn dataset cache.  None = sklearn default.
    """

    def __init__(
        self,
        min_faces_per_person: int = 20,
        resize: float = 0.5,
        max_images: Optional[int] = 300,
        data_home: Optional[str] = None,
    ) -> None:
        self.min_faces_per_person = min_faces_per_person
        self.resize = resize
        self.max_images = max_images
        self.data_home = data_home

    def load(self) -> FaceData:
        """Return a FaceData instance, using LFW or synthetic fallback."""
        try:
            return self._load_lfw()
        except Exception as exc:
            log.warning("LFW load failed (%s); using synthetic fallback.", exc)
            return _synthetic_faces()

    def _load_lfw(self) -> FaceData:
        from sklearn.datasets import fetch_lfw_people  # type: ignore

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lfw = fetch_lfw_people(
                min_faces_per_person=self.min_faces_per_person,
                resize=self.resize,
                color=True,
                data_home=self.data_home,
            )

        # lfw.images: (N, H, W, 3) float32 in [0, 255] — normalise to [0, 1]
        images = lfw.images.astype(np.float32)
        if images.max() > 1.5:
            images /= 255.0
        images = np.clip(images, 0.0, 1.0)

        labels = lfw.target.astype(np.int64)
        n_people = int(labels.max()) + 1

        if self.max_images is not None and len(images) > self.max_images:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(images), self.max_images, replace=False)
            idx.sort()
            images = images[idx]
            labels = labels[idx]
            n_people = int(labels.max()) + 1

        log.info("Loaded %d LFW images, %d identities.", len(images), n_people)
        return FaceData(images=images, labels=labels, n_people=n_people)
