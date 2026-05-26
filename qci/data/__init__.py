from .synthetic import SyntheticCrowdDataset
from .shanghaitech import ShanghaiTechDataset


def build_dataset(cfg: dict):
    """Factory: return the dataset specified by ``cfg['name']``."""
    name = cfg.get("name", "synthetic")
    if name == "synthetic":
        return SyntheticCrowdDataset(
            n_images=cfg.get("n_synthetic", 20),
            image_size=cfg.get("image_size", (256, 256)),
            max_count=cfg.get("max_count", 50),
            seed=cfg.get("seed", 42),
        )
    elif name == "shanghaitech":
        return ShanghaiTechDataset(
            root=cfg["root"],
            part=cfg.get("part", "A"),
            split=cfg.get("split", "test"),
            density_sigma=cfg.get("density_sigma", 15.0),
        )
    else:
        raise ValueError(f"Unknown dataset: {name!r}")


__all__ = ["SyntheticCrowdDataset", "ShanghaiTechDataset", "build_dataset"]
