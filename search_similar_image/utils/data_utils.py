from pathlib import Path
from typing import List

EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


def get_img_paths(source_dir: str) -> List[str]:
    img_paths = []
    for p in Path(source_dir).glob("*"):
        if p.suffix in EXTENSIONS:
            img_paths.append(str(p))

    return img_paths
