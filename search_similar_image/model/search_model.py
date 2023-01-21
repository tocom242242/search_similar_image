from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from search_similar_image.utils import data_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SearchModel:
    def __init__(self, model_name: str = "vit_base_patch16_224"):
        """Simple Search Similar Image Model.

        Args:
            model_name (str, optional): pretrained model name. Defaults to "vit_base_patch16_224".
        """
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model.to(device)
        self.model.eval()
        config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**config)
        self.index = None
        self.source_img_paths = None

    def fit(self, source_dir: str):
        """Fit k nearest neighbour model using features outputted from pretrained model

        Args:
            source_dir (str): Path of images
        """
        self.source_img_paths = np.array(data_utils.get_img_paths(source_dir))
        features = []
        for source_img_path in self.source_img_paths:
            img = Image.open(source_img_path).convert("RGB")
            feature = self._extract_feature(img)
            features.append(feature.squeeze(0).numpy())
        features = np.array(features)

        self.index = faiss.IndexFlatL2(features.shape[1])
        self.index.add(features)

    def _extract_feature(self, img: Image) -> torch.Tensor:
        with torch.no_grad():
            img_tensor = self.transform(img)
            feature = self.model(img_tensor.unsqueeze(0))
        return feature

    def predict(self, target_img_path: str) -> Tuple[List, List]:
        """Predict nearest neighbor of the image

        Args:
            target_img_path (str): The target image path for prediction.

        Returns:
            Tuple[List, List]: Scores and neighbor image paths
        """
        img = Image.open(target_img_path).convert("RGB")
        feature = self._extract_feature(img)
        feature = feature.numpy()

        D, indexes = self.index.search(feature, 5)
        neighbor_img_paths = self.source_img_paths[indexes]
        return D, neighbor_img_paths
