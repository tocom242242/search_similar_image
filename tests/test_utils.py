from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from search_similar_image import __version__
from search_similar_image.utils import data_utils


@pytest.fixture
def image_file(tmp_path_factory):
    img = Image.new("RGB", (320, 320))
    fn = tmp_path_factory.mktemp("data") / "img.png"
    img.save(fn)
    yield fn


@pytest.fixture
def image_dir(tmp_path_factory):
    img = Image.new("RGB", (320, 320))
    image_dir_path = tmp_path_factory.mktemp("data")
    fn1 = image_dir_path / "img.png"
    img.save(fn1)
    img = Image.new("RGB", (320, 320))
    fn2 = image_dir_path / "img2.jpeg"
    img.save(fn2)

    img = Image.new("RGB", (320, 320))
    fn3 = image_dir_path / "img3.jpg"
    img.save(fn3)

    yield image_dir_path


def test_get_img_paths(image_dir):
    img_paths = data_utils.get_img_paths(image_dir)
    assert isinstance(img_paths, list)
    for img_path in img_paths:
        assert isinstance(img_path, str)
