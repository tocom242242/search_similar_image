import numpy as np
import pytest
from PIL import Image

from search_similar_image import __version__
from search_similar_image.model.search_model import SearchModel


def test_version():
    assert __version__ == "0.1.0"


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
    fn2 = image_dir_path / "img2.png"
    img.save(fn2)
    yield image_dir_path


class TestSearchModel:
    def test_init_serch_model(self):
        model = SearchModel()
        assert model is not None

    def test_fit(self, image_dir):
        model = SearchModel()
        model.fit(image_dir)
        assert model.index is not None
        assert type(model.source_img_paths) is np.ndarray

    def test_predict(self, image_dir, image_file):
        model = SearchModel()
        model.fit(image_dir)
        scores, paths = model.predict(image_file)
        assert type(paths) is np.ndarray
        assert type(scores) is np.ndarray
