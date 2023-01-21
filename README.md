# Search Similar Image

Simple library to find similar image using pretrained model and KNN.

## How to Use

```bash
pip install search-similar-image
```

```python
from search_similar_image.model import SearchModel

model = SearchModel()
model.fit(r"<path/to/source_dir>")
scores, similar_image_paths = model.predict(r"<path/to/target_image_path>"))
print(scores)
# => [[3527.487  3911.2122 4135.191  4163.338  4209.0986]]

print(similar_image_paths)
#=> [['image1.jpg'
#  'image2.jpg'
#  'image3.jpg'
#  'image4.jpg'
#  'image5.jpg']]
```
