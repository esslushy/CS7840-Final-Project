# print_digit — the printed-digit dataset

A dataset of rendered digit glyphs, selected by `classification.py --dataset mnist_font`
and referred to as `mnist_font` throughout `results/` filenames. It is **not** torchvision
MNIST: these are printed/font-rendered digits rather than handwriting, which is why the
name appears as a separate dataset in the sweep.

```python
from print_digit import load_mnist_font_dataset
train, test = load_mnist_font_dataset(train_transforms, test_transforms)
```

## Contents

| path | what |
|---|---|
| `__init__.py` | The loader and the `PrintedDigitDataset` class — this is the whole API. |
| `dataset/assets/0` … `9` | ~6,300 `.jpeg` files, one folder per digit class (526–750 each). |
| `dataset/LICENSE` | The dataset's own license. Check it before redistributing. |

`dataset/assets/10/` exists and holds two stray JPEGs. There is no digit 10; the loader
iterates `range(10)` and never reads it.

## How the loader behaves

Worth knowing, because two of these are easy to trip over:

- **It reads a relative path.** `load_mnist_font_dataset` opens
  `print_digit/dataset/assets/<digit>/...`, so the process working directory must be
  `src/`. This is the same constraint the training scripts have.
- **It loads everything eagerly.** All ~6,300 images are decoded with OpenCV into one
  in-memory array before any split, converted to grayscale and scaled to [0, 1].
- **The split is a fixed slice of one shuffle.** Images are permuted with the global
  numpy RNG, then train takes `[:5000]` and test takes `[5000:6000]` — so the split
  depends on `set_seed()` having run first, and any images past index 6000 are silently
  unused. Train and test are two views of the same permutation, so they are disjoint but
  only reproducible per-seed.
- **It requires `cv2`.** OpenCV is imported at module load, so `import print_digit`
  fails without it even for the other datasets.

`classification.py` trains 400 epochs on this dataset rather than the 200 it uses for
CIFAR.
