# Classification nets

Two files, same four architectures, differing only in what the input looks like:

| file | input | used when |
|---|---|---|
| `CIFARNets.py` | 32×32 RGB, 3 channels | `classification.py --dataset cifar` |
| `MNISTNets.py` | 28×28 grayscale, 1 channel | `classification.py --dataset mnist_font` |

Both export `CNN`, `NaiveNet`, `ViT` (plus the `FeedForward` / `Attention` /
`Transformer` blocks the ViT is built from). `classification.py` imports the matching
module inside `main()`, branching on `--dataset`. Everything in
[`../README.md`](../README.md) about the shared architecture family applies.

What actually differs is the input-dependent widths: `CNN.fc1` takes `16·5·5` on CIFAR
against `16·4·4` on MNIST, and `NaiveNet.fc1` takes `32·32·3` against `28·28`. The two
files are otherwise the same code, so a change to one needs mirroring in the other.

`CIFARNets.CNN` is the model `../../metric_demos/lie_vs_cka_demo.py` probes, and it
hardcodes that class's layer names — `conv1, pool1, pool2, fc1, fc2, fc3, softmax` —
to walk the network depth. The `fc1` boundary is the point of that demo: it is where the
Lie-derivative reference convention switches from assuming features rotate to assuming
they are invariant.

`mnist_font` is the printed-digit dataset in [`../../print_digit/`](../../print_digit/),
not torchvision MNIST.
