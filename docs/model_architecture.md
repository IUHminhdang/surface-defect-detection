# YOLO11n architecture and custom insertion points

The installed Ultralytics version is `8.4.174`. The inspected `yolo11n.pt`
model has 24 top-level modules and 2,624,080 parameters. For a 320 x 320
input, the relevant feature shapes are:

| Layer | Module | Shape | Role |
|---:|---|---|---|
| 4 | C3k2 | 128 x 40 x 40 | native stride-8 backbone feature |
| 6 | C3k2 | 128 x 20 x 20 | backbone P3/P4 transition feature |
| 8 | C3k2 | 256 x 10 x 10 | backbone P4/P5 feature |
| 10 | C2PSA | 256 x 10 x 10 | deepest backbone feature |
| 16 | C3k2 | 64 x 40 x 40 | high-resolution detection input |
| 19 | C3k2 | 128 x 20 x 20 | middle detection input |
| 22 | C3k2 | 256 x 10 x 10 | deepest detection input |
| 23 | Detect | 3 scales | consumes layers 16, 19, and 22 |

The model therefore already exposes a high-resolution stride-8 feature (40 x
40 at this smoke-test size) to Detect. The custom P2 block uses the native
layer-4 feature at that same resolution, projects it to 128 channels, and
concatenates it with the upsampled neck feature before the layer-16 C3k2 block.
This preserves the high-resolution spatial information without pretending that
the layer-4 tensor is a lower-resolution feature.

## Custom variants

- **E0**: unchanged `yolo11n.pt` baseline.
- **E1**: P2 high-resolution projection and fusion before the high-resolution
  detection feature.
- **E2**: E1 plus `MultiScaleFusion` on the 64-channel high-resolution feature.
- **E3**: E2 plus `CoordinateAttention` on that same feature. The attention
  module pools height and width independently and preserves both spatial size
  and channel count.
- **E4**: the baseline high-resolution feature plus `MultiScaleFusion`, without
  the P2 projection.
- **E5**: the baseline high-resolution feature plus `CoordinateAttention`,
  without the P2 projection.

`MultiScaleFusion` contains parallel 1x1, 3x3, and dilated 3x3 (dilation 2)
branches followed by a 1x1 fusion convolution. Attention is intentionally not
implemented inside the multi-scale block.

## Pretrained loading

The custom YAML is generated under each experiment directory. It is
initialized from the actual YOLO11n graph and calls Ultralytics weight
transfer from `yolo11n.pt`. Compatible tensors are transferred by shape and
key; custom P2, multi-scale, and coordinate-attention layers are newly
initialized. The smoke run reported `264/539` transferred tensors for E3.
The original E0 path continues to use the unmodified baseline checkpoint.
