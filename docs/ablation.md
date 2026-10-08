# Ablation studies

The implemented controlled ablation is:

- E0 = unchanged YOLO11n
- E1 = YOLO11n + P2 high-resolution fusion
- E2 = E1 + multi-scale fusion
- E3 = E2 + coordinate attention

The same dataset, split, seed, image size, batch size, AdamW optimizer, cosine
schedule, warmup, patience, and checkpoint policy are used for every variant.
Only the explicitly enabled module changes.

For E1-E3, select the variant in `configs/default.yaml`:

```yaml
training:
  custom:
    ablation: E3
```

Then run:

```bash
python scripts/train.py --data-root /path/to/dataset \
  --config configs/default.yaml --model yolo11-p2-msc-ca \
  --experiment yolo11-e3
```
