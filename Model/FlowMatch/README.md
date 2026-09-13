# FlowMatch

Standalone conditional FlowMatch residual generator for ERA5-to-SolarCube SSR downscaling.

Typical training:

```bash
python train.py --train-index ../../splits/train_index.csv --val-index ../../splits/val_index.csv --out-dir ../../runs/flowmatch --epochs 100 --batch-size 4 --lr 0.00002 --base-channels 64 --channel-multipliers 1,2,4,8 --res-blocks 2 --sample-steps 50 --solver heun --time-sampling logit_normal --loss l1 --ema-decay 0.9999 --seed 42 --device cuda
```

Typical evaluation:

```bash
python evaluate.py --test-index ../../splits/test_index.csv --run-dir ../../runs/flowmatch --checkpoint best --batch-size 4 --sample-steps 50 --solver heun --device cuda
```

