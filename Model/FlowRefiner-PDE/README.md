# FlowRefiner-PDE

Post-hoc PDE-style multilevel denoising refiner. The FlowMatch generator is trained first, base predictions are cached, and this refiner is then trained without backpropagating into the generator.

Typical training:

```bash
python train.py --train-index ../../splits/train_index.csv --val-index ../../splits/val_index.csv --train-base-cache ../../runs/flowrefiner_pde/cache/train_base.npy --val-base-cache ../../runs/flowrefiner_pde/cache/val_base.npy --base-run-dir ../../runs/flowmatch --base-checkpoint best --out-dir ../../runs/flowrefiner_pde --epochs 120 --batch-size 4 --lr 0.00002 --base-channels 64 --channel-multipliers 1,2,4,8 --res-blocks 2 --refinement-steps 8 --sigma-max 0.35 --sigma-min 0.01 --refine-strength 1.0 --train-state-mode base_to_target --loss l1 --mse-loss-weight 0.1 --gradient-loss-weight 0.05 --ema-decay 0.9999 --seed 42 --device cuda
```

Typical evaluation:

```bash
python evaluate.py --test-index ../../splits/test_index.csv --test-base-cache ../../runs/flowrefiner_pde/cache/test_base.npy --run-dir ../../runs/flowrefiner_pde --checkpoint best --batch-size 4 --refinement-steps 8 --refine-strength 1.0 --device cuda
```

