# SolarFlowRefiner

End-to-end refinement-aware FlowMatch generator plus PDE-style refiner. The refinement objective is backpropagated through the differentiable FlowMatch sampler.

Typical training:

```bash
python train.py --train-index ../../splits/train_index.csv --val-index ../../splits/val_index.csv --out-dir ../../runs/solarflowrefiner --epochs 100 --batch-size 1 --flowmatch-lr 0.000002 --refiner-lr 0.00002 --base-channels 64 --channel-multipliers 1,2,4,8 --res-blocks 2 --flowmatch-sample-steps 8 --flowmatch-solver euler --refinement-steps 8 --sigma-max 0.35 --sigma-min 0.01 --refine-strength 1.0 --refiner-loss l1 --mse-loss-weight 0.1 --gradient-loss-weight 0.05 --seed 42 --device cuda
```

