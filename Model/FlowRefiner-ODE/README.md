# FlowRefiner-ODE

This folder contains the ODE residual refiner for FlowRefiner-ODE.

Use the top-level dispatcher from the repository root:

```powershell
python .\run_experiment.py --model flowrefiner-ode --stage precompute -- --split-root .\splits --out-root .\runs\flowmatch_base_predictions
python .\run_experiment.py --model flowrefiner-ode --stage train -- --split-root .\splits --precomputed-root .\runs\flowmatch_base_predictions --out-dir .\runs\flowrefiner_ode
python .\run_experiment.py --model flowrefiner-ode --stage evaluate -- --split-root .\splits --precomputed-root .\runs\flowmatch_base_predictions --ckpt .\runs\flowrefiner_ode\best.pt
```
