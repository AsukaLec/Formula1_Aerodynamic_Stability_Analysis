# Active Learning Iterative Refinement Report

## Mechanism

The active learning loop operates as follows:
1. **PSO exploration**: PSO optimises mu(x) - lambda*sigma(x) over the parameter space
2. **Uncertainty feedback**: DeepEnsemble estimates sigma(x) at each visited position
3. **Candidate selection**: Top-K high-sigma positions identify regions where the surrogate model is unreliable
4. **Data augmentation**: Nearest real training samples near uncertain regions are added to the training set
5. **Model refinement**: The surrogate model is retrained/fine-tuned on the augmented dataset
6. **Repeat**: PSO resumes with a more trustworthy model, converging to more reliable optima

## 5.1 Shallow: One-Round Post-hoc Refinement

| Metric | Baseline | Refined | Delta |
|--------|----------|---------|-------|
| R2 | 0.9995 | 0.9998 | +0.0003 |
| MSE | 0.3272 | 0.1444 | -0.1828 |
| MAE | 0.2712 | 0.1678 | -0.1034 |
| RMSE | 0.5720 | 0.3800 | -0.1921 |

- Candidates selected: 20
- Augmented samples: 100
- delta_R2 = +0.0003
- delta_MSE = -0.1828
- delta_MAE = -0.1034

### Top High-Uncertainty Candidates

| # | Speed | Wing | DRS | Downforce | Drag | mu | sigma |
|---|-------|------|-----|-----------|------|----|----|
| 1 | 175 | 5.1 | 0 | 8773 | 154.8 | 36.34 | 37.57 |
| 2 | 217 | 14.7 | 1 | 6081 | 260.0 | 46.02 | 36.73 |
| 3 | 139 | 34.6 | 0 | 6911 | 196.6 | 41.63 | 36.63 |
| 4 | 106 | 40.4 | 0 | 6331 | 203.4 | 36.17 | 32.13 |
| 5 | 114 | 22.3 | 0 | 9093 | 155.3 | 27.68 | 31.36 |

## PSO Convergence Comparison (Baseline vs Refined)

| Model | Final Gbest | Iterations |
|-------|-------------|------------|
| Baseline | 101.2989 | 37 |
| Refined  | 100.0162 | 18 |

- Convergence speed-up: 2.1x
- Note: The refined model converges faster and produces more conservative (trustworthy) gbest values. The baseline model may hallucinate unrealistically high stability in out-of-distribution regions, leading to deceptively high gbest fitness.