# Data Imbalance Report

## Overview

- Total samples: 149999
- Stability bins: severe: [0, 30), moderate: [30, 60), mild: [60, 95), stable: [95, 101)
- Unstable (stability < 95): 20971 (13.98%)

## Bin Distribution

| Bin | Count | Ratio | Weight |
|------|--------|------|------|
| severe | 9563 | 6.38% | 3.9213 |
| moderate | 4513 | 3.01% | 8.3093 |
| mild | 6895 | 4.60% | 5.4387 |
| stable | 129028 | 86.02% | 0.2906 |

## Skew Analysis

- stable / severe ratio: 13.49:1
- **Severely skewed**: stable samples dominate (ratio > 3:1)

## Weight Strategy

- Class-balanced weighting: `weight_i = N / (K * n_i)`
- severe weight: 3.9213
- moderate weight: 8.3093
- mild weight: 5.4387
- stable weight: 0.2906
- Weights saved to `data/processed/sample_weights.npy`
