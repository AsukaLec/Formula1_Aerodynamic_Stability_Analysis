# Statistics Report

**Samples**: 104,999  |  **Features**: 5 (+ 1 target)

## 1. Descriptive Statistics

|                 |       count |      mean |       std |      min |       Q1 |    median |        Q3 |       max |   skewness |   kurtosis |
|:----------------|------------:|----------:|----------:|---------:|---------:|----------:|----------:|----------:|-----------:|-----------:|
| speed_kmh       | 104999.0000 |  232.5354 |   76.4542 | 100.0000 | 166.1400 |  232.6900 |  298.9100 |  365.0000 |    -0.0023 |    -1.2006 |
| wing_angle_deg  | 104999.0000 |   20.0081 |    8.6613 |   5.0000 |  12.5100 |   20.0100 |   27.4600 |   35.0000 |     0.0010 |    -1.1992 |
| drs_active      | 104999.0000 |    0.2990 |    0.4578 |   0.0000 |   0.0000 |    0.0000 |    1.0000 |    1.0000 |     0.8783 |    -1.2287 |
| downforce_n     | 104999.0000 | 2392.4456 | 1844.6235 | 105.3000 | 929.1600 | 1842.4500 | 3423.5600 | 8969.1700 |     1.0667 |     0.4634 |
| drag_n          | 104999.0000 |  166.0911 |  106.4649 |  18.0500 |  74.5250 |  145.6100 |  240.0900 |  515.9200 |     0.6749 |    -0.3506 |
| stability_index | 104999.0000 |   91.3307 |   24.7771 |   0.0000 | 100.0000 |  100.0000 |  100.0000 |  100.0000 |    -2.8993 |     7.0735 |

## 2. Stability Index Distribution by Bin

| Bin     | Count   | Ratio   |
|---------|---------|---------|
| severe | 6,694 | 6.38% |
| moderate | 3,159 | 3.01% |
| mild | 4,827 | 4.60% |
| stable | 90,319 | 86.02% |

- Unstable (stability < 95): **13.98%**
- Stable   (stability >= 95): **86.02%**

## 3. Key Observations

- **Severe left-skew** on `stability_index` (skewness = -2.90): the distribution is heavily concentrated at the upper bound (100).
- `speed_kmh` is approximately symmetric (skewness = -0.00).
- `wing_angle_deg` is approximately symmetric (skewness = 0.00).
- `drs_active` is binary; the mean (0.30) reflects the proportion of DRS=1 samples.
