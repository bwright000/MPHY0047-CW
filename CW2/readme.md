# MPHY0047 Coursework 2 - Report

## How to Run

To execute all analysis scripts and generate figures (this report assumes the sds environment is used):

```bash
python question1.py  # Pearson correlation + linear regression (Q1)
python question2.py  # Image similarity metrics SSI/MI/CS (Q2)
python question3.py  # Metric correlations + polynomial/Gaussian regression (Q3)
python question4.py  # ECC rigid transformation + statistical testing (Q4)
```

---

## Question 1: Quality Score Agreement and Regression [20 marks]

### 1.1 Pearson Correlation Coefficient (Part i)

#### Table 1: Pearson Correlation Between General Impression and Criteria Percentage per View

| View | Pearson r | p-value | Significant? (p < 0.05) |
|------|-----------|---------|-------------------------|
| View 1 | 0.9274 | < 0.0001 | Yes |
| View 2 | 0.7866 | 0.000039 | Yes |
| View 3 | 0.8168 | 0.000011 | Yes |
| View 4 | 0.6545 | 0.002359 | Yes |
| View 5 | 0.8699 | 0.000001 | Yes |
| View 6 | 0.7994 | 0.000024 | Yes |
| View 7 | 0.7964 | 0.000027 | Yes |
| View 8 | 0.7095 | 0.000669 | Yes |
| View 9 | 0.9268 | < 0.0001 | Yes |
| View 10 | 0.8699 | 0.000003 | Yes |

**Highest agreement: View 1 (r = 0.9274)**

> **[YOUR ANALYSIS]** Discuss the Pearson correlation results. Consider: which views show strong/weak agreement between the two scoring methods, what r values indicate about the linear relationship, and why View 1 has the highest agreement.
>
> ...

### 1.2 Linear Regression Analysis (Part ii)

Linear regression was performed using general impression as the independent variable and criteria percentage as the dependent variable for each view.

#### Table 2: Linear Regression Results (gen_impr -> crit_perc) per View

| View | Slope | Intercept | RMSE | R² |
|------|-------|-----------|------|-----|
| View 1 | 19.0496 | 22.2668 | 5.8684 | 0.8600 |
| View 2 | 14.7533 | 34.6767 | 9.0995 | 0.6188 |
| View 3 | 18.0674 | 25.7058 | 9.6140 | 0.6671 |
| View 4 | 15.0933 | 32.5409 | 14.3392 | 0.4284 |
| View 5 | 18.0557 | 29.1665 | 9.3100 | 0.7566 |
| View 6 | 21.5538 | 10.3519 | 17.1100 | 0.6390 |
| View 7 | 18.9317 | 32.8973 | 17.7742 | 0.6343 |
| View 8 | 12.0292 | 43.9508 | 10.6929 | 0.5034 |
| View 9 | 20.1808 | 26.3443 | 12.8205 | 0.8590 |
| View 10 | 19.5312 | 23.5629 | 12.0564 | 0.7567 |

> **[YOUR ANALYSIS - comment on regression performance]** Discuss the RMSE and R² values across views. Consider: which views have good/poor fits, what the slope values indicate about the relationship between gen_impr and crit_perc, and any patterns in the regression quality.
>
> ...

### 1.3 True vs Estimated Plots - Three Best Performing Views (Part iii)

The three best performing views by R² are: **View 1** (R² = 0.8600), **View 9** (R² = 0.8590), and **View 10** (R² = 0.7567).

#### Figure 1: View 1 - True vs Estimated Criteria Percentage

![Figure 1](figures/q1_true_vs_estimated_view_1.png)

#### Figure 2: View 9 - True vs Estimated Criteria Percentage

![Figure 2](figures/q1_true_vs_estimated_view_9.png)

#### Figure 3: View 10 - True vs Estimated Criteria Percentage

![Figure 3](figures/q1_true_vs_estimated_view_10.png)

*Points are colour-coded by criteria percentage range: blue = CP 40-70%, green = CP 70-100%. The red dashed line represents perfect prediction (y = x).*

> **[YOUR ANALYSIS - comment on model performance for CP 40-70% vs CP 70-100%]** Compare the scatter of blue (40-70%) vs green (70-100%) points relative to the perfect prediction line. Consider: does the model predict better for high or low scoring images, what does clustering/spread suggest, and how do the two ranges differ in prediction accuracy?
>
> ...

---

## Question 2: Image Similarity Metrics [20 marks]

### 2.1 Top 3 Participants per View by Similarity Metric (Part i)

#### Table 3: Top 3 Participants by SSI per View

| View | #1 | #2 | #3 |
|------|-----|-----|-----|
| View 1 | P2 (0.8241) | P4 (0.8179) | P12 (0.8171) |
| View 2 | P14 (0.8619) | P19 (0.8560) | P8 (0.8462) |
| View 3 | P14 (0.8537) | P9 (0.8528) | P20 (0.8523) |
| View 4 | P2 (0.9143) | P15 (0.9111) | P14 (0.9098) |
| View 5 | P2 (0.8762) | P16 (0.8498) | P3 (0.8497) |
| View 6 | P2 (0.8255) | P19 (0.8228) | P1 (0.8178) |
| View 7 | P17 (0.8714) | P8 (0.8652) | P20 (0.8631) |
| View 8 | P2 (0.8287) | P4 (0.8269) | P6 (0.8206) |
| View 9 | P8 (0.7776) | P9 (0.7773) | P17 (0.7710) |
| View 10 | P5 (0.8461) | P2 (0.8332) | P11 (0.8278) |

#### Table 4: Top 3 Participants by MI per View

| View | #1 | #2 | #3 |
|------|-----|-----|-----|
| View 1 | P2 (0.0007) | P4 (0.0006) | P6 (0.0005) |
| View 2 | P3 (0.0027) | P14 (0.0018) | P19 (0.0015) |
| View 3 | P3 (0.0037) | P9 (0.0013) | P14 (0.0010) |
| View 4 | P2 (0.0015) | P5 (0.0010) | P3 (0.0010) |
| View 5 | P2 (0.0032) | P3 (0.0008) | P5 (0.0007) |
| View 6 | P2 (0.0017) | P19 (0.0010) | P12 (0.0008) |
| View 7 | P17 (0.0014) | P10 (0.0012) | P2 (0.0010) |
| View 8 | P4 (0.0009) | P2 (0.0009) | P18 (0.0007) |
| View 9 | P19 (0.0009) | P3 (0.0009) | P14 (0.0008) |
| View 10 | P5 (0.0037) | P2 (0.0020) | P10 (0.0011) |

#### Table 5: Top 3 Participants by CS per View

| View | #1 | #2 | #3 |
|------|-----|-----|-----|
| View 1 | P2 (0.0620) | P4 (0.0609) | P6 (0.0551) |
| View 2 | P14 (0.1065) | P19 (0.0961) | P6 (0.0924) |
| View 3 | P9 (0.0887) | P14 (0.0770) | P17 (0.0733) |
| View 4 | P2 (0.1057) | P5 (0.0846) | P3 (0.0826) |
| View 5 | P2 (0.1484) | P3 (0.0679) | P5 (0.0654) |
| View 6 | P2 (0.1017) | P19 (0.0768) | P12 (0.0694) |
| View 7 | P17 (0.0958) | P10 (0.0837) | P2 (0.0788) |
| View 8 | P4 (0.0746) | P2 (0.0733) | P18 (0.0657) |
| View 9 | P19 (0.0744) | P3 (0.0733) | P14 (0.0681) |
| View 10 | P5 (0.1559) | P2 (0.1090) | P10 (0.0791) |

> **[YOUR ANALYSIS]** Discuss patterns in the top 3 rankings. Consider: do experts (P1-P7) dominate the top 3 across metrics, are there specific participants who consistently rank highly, do the three metrics (SSI, MI, CS) agree on which participants produce the best images?
>
> ...

### 2.2 Statistical Testing - Expert vs Novice (Part ii)

The Mann-Whitney U test was used to evaluate differences between expert (P1-P7) and novice (P8-P20) groups for each similarity metric per view.

**H0:** No difference in similarity metric between expert and novice groups.
**H1:** Expert and novice groups differ in similarity metric.
**Significance level:** alpha = 0.05

#### Table 6: Mann-Whitney U Test Results - SSI

| View | Expert Mean | Novice Mean | U-stat | p-value | Significant? |
|------|-------------|-------------|--------|---------|--------------|
| View 1 | 0.8020 | 0.7893 | 58.00 | 0.1956 | No |
| View 2 | 0.8340 | 0.8369 | 35.00 | 0.4378 | No |
| View 3 | 0.8346 | 0.8424 | 22.00 | 0.0675 | No |
| View 4 | 0.9025 | 0.9019 | 42.00 | 1.0000 | No |
| View 5 | 0.8430 | 0.8336 | 51.00 | 0.6992 | No |
| View 6 | 0.7998 | 0.7953 | 54.00 | 0.5356 | No |
| View 7 | 0.8464 | 0.8540 | 36.00 | 0.4854 | No |
| View 8 | 0.8165 | 0.7976 | 75.00 | **0.0037** | **Yes** |
| View 9 | 0.7625 | 0.7567 | 52.00 | 0.6426 | No |
| View 10 | 0.8213 | 0.8160 | 43.00 | 0.7242 | No |

#### Table 7: Mann-Whitney U Test Results - MI

| View | Expert Mean | Novice Mean | U-stat | p-value | Significant? |
|------|-------------|-------------|--------|---------|--------------|
| View 1 | 0.0004 | 0.0002 | 74.00 | **0.0052** | **Yes** |
| View 2 | 0.0010 | 0.0008 | 44.00 | 0.9385 | No |
| View 3 | 0.0009 | 0.0005 | 52.00 | 0.6426 | No |
| View 4 | 0.0008 | 0.0005 | 64.00 | 0.0684 | No |
| View 5 | 0.0008 | 0.0003 | 59.00 | 0.3114 | No |
| View 6 | 0.0006 | 0.0003 | 65.00 | 0.1348 | No |
| View 7 | 0.0005 | 0.0006 | 45.00 | 1.0000 | No |
| View 8 | 0.0005 | 0.0002 | 68.00 | **0.0283** | **Yes** |
| View 9 | 0.0005 | 0.0004 | 55.00 | 0.4854 | No |
| View 10 | 0.0010 | 0.0005 | 40.00 | 0.9298 | No |

#### Table 8: Mann-Whitney U Test Results - CS

| View | Expert Mean | Novice Mean | U-stat | p-value | Significant? |
|------|-------------|-------------|--------|---------|--------------|
| View 1 | 0.0501 | 0.0355 | 75.00 | **0.0037** | **Yes** |
| View 2 | 0.0603 | 0.0665 | 35.00 | 0.4378 | No |
| View 3 | 0.0525 | 0.0537 | 50.00 | 0.7573 | No |
| View 4 | 0.0741 | 0.0573 | 64.00 | 0.0684 | No |
| View 5 | 0.0637 | 0.0430 | 59.00 | 0.3114 | No |
| View 6 | 0.0555 | 0.0416 | 65.00 | 0.1348 | No |
| View 7 | 0.0523 | 0.0549 | 45.00 | 1.0000 | No |
| View 8 | 0.0541 | 0.0377 | 68.00 | **0.0283** | **Yes** |
| View 9 | 0.0535 | 0.0497 | 56.00 | 0.4378 | No |
| View 10 | 0.0688 | 0.0536 | 39.00 | 1.0000 | No |

#### Table 9: Summary - Significant Views per Metric

| Metric | Significant Views (p < 0.05) | Count |
|--------|------------------------------|-------|
| SSI | View 8 | 1 / 10 |
| MI | Views 1, 8 | 2 / 10 |
| CS | Views 1, 8 | 2 / 10 |

**Best differentiating metric: MI and CS** (each significant in 2 / 10 views)

> **[YOUR ANALYSIS - discuss significance results and which metric best differentiates]** Discuss: why most views show no significant difference, what Views 1 and 8 have in common that produces significance, the interpretation of expert vs novice means, justification for Mann-Whitney U test choice, and which metric (SSI, MI, or CS) is the best differentiator and why.
>
> ...

---

## Question 3: Regression Analysis of Similarity Metrics [30 marks]

### 3.1 Correlation Between Similarity Metric Pairs (Part i)

#### Table 10: Pearson Correlation Between SSI-MI, SSI-CS, and MI-CS per View

| View | SSI-MI r | SSI-CS r | MI-CS r |
|------|----------|----------|---------|
| View 1 | 0.7530 | 0.7279 | 0.9935 |
| View 2 | 0.3253 | 0.7581 | 0.6421 |
| View 3 | 0.2237 | 0.5125 | 0.5692 |
| View 4 | 0.4888 | 0.5206 | 0.9962 |
| View 5 | 0.7165 | 0.7724 | 0.9801 |
| View 6 | 0.7622 | 0.7747 | 0.9858 |
| View 7 | 0.4941 | 0.4935 | 0.9910 |
| View 8 | 0.7367 | 0.7199 | 0.9886 |
| View 9 | 0.0144 | -0.0703 | 0.9860 |
| View 10 | 0.8252 | 0.8540 | 0.9864 |

**Highest agreement per pair:**
- SSI-MI: **View 10** (r = 0.8252)
- SSI-CS: **View 10** (r = 0.8540)
- MI-CS: **View 4** (r = 0.9962)

> **[YOUR ANALYSIS]** Discuss the correlation patterns. Consider: why MI-CS correlations are consistently very high (>0.98) across all views, why SSI shows weaker and more variable correlation with MI and CS, what this implies about the redundancy/complementarity of the metrics, and why View 9 shows near-zero SSI-MI and SSI-CS correlations.
>
> ...

### 3.2 Polynomial Regression with LASSO Regularization (Part ii)

LASSO regularization was selected to handle overfitting in polynomial regression up to degree 7. Coefficients smaller than 0.01 were treated as not contributing.

> **[YOUR ANALYSIS - justify regularization method selection]** Explain why LASSO was chosen over Ridge or Elastic Net. Consider: LASSO's L1 penalty drives coefficients to exactly zero (feature selection), which is desirable when many polynomial terms may not contribute; the 0.01 coefficient threshold aligns with LASSO's sparsity-inducing properties; comparison with alternatives.
>
> ...

#### Table 11: LASSO Polynomial Regression - SSI -> crit_perc

| View | RMSE | R² | Optimal Degree |
|------|------|-----|----------------|
| View 1 | 13.12 | 0.3000 | 7 |
| View 2 | 11.80 | 0.3593 | 7 |
| View 3 | 14.79 | 0.2124 | 6 |
| View 4 | 17.38 | 0.1604 | 1 |
| View 5 | 16.41 | 0.2440 | 6 |
| View 6 | 25.40 | 0.2047 | 7 |
| View 7 | 29.39 | 0.0000 | 1 |
| View 8 | 14.31 | 0.1111 | 7 |
| View 9 | 31.41 | 0.1535 | 7 |
| View 10 | 21.52 | 0.2250 | 1 |

#### Table 12: LASSO Polynomial Regression - SSI -> gen_impr

| View | RMSE | R² | Optimal Degree |
|------|------|-----|----------------|
| View 1 | 0.6190 | 0.3428 | 7 |
| View 2 | 0.7572 | 0.0715 | 1 |
| View 3 | 0.6912 | 0.1580 | 7 |
| View 4 | 0.8225 | 0.0000 | 1 |
| View 5 | 0.6784 | 0.4433 | 1 |
| View 6 | 0.8804 | 0.3051 | 7 |
| View 7 | 1.1816 | 0.0869 | 6 |
| View 8 | 0.8949 | 0.0000 | 1 |
| View 9 | 1.5680 | 0.0000 | 1 |
| View 10 | 1.0243 | 0.1148 | 7 |

#### Table 13: LASSO Polynomial Regression - MI -> crit_perc

| View | RMSE | R² | Optimal Degree |
|------|------|-----|----------------|
| View 1 | 10.63 | 0.5410 | 1 |
| View 2 | 14.74 | 0.0000 | 1 |
| View 3 | 16.33 | 0.0393 | 1 |
| View 4 | 18.49 | 0.0498 | 1 |
| View 5 | 17.83 | 0.1071 | 1 |
| View 6 | 23.83 | 0.2998 | 1 |
| View 7 | 25.89 | 0.2240 | 1 |
| View 8 | 12.86 | 0.2812 | 1 |
| View 9 | 34.14 | 0.0000 | 1 |
| View 10 | 22.14 | 0.1799 | 1 |

#### Table 14: LASSO Polynomial Regression - MI -> gen_impr

| View | RMSE | R² | Optimal Degree |
|------|------|-----|----------------|
| View 1 | 0.5145 | 0.5458 | 1 |
| View 2 | 0.7813 | 0.0116 | 1 |
| View 3 | 0.7200 | 0.0865 | 1 |
| View 4 | 0.6994 | 0.2769 | 1 |
| View 5 | 0.8098 | 0.2068 | 1 |
| View 6 | 0.9157 | 0.2484 | 1 |
| View 7 | 1.0906 | 0.2220 | 1 |
| View 8 | 0.8949 | 0.0000 | 1 |
| View 9 | 1.5680 | 0.0000 | 1 |
| View 10 | 0.9904 | 0.1723 | 1 |

#### Table 15: LASSO Polynomial Regression - CS -> crit_perc

| View | RMSE | R² | Optimal Degree |
|------|------|-----|----------------|
| View 1 | 10.71 | 0.5338 | 1 |
| View 2 | 14.74 | 0.0000 | 1 |
| View 3 | 15.40 | 0.1463 | 2 |
| View 4 | 17.59 | 0.1404 | 2 |
| View 5 | 17.67 | 0.1236 | 1 |
| View 6 | 23.22 | 0.3351 | 1 |
| View 7 | 25.63 | 0.2395 | 1 |
| View 8 | 12.69 | 0.3009 | 1 |
| View 9 | 34.14 | 0.0000 | 1 |
| View 10 | 21.85 | 0.2010 | 1 |

#### Table 16: LASSO Polynomial Regression - CS -> gen_impr

| View | RMSE | R² | Optimal Degree |
|------|------|-----|----------------|
| View 1 | 0.5263 | 0.5248 | 1 |
| View 2 | 0.7859 | 0.0000 | 1 |
| View 3 | 0.6153 | 0.3329 | 4 |
| View 4 | 0.6826 | 0.3112 | 2 |
| View 5 | 0.8001 | 0.2255 | 1 |
| View 6 | 0.9144 | 0.2505 | 1 |
| View 7 | 1.0717 | 0.2488 | 1 |
| View 8 | 0.8949 | 0.0000 | 1 |
| View 9 | 1.5680 | 0.0000 | 1 |
| View 10 | 1.0002 | 0.1560 | 1 |

#### Best 3 Performing Views (Top 3 per combination plotted)

**SSI -> crit_perc:** Views 2 (R²=0.36), 1 (R²=0.30), 5 (R²=0.24)

| | |
|:-:|:-:|
| ![SSI->CP View 1](figures/q3ii_SSI_crit_perc_view_1.png) | ![SSI->CP View 2](figures/q3ii_SSI_crit_perc_view_2.png) |
| ![SSI->CP View 5](figures/q3ii_SSI_crit_perc_view_5.png) | |

**SSI -> gen_impr:** Views 5 (R²=0.44), 1 (R²=0.34), 6 (R²=0.31)

| | |
|:-:|:-:|
| ![SSI->GI View 5](figures/q3ii_SSI_gen_impr_view_5.png) | ![SSI->GI View 1](figures/q3ii_SSI_gen_impr_view_1.png) |
| ![SSI->GI View 6](figures/q3ii_SSI_gen_impr_view_6.png) | |

**MI -> crit_perc:** Views 1 (R²=0.54), 6 (R²=0.30), 8 (R²=0.28)

| | |
|:-:|:-:|
| ![MI->CP View 1](figures/q3ii_MI_crit_perc_view_1.png) | ![MI->CP View 6](figures/q3ii_MI_crit_perc_view_6.png) |
| ![MI->CP View 8](figures/q3ii_MI_crit_perc_view_8.png) | |

**MI -> gen_impr:** Views 1 (R²=0.55), 4 (R²=0.28), 6 (R²=0.25)

| | |
|:-:|:-:|
| ![MI->GI View 1](figures/q3ii_MI_gen_impr_view_1.png) | ![MI->GI View 4](figures/q3ii_MI_gen_impr_view_4.png) |
| ![MI->GI View 6](figures/q3ii_MI_gen_impr_view_6.png) | |

**CS -> crit_perc:** Views 1 (R²=0.53), 6 (R²=0.34), 8 (R²=0.30)

| | |
|:-:|:-:|
| ![CS->CP View 1](figures/q3ii_CS_crit_perc_view_1.png) | ![CS->CP View 6](figures/q3ii_CS_crit_perc_view_6.png) |
| ![CS->CP View 8](figures/q3ii_CS_crit_perc_view_8.png) | |

**CS -> gen_impr:** Views 1 (R²=0.52), 3 (R²=0.33), 4 (R²=0.31)

| | |
|:-:|:-:|
| ![CS->GI View 1](figures/q3ii_CS_gen_impr_view_1.png) | ![CS->GI View 3](figures/q3ii_CS_gen_impr_view_3.png) |
| ![CS->GI View 4](figures/q3ii_CS_gen_impr_view_4.png) | |

> **[YOUR ANALYSIS - comment on regression performance across combinations]** Discuss: which metric/score combinations produce the best fits, why View 1 consistently performs well, the observation that many MI/CS regressions collapse to degree 1 (linear), what the generally low R² values indicate about the predictive power of similarity metrics for quality scores, and comparison of crit_perc vs gen_impr as dependent variables.
>
> ...

### 3.3 Gaussian Basis Regression - SSI -> gen_impr (Part iii)

Linear regression using Gaussian basis functions was performed for SSI (independent) against general impression (dependent). LASSO regularization with cross-validation was used to select the optimal basis order (range 2-10).

#### Table 17: Gaussian Basis Regression Results (SSI -> gen_impr)

| View | RMSE | R² | Optimal Order |
|------|------|-----|---------------|
| View 1 | 0.4185 | 0.6996 | 10 |
| View 2 | 0.7091 | 0.1857 | 9 |
| View 3 | 0.6649 | 0.2210 | 6 |
| View 4 | 0.3785 | 0.7883 | 10 |
| View 5 | 0.6549 | 0.4812 | 6 |
| View 6 | 0.8218 | 0.3947 | 5 |
| View 7 | 0.9067 | 0.4622 | 5 |
| View 8 | 0.8483 | 0.1014 | 10 |
| View 9 | 1.4023 | 0.2001 | 10 |
| View 10 | 0.7563 | 0.5174 | 6 |

**Top 3 views:** View 4 (R² = 0.7883), View 1 (R² = 0.6996), View 10 (R² = 0.5174)

#### Figures: Top 3 Gaussian Basis Regression Plots

| | |
|:-:|:-:|
| ![GB View 4](figures/q3iii_gaussian_view_4.png) | ![GB View 1](figures/q3iii_gaussian_view_1.png) |
| ![GB View 10](figures/q3iii_gaussian_view_10.png) | |

> **[YOUR ANALYSIS]** Discuss: how Gaussian basis regression compares to the polynomial LASSO regression (e.g., View 4 improved from R²=0.00 polynomial to R²=0.79 Gaussian basis), what the optimal orders suggest about the nonlinearity of the SSI-gen_impr relationship, and whether the improvement justifies the additional model complexity.
>
> ...

---

## Question 4: Image Alignment and Transformation Analysis [30 marks]

### 4.1 Rotation and Translation Values (Part i)

Rotation (degrees) and translation (pixel displacement) were extracted from the 2x3 ECC warp matrix using MOTION_EUCLIDEAN for each test image against its gold standard.

#### Table 18: Rotation Values (degrees) per Participant per View

| Participant | View 1 | View 2 | View 3 | View 4 | View 5 | View 6 | View 7 | View 8 | View 9 | View 10 |
|-------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|
| P1 | -3.76 | 0.70 | 4.68 | 4.90 | -7.54 | 1.91 | -4.19 | 1.76 | 7.71 | 5.70 |
| P2 | -1.37 | 0.41 | -5.05 | 9.10 | 1.68 | 1.14 | 5.81 | -3.86 | -0.60 | -0.12 |
| P3 | 0.29 | 4.34 | 1.27 | -18.44 | -2.63 | 0.13 | 1.62 | 1.98 | -4.06 | -0.00 |
| P4 | 0.28 | -1.69 | -0.21 | 1.65 | 4.02 | 2.98 | -1.87 | 8.30 | 1.33 | -0.31 |
| P5 | -1.97 | -3.34 | -3.01 | 3.07 | 1.66 | -1.46 | -2.52 | -4.23 | -2.04 | -0.24 |
| P6 | 2.65 | 3.09 | -8.26 | -1.10 | -5.90 | -3.43 | 3.21 | 1.18 | 3.21 | -1.92 |
| P7 | -3.66 | 6.46 | 4.70 | -8.04 | 0.27 | 3.17 | 4.78 | 0.00 | 4.00 | -2.57 |
| P8 | -0.61 | 2.91 | -17.30 | -5.07 | -1.73 | -9.74 | -11.63 | -3.75 | 5.41 | -0.63 |
| P9 | -1.04 | 5.34 | 0.31 | -7.66 | 3.32 | -0.28 | -4.26 | -0.99 | 2.86 | N/A |
| P10 | -19.14 | -0.80 | 2.02 | 5.06 | 6.04 | -3.07 | 3.97 | -2.71 | 0.27 | -1.99 |
| P11 | 0.44 | 3.02 | 2.22 | -6.57 | 5.39 | -0.65 | -0.81 | -1.46 | -8.26 | 2.75 |
| P12 | 5.69 | -1.60 | -1.38 | -2.97 | 1.63 | 3.07 | -1.03 | 0.01 | -0.14 | -1.74 |
| P13 | -1.60 | -0.30 | -9.73 | -4.29 | -4.33 | 3.01 | -7.48 | N/A | -2.07 | -2.86 |
| P14 | 7.84 | 0.15 | -1.02 | -4.90 | -2.87 | -2.25 | -2.32 | 4.11 | 5.29 | N/A |
| P15 | N/A | 1.48 | 1.17 | -2.53 | 5.69 | 2.40 | 0.44 | -1.63 | -2.61 | -2.72 |
| P16 | 2.22 | 4.02 | -0.77 | N/A | -0.95 | 1.20 | 5.97 | 4.98 | -1.18 | -1.85 |
| P17 | -3.41 | -4.76 | -4.88 | -5.71 | -1.55 | 0.00 | -1.53 | -1.31 | -7.50 | 0.61 |
| P18 | -3.72 | 2.73 | -6.06 | 2.76 | 3.60 | -1.83 | -5.81 | -4.75 | -3.22 | 1.43 |
| P19 | -3.31 | 1.29 | 4.72 | 2.29 | 0.80 | -0.82 | -1.10 | 1.20 | 0.39 | 0.31 |
| P20 | 1.64 | -0.63 | 0.33 | 19.85 | -2.99 | 3.61 | 0.01 | 8.17 | 0.84 | 5.94 |

#### Table 19: Translation Values (pixels) per Participant per View

| Participant | View 1 | View 2 | View 3 | View 4 | View 5 | View 6 | View 7 | View 8 | View 9 | View 10 |
|-------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|
| P1 | 29.92 | 6.15 | 15.51 | 16.66 | 29.70 | 7.71 | 19.53 | 8.38 | 30.83 | 21.74 |
| P2 | 2.91 | 5.50 | 13.38 | 33.98 | 4.82 | 4.72 | 24.93 | 12.70 | 7.49 | 0.97 |
| P3 | 3.08 | 15.87 | 15.63 | 62.63 | 11.35 | 1.05 | 15.56 | 16.73 | 10.72 | 1.34 |
| P4 | 5.85 | 8.82 | 4.77 | 9.52 | 4.81 | 22.84 | 3.02 | 36.71 | 12.92 | 0.82 |
| P5 | 10.22 | 13.99 | 11.47 | 9.03 | 15.06 | 3.36 | 12.10 | 16.10 | 9.88 | 2.62 |
| P6 | 16.65 | 9.98 | 31.88 | 4.37 | 39.17 | 10.50 | 17.27 | 6.56 | 4.73 | 6.24 |
| P7 | 22.15 | 35.05 | 22.97 | 33.73 | 9.86 | 12.58 | 27.17 | 1.86 | 17.90 | 7.29 |
| P8 | 4.07 | 12.03 | 46.67 | 16.68 | 2.50 | 35.68 | 47.30 | 23.34 | 17.04 | 5.04 |
| P9 | 9.83 | 19.02 | 0.60 | 32.79 | 12.41 | 4.18 | 19.43 | 4.91 | 11.26 | N/A |
| P10 | 72.02 | 1.75 | 4.04 | 14.49 | 21.52 | 10.70 | 17.50 | 12.57 | 5.11 | 7.77 |
| P11 | 2.63 | 13.73 | 9.38 | 25.37 | 25.03 | 11.80 | 5.84 | 11.13 | 28.65 | 10.96 |
| P12 | 36.26 | 7.07 | 9.33 | 9.02 | 10.36 | 12.12 | 1.84 | 3.55 | 0.40 | 8.03 |
| P13 | 7.37 | 4.23 | 33.82 | 15.23 | 12.75 | 8.20 | 28.44 | N/A | 6.64 | 11.18 |
| P14 | 39.46 | 2.57 | 2.85 | 22.47 | 7.16 | 8.73 | 2.20 | 15.88 | 27.51 | N/A |
| P15 | N/A | 5.89 | 4.33 | 7.10 | 15.87 | 8.85 | 3.36 | 7.30 | 8.68 | 18.81 |
| P16 | 9.12 | 12.77 | 5.68 | N/A | 2.72 | 6.43 | 33.83 | 20.20 | 8.95 | 16.23 |
| P17 | 16.73 | 19.72 | 16.64 | 17.04 | 2.98 | 1.79 | 5.14 | 5.37 | 40.18 | 1.37 |
| P18 | 15.43 | 11.95 | 29.37 | 11.16 | 13.85 | 4.85 | 29.28 | 14.60 | 12.69 | 15.51 |
| P19 | 17.22 | 7.09 | 19.76 | 11.85 | 13.84 | 2.87 | 9.58 | 6.20 | 2.11 | 4.52 |
| P20 | 11.45 | 1.03 | 2.83 | 70.52 | 17.90 | 12.07 | 0.60 | 48.38 | 13.55 | 26.13 |



### 4.2 Statistical Testing - Expert vs Novice (Part ii)

**H0:** No difference in alignment metric between expert and novice groups.
**H1:** Expert and novice groups differ in alignment metric.
**Significance level:** alpha = 0.05

#### Table 20: Mann-Whitney U Test Results - Rotation (degrees)

| View | Expert Mean | Novice Mean | U-stat | p-value | Significant? |
|------|-------------|-------------|--------|---------|--------------|
| View 1 | -1.08 | -1.25 | 36.00 | 0.6504 | No |
| View 2 | 1.42 | 0.99 | 50.00 | 0.7573 | No |
| View 3 | -0.84 | -2.34 | 50.00 | 0.7573 | No |
| View 4 | -1.27 | -0.81 | 47.00 | 0.7108 | No |
| View 5 | -1.21 | 0.93 | 35.00 | 0.4378 | No |
| View 6 | 0.63 | -0.41 | 52.00 | 0.6426 | No |
| View 7 | 0.98 | -1.97 | 59.00 | 0.3114 | No |
| View 8 | 0.73 | 0.16 | 47.00 | 0.7108 | No |
| View 9 | 1.37 | -0.76 | 58.00 | 0.3507 | No |
| View 10 | 0.08 | -0.07 | 39.00 | 1.0000 | No |

#### Table 21: Mann-Whitney U Test Results - Translation (pixels)

| View | Expert Mean | Novice Mean | U-stat | p-value | Significant? |
|------|-------------|-------------|--------|---------|--------------|
| View 1 | 12.97 | 20.13 | 34.00 | 0.5358 | No |
| View 2 | 13.62 | 9.14 | 58.00 | 0.3507 | No |
| View 3 | 16.52 | 14.25 | 58.00 | 0.3507 | No |
| View 4 | 24.27 | 21.14 | 43.00 | 0.9671 | No |
| View 5 | 16.40 | 12.22 | 50.00 | 0.7573 | No |
| View 6 | 8.96 | 9.87 | 42.00 | 0.8168 | No |
| View 7 | 17.08 | 15.72 | 51.00 | 0.6992 | No |
| View 8 | 14.15 | 14.45 | 45.00 | 0.8369 | No |
| View 9 | 13.50 | 14.06 | 48.00 | 0.8773 | No |
| View 10 | 5.86 | 11.41 | 17.00 | 0.0556 | No |

#### Table 22: Summary - Significant Views per Alignment Metric

| Metric | Significant Views (p < 0.05) | Count |
|--------|------------------------------|-------|
| Rotation | None | 0 / 10 |
| Translation | None | 0 / 10 |

> **[YOUR ANALYSIS - discuss significance results]** Discuss: why neither rotation nor translation significantly differentiates experts from novices across any view, what this implies about the relationship between image alignment and expertise level, potential explanations (e.g., both groups may produce images with similar misalignment distributions, the simulator environment may limit variability), and comparison with the similarity metric results from Q2.
>
> ...

### 4.3 Linear Regression - Alignment Metrics vs Quality Scores (Part iii)

#### Table 23: Linear Regression - Rotation -> crit_perc

| View | Slope | Intercept | RMSE | R² |
|------|-------|-----------|------|-----|
| View 1 | 0.5589 | 75.40 | 15.41 | 0.0349 |
| View 2 | 0.8877 | 73.99 | 14.52 | 0.0288 |
| View 3 | 1.5454 | 72.47 | 14.46 | 0.2468 |
| View 4 | -1.0618 | 60.36 | 17.06 | 0.1908 |
| View 5 | 0.7320 | 68.76 | 18.67 | 0.0216 |
| View 6 | 4.7140 | 56.55 | 24.48 | 0.2608 |
| View 7 | 2.3662 | 80.55 | 27.50 | 0.1247 |
| View 8 | 0.5019 | 64.03 | 15.06 | 0.0155 |
| View 9 | 1.4119 | 62.36 | 33.67 | 0.0276 |
| View 10 | 2.9248 | 69.17 | 23.30 | 0.0912 |

#### Table 24: Linear Regression - Rotation -> gen_impr

| View | Slope | Intercept | RMSE | R² |
|------|-------|-----------|------|-----|
| View 1 | 0.0444 | 2.81 | 0.73 | 0.0931 |
| View 2 | 0.0744 | 2.65 | 0.76 | 0.0712 |
| View 3 | 0.0627 | 2.55 | 0.67 | 0.1990 |
| View 4 | -0.0269 | 1.89 | 0.80 | 0.0650 |
| View 5 | 0.0148 | 2.20 | 0.91 | 0.0038 |
| View 6 | 0.1211 | 2.14 | 0.99 | 0.1251 |
| View 7 | 0.1030 | 2.50 | 1.15 | 0.1335 |
| View 8 | -0.0350 | 1.70 | 0.89 | 0.0216 |
| View 9 | 0.0921 | 1.79 | 1.52 | 0.0556 |
| View 10 | 0.0777 | 2.33 | 1.07 | 0.0324 |

#### Table 25: Linear Regression - Translation -> crit_perc

| View | Slope | Intercept | RMSE | R² |
|------|-------|-----------|------|-----|
| View 1 | -0.1538 | 77.43 | 15.47 | 0.0267 |
| View 2 | -0.7265 | 82.78 | 13.64 | 0.1434 |
| View 3 | -0.3199 | 74.48 | 16.20 | 0.0543 |
| View 4 | 0.0291 | 60.75 | 18.96 | 0.0007 |
| View 5 | -0.0259 | 69.24 | 18.87 | 0.0002 |
| View 6 | 0.1668 | 54.74 | 28.45 | 0.0020 |
| View 7 | -1.1667 | 97.23 | 25.56 | 0.2438 |
| View 8 | -0.2279 | 67.48 | 14.95 | 0.0292 |
| View 9 | 0.4549 | 56.03 | 33.83 | 0.0184 |
| View 10 | 1.2439 | 57.63 | 22.63 | 0.1427 |

#### Table 26: Linear Regression - Translation -> gen_impr

| View | Slope | Intercept | RMSE | R² |
|------|-------|-----------|------|-----|
| View 1 | -0.0126 | 2.97 | 0.73 | 0.0753 |
| View 2 | -0.0019 | 2.75 | 0.79 | 0.0003 |
| View 3 | -0.0174 | 2.70 | 0.72 | 0.0789 |
| View 4 | 0.0081 | 1.73 | 0.81 | 0.0299 |
| View 5 | 0.0085 | 2.08 | 0.91 | 0.0074 |
| View 6 | 0.0023 | 2.11 | 1.06 | 0.0003 |
| View 7 | -0.0279 | 2.85 | 1.19 | 0.0786 |
| View 8 | -0.0150 | 1.90 | 0.88 | 0.0364 |
| View 9 | 0.0263 | 1.42 | 1.54 | 0.0292 |
| View 10 | 0.0178 | 2.17 | 1.08 | 0.0148 |

#### Top 3 Best Performing Views (across all combinations, by R²)

| Rank | View | Combination | R² | RMSE |
|------|------|-------------|-----|------|
| 1 | View 6 | Rotation -> crit_perc | 0.2608 | 24.48 |
| 2 | View 3 | Rotation -> crit_perc | 0.2468 | 14.46 |
| 3 | View 7 | Translation -> crit_perc | 0.2438 | 25.56 |

#### Figures: Top 3 Linear Regression Plots

| | |
|:-:|:-:|
| ![Rotation->CP View 6](figures/q4iii_Rotation_crit_perc_view_6.png) | ![Rotation->CP View 3](figures/q4iii_Rotation_crit_perc_view_3.png) |
| ![Translation->CP View 7](figures/q4iii_Translation_crit_perc_view_7.png) | |

> **[YOUR ANALYSIS]** Discuss: the generally very low R² values (all below 0.27) indicating weak linear relationships between alignment metrics and quality scores, why rotation shows slightly better predictive power than translation for crit_perc, and what the poor regression performance suggests about whether simple rigid transformation parameters are sufficient to capture image quality.
>
> ...

---

## Summary

| Question | Key Finding |
|----------|-------------|
| Q1 | View 1 has the highest Pearson correlation (r = 0.9274) between the two quality scores; linear regression achieves R² = 0.86 for View 1 |
| Q2 | MI and CS are the best differentiating similarity metrics (each significant in 2/10 views); SSI significant in only 1/10 views |
| Q3 | MI -> gen_impr at View 1 achieves the best polynomial regression (R² = 0.55); Gaussian basis regression substantially improves fit (View 4 R² = 0.79) |
| Q4 | Neither rotation nor translation significantly differentiates experts from novices in any view; low R² values across all linear regressions |
