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

The Pearson correlation coefficient measures the strength and direction of the linear relationship between two variables. For paired observations $(x_i, y_i)$:

$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2 \sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

where $\bar{x}$ and $\bar{y}$ are the sample means. Values range from $-1$ (perfect negative) to $+1$ (perfect positive), with $0$ indicating no linear association. Here, $x$ = general impression and $y$ = criteria percentage for each view.

<h4 align="center">Table 1: Pearson Correlation Between General Impression and Criteria Percentage per View</h4>

<div align="center">

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

</div>

<p align="center"><strong>Highest agreement: View 1 (r = 0.9274)</strong></p>

Pearson's correlation coefficient measures the linear relationship between two variables; in this case measuring the relationship between the general impression and criteria percentage for each view. As previously stated, a result of +1 demonstrates a perfect positive correlation. The highest agreement is View 1 with an r value of 0.9274. High scores represent a strong correlation between the general impression — a holistic score given by an assessor, and the criteria percentage — a checklist-based score derived from predefined anatomical criteria. Therefore a high Pearson's r — like that given by View 1 — is representative of a view where the assessor's overall quality impression matches closely with the checklist-derived value.
View 9 is nearly tied with View 1 at r = 0.9268 - these 2 views show essentially equivalent agreement between the scoring methods. At the other end, View 4 has the lowest correlation (r = 0.6545), suggesting this view has more ambiguous quality criteria where the assessor perception and the checklist diverge. Notably, all 10 views are statistically significant (p < 0.05), confirming a genuine positive linear relationship between the two scores across every view - The p-value test is used to test the null hypothesis; it gives the probability of observing a similar correlation if the null hypothesis is true (i.e., that there is no relationship). The r values range from 0.6545 to 0.9274 — all positive and at least moderate-strong — indicating that holistic and checklist-based assessments broadly agree across all TOE views, just to varying degrees. The views cluster naturally: very strong agreement (Views 1 and 9, r ≈ 0.93), strong agreement (Views 5 and 10, r ≈ 0.87), and moderate agreement (View 4, r ≈ 0.65), suggesting some views have more straightforward quality criteria than others.

### 1.2 Linear Regression Analysis (Part ii)

A simple linear regression model is fitted for each view:

$$\hat{y}_i = \beta_0 + \beta_1 x_i$$

where $x_i$ = general impression score (independent variable), $\hat{y}_i$ = predicted criteria percentage (dependent variable), $\beta_1$ = slope, and $\beta_0$ = intercept. The parameters are estimated by ordinary least squares, minimising $\sum(y_i - \hat{y}_i)^2$.

Model performance is evaluated using:

**Root Mean Square Error (RMSE):**

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**Coefficient of Determination ($R^2$):**

$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

where $R^2 = 1$ indicates a perfect fit and $R^2 = 0$ indicates the model explains no variance beyond the mean.

<h4 align="center">Table 2: Linear Regression Results (gen_impr -> crit_perc) per View</h4>

<div align="center">

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

</div>

The linear regression results are similar to the Pearson correlation findings. View 1 achieves the best fit with the lowest RMSE (5.87) and highest R² (0.86), meaning the model explains 86% of the variance in criteria percentage from general impression alone. View 9 is again close  with an R² = 0.8590, though its higher RMSE (12.82) reflects greater spread in the criteria percentage values for that view.

On the other hand, View 4 has the worst fit (R² = 0.4284, RMSE = 14.34), consistent with its lowest Pearson correlation. Views 6 and 7 also show high RMSE values (17.1 and 17.71 respectively), suggesting these views have wider variability in criteria percentage that a simple linear model struggles to capture.

The slope values are broadly consistent across views, ranging from 12.03 (View 8) to 21.55 (View 6). This means that for each 1-point increase in general impression, the criteria percentage increases by roughly 12–22 percentage points depending on the view. The intercepts vary more widely — View 8 has the highest intercept (43.95) with the shallowest slope, suggesting that even participants scoring 0 on general impression still meet a substantial proportion of the checklist criteria for this view. Conversely, View 6 has a low intercept (10.35) but the steepest slope (21.55), indicating the quality scores for this view span a wider range and are more sensitive to changes in general impression.

### 1.3 True vs Estimated Plots - Three Best Performing Views (Part iii)

The three best performing views by R² are: **View 1** (R² = 0.8600), **View 9** (R² = 0.8590), and **View 10** (R² = 0.7567).

<h4 align="center">Figure 1: View 1 - True vs Estimated Criteria Percentage</h4>

<p align="center"><img src="figures/q1_true_vs_estimated_view_1.png" alt="Figure 1" width="70%"></p>

<h4 align="center">Figure 2: View 9 - True vs Estimated Criteria Percentage</h4>

<p align="center"><img src="figures/q1_true_vs_estimated_view_9.png" alt="Figure 2" width="70%"></p>

<h4 align="center">Figure 3: View 10 - True vs Estimated Criteria Percentage</h4>

<p align="center"><img src="figures/q1_true_vs_estimated_view_10.png" alt="Figure 3" width="70%"></p>

<p align="center"><em>Points are colour-coded by criteria percentage range: blue = CP 40-70%, green = CP 70-100%. The red dashed line represents perfect prediction (y = x).</em></p>

Across the three best-performing views, the model tends to predict differently for the two criteria percentage ranges. In View 1, the blue points (CP 40–70%) tend to sit above the y = x line, meaning the model overpredicts the criteria percentage for lower-quality images. The green points (CP 70–100%) cluster more tightly around the perfect prediction line, indicating the model is more accurate for higher-scoring images. This suggests the linear relationship is stronger at the upper end of the quality spectrum, where high general impression reliably corresponds to high criteria percentage.

In View 10, this pattern becomes more pronounced — the blue points (40–70%) are consistently overpredicted, while the green points (70–100%) are underpredicted. The model regresses toward the mean, pulling extreme values toward the centre: Linear Regression tends to compress predictions toward the average, overpredicting low-scoring images and underpredicting high-scoring ones.

View 9 shows a different distribution, with scores clustering at the extremes (very low and very high) rather than in the mid-range. Most high-scoring predictions sit close to the line, though one notable outlier at approximately 80% true criteria percentage is severely underpredicted. The absence of points in the 40–70% range for View 9 suggests this view produces either clearly good or clearly poor images, with little ambiguity in between.

---

## Question 2: Image Similarity Metrics [20 marks]

Three similarity metrics are computed between each participant's test image and the gold standard image for the corresponding view.

**Structural Similarity Index (SSI)** — Equation (1): Compares luminance, contrast, and structure:

$$\text{SSI}(a, b) = l(a, b) \cdot c(a, b) \cdot s(a, b)$$

where:

$$l(a,b) = \frac{2\mu_a\mu_b + C_l}{\mu_a^2 + \mu_b^2 + C_l}, \quad c(a,b) = \frac{2\sigma_a\sigma_b + C_c}{\sigma_a^2 + \sigma_b^2 + C_c}, \quad s(a,b) = \frac{\sigma_{ab} + C_s}{\sigma_a\sigma_b + C_s}$$

$\mu$ = pixel mean, $\sigma$ = standard deviation, $\sigma_{ab}$ = covariance, and $C_l, C_c, C_s$ are stabilising constants. Implemented via `skimage.metrics.structural_similarity`.

**Mutual Information (MI)** — Equation (2): Measures shared information via entropy:

$$\text{MI}(a, b) = H(a) + H(b) - H(a, b)$$

where $H(a) = -\sum_x p_a(x) \log p_a(x)$ is the marginal entropy and $H(a,b) = -\sum_{x,y} p_{ab}(x,y) \log p_{ab}(x,y)$ is the joint entropy. Implemented via `sklearn.metrics.mutual_info_score`.

**Cosine Similarity (CS)** — Equation (3): Measures the cosine of the angle between two image vectors:

$$\text{CS}(I_a, I_b) = \frac{I_a \cdot I_b}{\|I_a\| \, \|I_b\|}$$

where $I_a, I_b$ are the flattened image vectors and $\|I\| = \sqrt{I \cdot I}$.

### 2.1 Top 3 Participants per View by Similarity Metric (Part i)

<h4 align="center">Table 3: Top 3 Participants by SSI per View</h4>

<div align="center">

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

</div>

<h4 align="center">Table 4: Top 3 Participants by MI per View</h4>

<div align="center">

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

</div>

<h4 align="center">Table 5: Top 3 Participants by CS per View</h4>

<div align="center">

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

</div>

Participant 2 (P2, expert) stands out as the most consistently high-performing participant, appearing in the top 3 across the majority of views for all three metrics — including first place for SSI in Views 1, 4, 5, 6, 8, and 10. This suggests P2 produces images that are structurally, informationally, and directionally closest to the gold standard across most views.

However, experts do not dominate the rankings overall. Several novice participants also rank highly: P14 (novice) appears frequently in the SSI top 3 for Views 2, 3, and 4, and P17 and P19 (both novices) rank in the top 3 for multiple views across MI and CS. This indicates that some novice participants are capable of producing images with comparable similarity to the gold standard, at least for certain views.

The three metrics largely agree on which participants produce the best images, particularly MI and CS which tend to rank the same participants in the same order. This is consistent with the high MI-CS correlation observed later in Question 3. SSI occasionally diverges — for example, in View 9, SSI highlights P8 while MI and CS highlight P19. Overall, the agreement between metrics suggests the rankings reflect genuine image quality rather than artefacts of a particular metric.

### 2.2 Statistical Testing - Expert vs Novice (Part ii)

The Mann-Whitney U test is a non-parametric test comparing two independent groups. It is chosen here because the sample sizes are small and we cannot assume normality. The test statistic is:

$$U = \sum_{i=1}^{n_1} \sum_{j=1}^{n_2} S(x_i, y_j), \quad S(x_i, y_j) = \begin{cases} 1 & \text{if } x_i > y_j \\ 0.5 & \text{if } x_i = y_j \\ 0 & \text{if } x_i < y_j \end{cases}$$

where $\{x_i\}$ are expert values ($n_1 = 7$) and $\{y_j\}$ are novice values ($n_2 = 13$). A two-sided test is performed with:

**$H_0$:** No difference in similarity metric between expert and novice groups.
**$H_1$:** Expert and novice groups differ in similarity metric.
**Significance level:** $\alpha = 0.05$

<h4 align="center">Table 6: Mann-Whitney U Test Results - SSI</h4>

<div align="center">

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

</div>

<h4 align="center">Table 7: Mann-Whitney U Test Results - MI</h4>

<div align="center">

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

</div>

<h4 align="center">Table 8: Mann-Whitney U Test Results - CS</h4>

<div align="center">

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

</div>

<p align="center"><strong>Best differentiating metric: MI and CS</strong> (each significant in 2 / 10 views; SSI significant in 1 / 10)</p>

The Mann-Whitney U test was chosen because the sample sizes are small (7 experts, 13 novices) and normality of the score distributions cannot be assumed. It is a non-parametric alternative to the independent samples t-test, comparing the rank distributions of the two groups rather than their means.

Across all three metrics, the majority of views show no statistically significant difference between experts and novices (p > 0.05). Only View 8 is significant for all three metrics (SSI p = 0.0037, MI p = 0.0283, CS p = 0.0283), and View 1 is additionally significant for MI (p = 0.0052) and CS (p = 0.0037) but not SSI (p = 0.1956). In both cases, the expert group has higher mean similarity values, indicating their images more closely resemble the gold standard.

The low number of significant results suggests that similarity metrics alone do not strongly differentiate expertise level for most views. This could be because both groups are imaging the same simulated heart using the same equipment, limiting the range of possible image variation. The simulator environment may constrain the degree to which expertise affects image content, particularly for views where the probe positioning is relatively straightforward.

MI and CS are tied as the best differentiating metrics, each achieving significance in 2 out of 10 views, while SSI is significant in only 1 view. MI and CS may be more sensitive to the specific content differences between expert and novice images because they directly measure pixel-level information overlap, whereas SSI captures broader structural properties (luminance, contrast, structure) that may be similar regardless of expertise.

---

## Question 3: Regression Analysis of Similarity Metrics [30 marks]

### 3.1 Correlation Between Similarity Metric Pairs (Part i)

The Pearson correlation coefficient (as defined in Section 1.1) is computed between each pair of similarity metrics (SSI-MI, SSI-CS, MI-CS) for each view to assess their agreement.

<h4 align="center">Table 9: Pearson Correlation Between SSI-MI, SSI-CS, and MI-CS per View</h4>

<div align="center">

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

</div>

<div align="center">

**Highest agreement per pair:**
- SSI-MI: **View 10** (r = 0.8252)
- SSI-CS: **View 10** (r = 0.8540)
- MI-CS: **View 4** (r = 0.9962)

</div>

The most striking pattern is the consistently very high correlation between MI and CS across all views, with r values ranging from 0.6421 (View 2) to 0.9962 (View 4) and exceeding 0.98 in 8 out of 10 views. This near-perfect agreement indicates that MI and CS are capturing essentially the same information about image similarity — they are largely redundant metrics. This makes intuitive sense: both measure pixel-level overlap between two images, just through different mathematical formulations (entropy-based vs vector angle).

In contrast, SSI shows weaker and more variable correlations with both MI and CS. The SSI-MI and SSI-CS correlations range from near-zero to 0.85, suggesting SSI captures different aspects of image quality. SSI compares luminance, contrast, and structural patterns rather than raw pixel overlap, making it sensitive to different image properties.

View 9 is a notable outlier, with SSI-MI correlation of just 0.0144 and SSI-CS correlation of -0.0703. This near-zero correlation means SSI provides entirely independent information from MI and CS for this view. View 10 shows the opposite, with the highest SSI-MI (0.8252) and SSI-CS (0.8540) correlations, suggesting the three metrics converge more for this particular view.

The practical implication is that MI and CS provide little additional information beyond each other, while SSI offers a complementary perspective. A comprehensive assessment of image similarity benefits from using both SSI and at least one of MI/CS, rather than relying on any single metric.

### 3.2 Polynomial Regression with LASSO Regularization (Part ii)

For each similarity metric (SSI, MI, CS) as the independent variable and each quality score (crit_perc, gen_impr) as the dependent variable, a polynomial regression model up to degree $d$ is fitted:

$$\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \cdots + \beta_d x^d$$

where $x$ is the similarity metric value and $d \in \{1, 2, \ldots, 7\}$. To prevent overfitting, **LASSO (Least Absolute Shrinkage and Selection Operator)** regularization is applied. LASSO minimises:

$$\min_{\beta} \lbrace \frac{1}{2n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{d} |\beta_j| \rbrace$$

where $\lambda \geq 0$ is the regularization parameter selected via 5-fold cross-validation (`LassoCV`). The $L_1$ penalty $\lambda \sum |\beta_j|$ drives small coefficients to exactly zero, performing automatic feature selection. Coefficients with $|\beta_j| < 0.01$ are treated as not contributing.

The optimal polynomial degree is selected by comparing the **cross-validated RMSE** (from `LassoCV`'s internal CV at the optimal $\lambda$) across degrees $1$ to $7$, choosing the degree that minimises generalisation error. RMSE and $R^2$ are then reported on the full training set for the selected degree.

LASSO was selected over Ridge and Elastic Net regression for its ability to perform automatic feature selection through the L1 penalty. When fitting a polynomial up to degree 7, many of the higher-order terms may not meaningfully contribute to the model. LASSO's L1 penalty drives the coefficients of irrelevant terms to exactly zero, effectively removing them from the model. This is particularly desirable here because the coursework specifies that coefficients smaller than 0.01 should be considered as not contributing — LASSO naturally enforces this sparsity.

Ridge regression, by contrast, uses an L2 penalty that shrinks coefficients toward zero but never sets them exactly to zero. This means all polynomial terms would remain in the model regardless of their relevance, making it harder to identify which terms genuinely contribute. Elastic Net combines both L1 and L2 penalties, but introduces an additional hyperparameter (the L1/L2 mixing ratio) without clear benefit in this context where the primary goal is feature elimination rather than handling correlated predictors.

The regularization parameter lambda is selected via 5-fold cross-validation using LassoCV, and the optimal polynomial degree is chosen by comparing the cross-validated RMSE across degrees 1 to 7, minimising generalisation error rather than training error.

<h4 align="center">Table 10: LASSO Polynomial Regression -> crit_perc</h4>

<div align="center">
<table>
<tr>
<td align="center">

<strong>SSI</strong>

| View | RMSE | R² | Degree |
|------|------|-----|--------|
| View 1 | 13.12 | 0.3000 | 7 |
| View 2 | 11.81 | 0.3584 | 4 |
| View 3 | 14.79 | 0.2123 | 7 |
| View 4 | 17.38 | 0.1604 | 1 |
| View 5 | 16.41 | 0.2438 | 5 |
| View 6 | 25.40 | 0.2047 | 7 |
| View 7 | 29.39 | 0.0000 | 1 |
| View 8 | 14.31 | 0.1111 | 7 |
| View 9 | 31.41 | 0.1535 | 7 |
| View 10 | 21.52 | 0.2250 | 1 |

</td>
<td align="center">

<strong>MI</strong>

| View | RMSE | R² | Degree |
|------|------|-----|--------|
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

</td>
<td align="center">

<strong>CS</strong>

| View | RMSE | R² | Degree |
|------|------|-----|--------|
| View 1 | 10.71 | 0.5338 | 1 |
| View 2 | 14.74 | 0.0000 | 1 |
| View 3 | 15.40 | 0.1463 | 4 |
| View 4 | 17.59 | 0.1404 | 4 |
| View 5 | 17.67 | 0.1236 | 1 |
| View 6 | 23.22 | 0.3351 | 1 |
| View 7 | 25.63 | 0.2395 | 1 |
| View 8 | 12.69 | 0.3009 | 1 |
| View 9 | 34.14 | 0.0000 | 1 |
| View 10 | 21.85 | 0.2010 | 1 |

</td>
</tr>
</table>
</div>

<h4 align="center">Table 11: LASSO Polynomial Regression -> gen_impr</h4>

<div align="center">
<table>
<tr>
<td align="center">

<strong>SSI</strong>

| View | RMSE | R² | Degree |
|------|------|-----|--------|
| View 1 | 0.6190 | 0.3428 | 7 |
| View 2 | 0.7572 | 0.0715 | 1 |
| View 3 | 0.6912 | 0.1580 | 7 |
| View 4 | 0.8225 | 0.0000 | 1 |
| View 5 | 0.6788 | 0.4426 | 3 |
| View 6 | 0.8804 | 0.3051 | 7 |
| View 7 | 1.1826 | 0.0852 | 2 |
| View 8 | 0.8949 | 0.0000 | 1 |
| View 9 | 1.5680 | 0.0000 | 4 |
| View 10 | 1.0248 | 0.1139 | 6 |

</td>
<td align="center">

<strong>MI</strong>

| View | RMSE | R² | Degree |
|------|------|-----|--------|
| View 1 | 0.5145 | 0.5458 | 1 |
| View 2 | 0.7813 | 0.0116 | 1 |
| View 3 | 0.7200 | 0.0865 | 1 |
| View 4 | 0.6994 | 0.2769 | 2 |
| View 5 | 0.8098 | 0.2068 | 1 |
| View 6 | 0.9157 | 0.2484 | 1 |
| View 7 | 1.0906 | 0.2220 | 1 |
| View 8 | 0.8949 | 0.0000 | 1 |
| View 9 | 1.5680 | 0.0000 | 1 |
| View 10 | 0.9904 | 0.1723 | 1 |

</td>
<td align="center">

<strong>CS</strong>

| View | RMSE | R² | Degree |
|------|------|-----|--------|
| View 1 | 0.5263 | 0.5248 | 1 |
| View 2 | 0.7859 | 0.0000 | 1 |
| View 3 | 0.6153 | 0.3329 | 2 |
| View 4 | 0.6826 | 0.3112 | 4 |
| View 5 | 0.8001 | 0.2255 | 1 |
| View 6 | 0.9144 | 0.2505 | 1 |
| View 7 | 1.0717 | 0.2488 | 1 |
| View 8 | 0.8949 | 0.0000 | 2 |
| View 9 | 1.5680 | 0.0000 | 4 |
| View 10 | 1.0002 | 0.1560 | 1 |

</td>
</tr>
</table>
</div>

<h4 align="center">Best 3 Performing Views (Top 3 per combination plotted)</h4>

<p align="center"><strong>SSI -> crit_perc:</strong> Views 2 (R²=0.36), 1 (R²=0.30), 5 (R²=0.24)</p>

<div align="center">

| | | |
|:-:|:-:|:-:|
| <img src="figures/q3ii_SSI_crit_perc_view_1.png" alt="q3ii_SSI_crit_perc_view_1" width="70%"> | <img src="figures/q3ii_SSI_crit_perc_view_2.png" alt="q3ii_SSI_crit_perc_view_2" width="70%"> | <img src="figures/q3ii_SSI_crit_perc_view_5.png" alt="q3ii_SSI_crit_perc_view_5" width="70%"> |

</div>

<p align="center"><strong>SSI -> gen_impr:</strong> Views 5 (R²=0.44), 1 (R²=0.34), 6 (R²=0.31)</p>
<div align="center">

| | | |
|:-:|:-:|:-:|
| <img src="figures/q3ii_SSI_gen_impr_view_5.png" alt="q3ii_SSI_gen_impr_view_5" width="70%"> | <img src="figures/q3ii_SSI_gen_impr_view_1.png" alt="q3ii_SSI_gen_impr_view_1" width="70%"> | <img src="figures/q3ii_SSI_gen_impr_view_6.png" alt="q3ii_SSI_gen_impr_view_6" width="70%"> |

</div>

<p align="center"><strong>MI -> crit_perc:</strong> Views 1 (R²=0.54), 6 (R²=0.30), 8 (R²=0.28)</p>

<div align="center">

| | | |
|:-:|:-:|:-:|
| <img src="figures/q3ii_MI_crit_perc_view_1.png" alt="q3ii_MI_crit_perc_view_1" width="70%"> | <img src="figures/q3ii_MI_crit_perc_view_6.png" alt="q3ii_MI_crit_perc_view_6" width="70%"> | <img src="figures/q3ii_MI_crit_perc_view_8.png" alt="q3ii_MI_crit_perc_view_8" width="70%"> |

</div>

<p align="center"><strong>MI -> gen_impr:</strong> Views 1 (R²=0.55), 4 (R²=0.28), 6 (R²=0.25)</p>

<div align="center">

| | | |
|:-:|:-:|:-:|
| <img src="figures/q3ii_MI_gen_impr_view_1.png" alt="q3ii_MI_gen_impr_view_1" width="70%"> | <img src="figures/q3ii_MI_gen_impr_view_4.png" alt="q3ii_MI_gen_impr_view_4" width="70%"> | <img src="figures/q3ii_MI_gen_impr_view_6.png" alt="q3ii_MI_gen_impr_view_6" width="70%"> |

</div>

<p align="center"><strong>CS -> crit_perc:</strong> Views 1 (R²=0.53), 6 (R²=0.34), 8 (R²=0.30)</p>

<div align="center">

| | | |
|:-:|:-:|:-:|
| <img src="figures/q3ii_CS_crit_perc_view_1.png" alt="q3ii_CS_crit_perc_view_1" width="70%"> | <img src="figures/q3ii_CS_crit_perc_view_6.png" alt="q3ii_CS_crit_perc_view_6" width="70%"> | <img src="figures/q3ii_CS_crit_perc_view_8.png" alt="q3ii_CS_crit_perc_view_8" width="70%"> |

</div>

<p align="center"><strong>CS -> gen_impr:</strong> Views 1 (R²=0.52), 3 (R²=0.33), 4 (R²=0.31)</p>

<div align="center">

| | | |
|:-:|:-:|:-:|
| <img src="figures/q3ii_CS_gen_impr_view_1.png" alt="q3ii_CS_gen_impr_view_1" width="70%"> | <img src="figures/q3ii_CS_gen_impr_view_3.png" alt="q3ii_CS_gen_impr_view_3" width="70%"> | <img src="figures/q3ii_CS_gen_impr_view_4.png" alt="q3ii_CS_gen_impr_view_4" width="70%"> |

</div>

Across all metric-score combinations, MI and CS consistently produce the strongest regression fits. The best overall result is MI -> gen_impr at View 1 (R² = 0.55), followed closely by MI -> crit_perc at View 1 (R² = 0.54) and CS -> crit_perc at View 1 (R² = 0.53). View 1 performs well across nearly every combination, consistent with its strong Pearson correlation in Q1 and significant Mann-Whitney results in Q2 — this view appears to have the most informative relationship between image content and quality scores.

A notable observation is that many MI and CS regressions collapse to degree 1 (linear), with LASSO zeroing all higher-order polynomial coefficients. This suggests the relationship between these metrics and quality scores is predominantly linear, and higher-order terms do not improve generalisation. SSI regressions are more likely to retain higher degrees (5–7), indicating a more complex, nonlinear relationship between structural similarity and quality scores.

The R² values are generally low across all combinations — the best is 0.55 and most fall below 0.35. This indicates that individual similarity metrics explain only a limited portion of the variance in quality scores. This is not unexpected: image quality as assessed by human evaluators is likely influenced by factors beyond what any single pixel-based metric captures, such as anatomical landmark visibility and clinical interpretability.

The performance of crit_perc and gen_impr as dependent variables is broadly comparable, with no consistent advantage for either. This aligns with the strong Pearson correlation between the two scores found in Q1 — since the two quality scores are closely related, predicting one is approximately as easy as predicting the other.

### 3.3 Gaussian Basis Regression - SSI -> gen_impr (Part iii)

Instead of polynomial features, the input $x$ (SSI) is transformed using Gaussian basis functions. For a given order $M$, the $j$-th basis function is:

$$\phi_j(x) = \exp\!\bigg(-\frac{(x - \mu_j)^2}{2s^2}\bigg), \quad j = 1, \ldots, M$$

where $\mu_j$ are $M$ centres equally spaced across the range of $x$, and $s = (\max(x) - \min(x)) / M$ is the basis width. The regression model becomes:

$$\hat{y} = \beta_0 + \sum_{j=1}^{M} \beta_j \, \phi_j(x)$$

LASSO regularization with 5-fold cross-validation is applied (as in Section 3.2). The optimal order $M \in \{2, 3, \ldots, 10\}$ is selected by minimising the cross-validated RMSE.

<h4 align="center">Table 12: Gaussian Basis Regression Results (SSI -> gen_impr)</h4>

<div align="center">

| View | RMSE | R² | Optimal Order |
|------|------|-----|---------------|
| View 1 | 0.4535 | 0.6472 | 8 |
| View 2 | 0.7534 | 0.0809 | 2 |
| View 3 | 0.6775 | 0.1912 | 4 |
| View 4 | 0.3785 | 0.7883 | 10 |
| View 5 | 0.6747 | 0.4494 | 2 |
| View 6 | 0.8337 | 0.3770 | 4 |
| View 7 | 0.9067 | 0.4622 | 5 |
| View 8 | 0.8949 | 0.0000 | 2 |
| View 9 | 1.4337 | 0.1639 | 6 |
| View 10 | 0.7563 | 0.5174 | 6 |

</div>

<p align="center"><strong>Top 3 views:</strong> View 4 (R² = 0.7883), View 1 (R² = 0.6472), View 10 (R² = 0.5174)</p>

<h4 align="center">Figures: Top 3 Gaussian Basis Regression Plots</h4>

<div align="center">

| | | |
|:-:|:-:|:-:|
| <img src="figures/q3iii_gaussian_view_4.png" alt="GB View 4" width="70%"> | <img src="figures/q3iii_gaussian_view_1.png" alt="GB View 1" width="70%"> | <img src="figures/q3iii_gaussian_view_10.png" alt="GB View 10" width="70%"> |

</div>

The Gaussian basis regression substantially outperforms the polynomial LASSO regression for several views. The most dramatic improvement is View 4, which achieved R² = 0.00 with polynomial regression (the model explained none of the variance) but R² = 0.79 with Gaussian basis functions at order 10. View 1 improved from R² = 0.34 to R² = 0.65, and View 10 from R² = 0.11 to R² = 0.52. These improvements suggest the relationship between SSI and general impression is better captured by localised, bell-shaped basis functions than by global polynomial terms.

The optimal orders vary across views — from 2 (Views 2, 5, 8) to 10 (View 4) — indicating different levels of nonlinearity in the SSI-gen_impr relationship depending on the view. Lower orders suggest a simpler, smoother relationship, while higher orders allow the model to fit more localised patterns in the data. Cross-validated model selection was used to determine the optimal order, avoiding overfitting by selecting the order that minimises generalisation error rather than training error.

However, the improvement is not universal. Views 8 and 9 remain poorly predicted (R² = 0.00 and R² = 0.16 respectively), suggesting that for these views, SSI simply does not contain sufficient information to predict general impression regardless of the regression approach. The Gaussian basis approach justifies its additional complexity for views where clear nonlinear patterns exist, but cannot compensate where the underlying relationship between the metrics is weak.

---

## Question 4: Image Alignment and Transformation Analysis [30 marks]

### 4.1 Rotation and Translation Values (Part i)

The **Enhanced Correlation Coefficient (ECC)** algorithm (`cv2.findTransformECC`) is used to estimate a rigid (Euclidean) transformation aligning each test image to its gold standard. The motion model `MOTION_EUCLIDEAN` yields a $2 \times 3$ warp matrix:

$$M = \begin{bmatrix} \cos\theta & -\sin\theta & t_x \\ \sin\theta & \cos\theta & t_y \end{bmatrix}$$

with parameters: 500 iterations, termination threshold $\varepsilon = 10^{-10}$.

**Rotation** (degrees) is extracted from the matrix elements:

$$\theta = \arctan2(m_{21}, m_{11}) \times \frac{180}{\pi}$$

**Translation** (total pixel displacement) is the Euclidean norm of the translation vector:

$$d = \sqrt{t_x^2 + t_y^2} = \sqrt{m_{13}^2 + m_{23}^2}$$

<h4 align="center">Table 13: Rotation Values (degrees) per Participant per View</h4>

<div align="center">

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

</div>

<h4 align="center">Table 14: Translation Values (pixels) per Participant per View</h4>

<div align="center">

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

</div>



### 4.2 Statistical Testing - Expert vs Novice (Part ii)

The Mann-Whitney U test (as defined in Section 2.2) is applied to compare expert and novice groups on rotation and translation for each view.

**$H_0$:** No difference in alignment metric between expert and novice groups.
**$H_1$:** Expert and novice groups differ in alignment metric.
**Significance level:** $\alpha = 0.05$

<h4 align="center">Table 15: Mann-Whitney U Test Results - Rotation (degrees)</h4>

<div align="center">

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

</div>

<h4 align="center">Table 16: Mann-Whitney U Test Results - Translation (pixels)</h4>

<div align="center">

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

</div>

Neither rotation nor translation produces a statistically significant difference between the expert and novice groups for any of the 10 views (all p > 0.05). This is in contrast to the similarity metric results from Q2, where MI and CS each achieved significance in 2 views and SSI in 1 view. The alignment parameters appear to be even less discriminative of expertise than the similarity metrics.

The mean rotation values are close to zero for both groups across most views, suggesting that neither experts nor novices systematically over- or under-rotate the probe. The mean translation values are also comparable between groups, with overlapping ranges in every view. This implies that the physical positioning of the probe — at least as captured by rigid transformation parameters — is not what differentiates expert from novice performance.

One possible explanation is that the HeartWorks simulator constrains the range of probe movements, limiting the variability in rotation and translation between participants regardless of experience. Another is that image quality depends more on subtle aspects of probe orientation and image plane selection that are not fully captured by a simple Euclidean (rigid) transformation model. The ECC algorithm estimates the best global rotation and translation to align two images, but cannot account for differences in image content that arise from probe angulation, depth, or gain settings.

### 4.3 Linear Regression - Alignment Metrics vs Quality Scores (Part iii)

Simple linear regression (as defined in Section 1.2: $\hat{y} = \beta_0 + \beta_1 x$) is fitted for each combination of alignment metric (rotation or translation) as independent variable and quality score (crit_perc or gen_impr) as dependent variable, yielding $2 \times 2 \times 10 = 40$ models.

<h4 align="center">Table 17: Linear Regression - Rotation -> crit_perc</h4>

<div align="center">

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

</div>

<h4 align="center">Table 18: Linear Regression - Rotation -> gen_impr</h4>

<div align="center">

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

</div>

<h4 align="center">Table 19: Linear Regression - Translation -> crit_perc</h4>

<div align="center">

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

</div>

<h4 align="center">Table 20: Linear Regression - Translation -> gen_impr</h4>

<div align="center">

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

</div>

<h4 align="center">Top 3 Best Performing Views (across all combinations, by R²)</h4>

<div align="center">

| Rank | View | Combination | R² | RMSE |
|------|------|-------------|-----|------|
| 1 | View 6 | Rotation -> crit_perc | 0.2608 | 24.48 |
| 2 | View 3 | Rotation -> crit_perc | 0.2468 | 14.46 |
| 3 | View 7 | Translation -> crit_perc | 0.2438 | 25.56 |

</div>

<h4 align="center">Figures: Top 3 Linear Regression Plots</h4>

<div align="center">

| | | |
|:-:|:-:|:-:|
| <img src="figures/q4iii_Rotation_crit_perc_view_6.png" alt="q4iii_Rotation_crit_perc_view_6" width="70%"> | <img src="figures/q4iii_Rotation_crit_perc_view_3.png" alt="q4iii_Rotation_crit_perc_view_3" width="70%"> | <img src="figures/q4iii_Translation_crit_perc_view_7.png" alt="q4iii_Translation_crit_perc_view_7" width="70%"> |

</div>

The linear regression results confirm the weak relationship between alignment metrics and quality scores. All R² values fall below 0.27, meaning even the best model (Rotation -> crit_perc at View 6, R² = 0.26) explains only about a quarter of the variance in quality scores. Most models explain less than 10% of the variance.

The top 3 performing models all involve crit_perc as the dependent variable rather than gen_impr, suggesting that alignment metrics have a slightly stronger (though still weak) association with the checklist-based score than with the holistic impression score. This could be because certain checklist criteria — such as correct probe rotation or visualisation of specific anatomical structures — are more directly influenced by probe positioning, whereas the general impression captures a broader, less position-dependent assessment.

Rotation shows marginally better predictive power than translation for crit_perc, appearing in 2 of the top 3 models. This makes physical sense: rotating the probe changes which anatomical structures fall within the imaging plane, directly affecting whether checklist criteria are met. Translation (shifting the probe position) has a less direct relationship with the image content.

Overall, the poor regression performance across all 40 models reinforces the conclusion from Q4 Part ii: simple rigid transformation parameters (rotation and translation) are insufficient to capture or predict image quality. The quality of a TOE image depends on a complex combination of factors — probe positioning, angulation, image plane selection, and real-time optimisation — that cannot be reduced to a single rotation angle and displacement value.

---

## Summary

<div align="center">

| Question | Key Finding |
|----------|-------------|
| Q1 | View 1 has the highest Pearson correlation (r = 0.9274) between the two quality scores; linear regression achieves R² = 0.86 for View 1 |
| Q2 | MI and CS are the best differentiating similarity metrics (each significant in 2/10 views); SSI significant in only 1/10 views |
| Q3 | MI -> gen_impr at View 1 achieves the best polynomial regression (R² = 0.55); Gaussian basis regression (with CV-based order selection) substantially improves fit (View 4 R² = 0.79) |
| Q4 | Neither rotation nor translation significantly differentiates experts from novices in any view; low R² values across all linear regressions |

</div>
