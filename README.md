# AGVI
This repository provides illustrative examples for application of the approximate Gaussian variance inference (AGVI) method. AGVI is an online Bayesian inference method that allows inferring the process error variance terms analytically through closed-form equations. The method provides unbiased and statistically consistent estimates for both the mean as well as the uncertainty associated with the variance terms.
## Examples
Two Case studies are provided in this repository. Case study $1$ focuses on the online estimation of error variance for a linear time-varying (LTV) model. The study includes statistical consistency tests to showcase the filter's optimality, t-tests to prove that the estimates are unbiased, empirical validation of uncertainty associated with error variance estimates, and analysis of the impact of $\frac{Q}{R}$ ratio on the posterior mean estimate $\mu_{\mathtt{T}|\mathtt{T}}$ of the error variance term. The AGVI method is also compared to two adaptive Kalman filtering (AKF) approaches: the sliding window variational adaptive Kalman filter} (SWVAKF)  and the measurement difference method (MDM). These AKF methods fall under different categories, with MDM being a correlation method and SWVAKF being a Bayesian method. Case study $2$ presents a simulated multivariate random walk model with a full process error covariance matrix $\mathbf{Q}$ and a comparison of the AGVI method to the AKF methods.
## Illustration
![Online estimation of the error variance term for three diffferent cases with different initialization. The true $\sigma^2_{W}$ value in each case is shown in red dashed line, while the estimated values and their $\pm1\sigma$ uncertainty bound are shown in black and green shaded area.](Figures/Figure1_case_study1.png)


  
