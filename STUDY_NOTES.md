# Groundwater Geostatistics and Inverse Modeling

**A teaching guide with mathematical derivations, questions, solutions, and a [pyRGA](https://pypi.org/project/pyrga-ht/) case study.**

> Audience: students with basic calculus, linear algebra, probability, and introductory groundwater hydrology.  
> Scope: variograms → Gaussian random fields → kriging → co-kriging → groundwater inversion → optimization.

## 0. The case study and roadmap

Consider an aquifer with heterogeneous positive hydraulic conductivity $K(\mathbf x)$. We may have sparse borehole measurements of $K$, and head observations from multiple pumping configurations. We want to (1) characterize spatial continuity, (2) estimate unsampled properties, and (3) infer a conductivity field consistent with observations and groundwater physics.

| Method | Question | Result |
|---|---|---|
| Variogram | How does dissimilarity change with separation? | Spatial dependence model |
| Gaussian field | What heterogeneous fields are plausible? | Statistical prior/realizations |
| Kriging | What is $\log K$ at unsampled points? | Best linear prediction and variance |
| Co-kriging | Can auxiliary observations improve predictions? | Multivariate spatial prediction |
| Inversion | Which conductivity fields explain observed heads? | Calibrated parameter field |
| Optimization | How do we efficiently solve the inverse problem? | Parameter estimates and diagnostics |

**Why use $Y=\log K$?** Conductivity must be positive, and the logarithm often gives a more manageable distribution. A Gaussian model for $Y$ produces positive, lognormal $K=\exp(Y)$. The lognormal assumption is a model choice, not a universal law.

## 1. Variograms and covariance

### 1.1 Random-field statistics

For a spatial process $Y(\mathbf x)$, define

$$m(\mathbf x)=E[Y(\mathbf x)],\qquad C(\mathbf x,\mathbf x')=\operatorname{Cov}(Y(\mathbf x),Y(\mathbf x')).$$

Second-order stationarity means a constant mean and covariance depending only on the separation vector $\mathbf h=\mathbf x'-\mathbf x$. Isotropy additionally means covariance depends only on $h=\|\mathbf h\|$.

### 1.2 Semivariogram

$$\gamma(\mathbf h)=\frac12 E[(Y(\mathbf x+\mathbf h)-Y(\mathbf x))^2].$$

Under second-order stationarity, expand the square:

$$2\gamma(\mathbf h)=\operatorname{Var}(Y_1)+\operatorname{Var}(Y_2)-2\operatorname{Cov}(Y_1,Y_2)=2C(0)-2C(\mathbf h),$$

so

$$\boxed{\gamma(\mathbf h)=C(0)-C(\mathbf h)}.$$

Estimate an isotropic semivariogram from sampled pairs in a lag bin $B(h)$:

$$\hat\gamma(h)=\frac{1}{2N(h)}\sum_{(i,j)\in B(h)}[Y(\mathbf x_i)-Y(\mathbf x_j)]^2.$$

The **nugget** is a discontinuity at zero distance (microscale variability or measurement error); **sill** is the large-distance semivariance of a stationary process; **range** measures spatial continuity. An exponential model approaches its sill asymptotically, so distinguish the scale parameter from the **practical range**.

### 1.3 Three models

For $h>0$, with nugget $c_0$, partial sill $c$:

- Exponential: $\gamma(h)=c_0+c(1-e^{-h/a})$; practical range for 95% of partial sill $\approx 3a$.
- Gaussian: $\gamma(h)=c_0+c(1-e^{-(h/a)^2})$; practical range $\approx\sqrt{-\log(0.05)}a\approx1.73a$.
- Spherical: $\gamma(h)=c_0+c[1.5(h/a)-0.5(h/a)^3]$ for $0<h\le a$, and $c_0+c$ beyond $a$.

Theoretical $\gamma(0)=0$ even with a nugget; $\lim_{h\downarrow0}\gamma(h)=c_0$.

Anisotropy may be expressed by $r=\sqrt{(\Delta x/a_x)^2+(\Delta y/a_y)^2}$ and $C(\Delta x,\Delta y)=\sigma^2e^{-r}$, with direction-specific correlation scales. Statistical anisotropy of the log-$K$ field is different from a physical anisotropic conductivity tensor.

### Worked problem 1

Given $C(h)=2e^{-h/10}$, calculate the semivariogram at $h=10$.

**Solution:** $\gamma(10)=2(1-e^{-1})\approx1.2642$.

### Questions with answers

1. **Why do nearby observations tend to be more similar?** When covariance decreases with separation, near points are more positively correlated; geological discontinuities can violate this simple expectation.
2. **Does a 100 m variogram range imply constant $K$ inside 100 m?** No. It describes decay of statistical dependence, not a homogeneous block.
3. **Why might an empirical variogram exceed the fitted sill?** Sampling noise, nonstationarity, a wrong model, directional mixing, or outliers.
4. **What changes when the field is anisotropic?** Dependence is a function of the *separation vector* rather than distance alone; compute directional variograms.

## 2. Gaussian random fields (GRFs)

For finite points $\mathbf x_1,\ldots,\mathbf x_n$, a GRF satisfies

$$\mathbf Y=[Y(\mathbf x_1),\ldots,Y(\mathbf x_n)]^T\sim\mathcal N(\boldsymbol\mu,\mathbf C).$$

A valid covariance matrix must be symmetric and positive semidefinite for every set of locations. A field having a *Gaussian-shaped covariance kernel* does **not** by itself imply a Gaussian probability distribution.

### 2.1 Sampling and conditioning

With $\mathbf C=\mathbf L\mathbf L^T$ and $\mathbf z\sim\mathcal N(0,\mathbf I)$,

$$\mathbf Y=\boldsymbol\mu+\mathbf L\mathbf z.$$

For unobserved $u$ and observed $o$ components of a joint Gaussian vector,

$$\boxed{E[\mathbf Y_u\mid\mathbf y_o]=\boldsymbol\mu_u+\mathbf C_{uo}\mathbf C_{oo}^{-1}(\mathbf y_o-\boldsymbol\mu_o)}$$

$$\boxed{\mathbf C_{u\mid o}=\mathbf C_{uu}-\mathbf C_{uo}\mathbf C_{oo}^{-1}\mathbf C_{ou}}.$$

If observations have independent additive error with covariance $\mathbf R$, replace $\mathbf C_{oo}$ by $\mathbf C_{oo}+\mathbf R$ when conditioning on noisy data.

### 2.2 Reduced-dimensional parameterization

A truncated Karhunen–Loève/principal-component-style representation is

$$Y(\mathbf x,\boldsymbol\xi)\approx m(\mathbf x)+\sum_{j=1}^{r}\sqrt{\lambda_j}\phi_j(\mathbf x)\xi_j,$$

where $\xi_j\sim\mathcal N(0,1)$ under the Gaussian prior. Instead of estimating thousands of cellwise values, estimate the retained coefficients. The precise scaling and basis construction in a particular software package must be checked rather than assumed identical to this expression.

### Worked problem 2

Let standardized Gaussian $Y_1,Y_2$ have correlation $\rho=0.8$. After observing $Y_1=2$, what are $E[Y_2\mid Y_1]$ and $\operatorname{Var}(Y_2\mid Y_1)$?

**Solution:** conditional mean $0.8(2)=1.6$; conditional variance $1-0.8^2=0.36$.

### Questions with answers

1. **Can two fields have the same covariance but different realizations?** Yes; covariance describes ensemble statistics, not a unique field.
2. **Does conditioning always make every field value exact?** Only at noise-free directly observed points under an exact conditioning model.
3. **Why use a reduced basis?** It reduces inverse problem dimension and can encode prior smoothness/correlation.
4. **Is the conditional mean a typical realization?** Not necessarily. The mean suppresses unresolved high-frequency variability.

## 3. Kriging

Write a linear estimate at $\mathbf x_0$ as $\hat Y_0=\sum_i w_iY_i$ (ordinary kriging), or $\hat Y_0=m+\sum_iw_i(Y_i-m)$ for known mean $m$ (simple kriging).

### 3.1 Simple kriging

$$\boxed{\hat Y_0=m+\mathbf c^T\mathbf C^{-1}(\mathbf y-m\mathbf 1)}$$

$$\boxed{\sigma_{SK}^2=C(0)-\mathbf c^T\mathbf C^{-1}\mathbf c},$$

with $\mathbf c_i=\operatorname{Cov}(Y(\mathbf x_0),Y(\mathbf x_i))$. Under a joint Gaussian model, these equal the conditional mean and variance.

### 3.2 Ordinary kriging

Unknown constant mean requires unbiasedness $\mathbf 1^T\mathbf w=1$:

$$\begin{bmatrix}\mathbf C&\mathbf1\\\mathbf1^T&0\end{bmatrix}\begin{bmatrix}\mathbf w\\\eta\end{bmatrix}=\begin{bmatrix}\mathbf c\\1\end{bmatrix}.$$

With this sign convention, $\sigma^2_{OK}=C(0)-\mathbf w^T\mathbf c-\eta$.

**Example:** At the midpoint of two symmetrically located wells with values $10$ and $14$, ordinary kriging weights are $0.5$ and $0.5$, producing $12$. The numerical prediction *variance* still requires a covariance model.

**Practical distinction:** Kriging from borehole conductivity data is *not* the same as assimilating hydraulic heads through a groundwater PDE. In our code, kriging uses synthetic **direct log-conductivity samples**; inversion uses **heads**.

### Questions with answers

1. **Can a kriging weight be negative?** Yes; weights are constrained covariance-optimal coefficients, not probabilities.
2. **What makes ordinary and simple kriging different?** Simple kriging treats the mean as known; ordinary kriging accounts for an unknown constant mean.
3. **What uncertainty does reported kriging variance omit?** Commonly, uncertainty in the fitted variogram parameters and structural misspecification.
4. **Why is a kriging map often too smooth?** It is a conditional mean rather than a conditional simulation.

## 4. Co-kriging and secondary data

Let $Y_1=\log K$ (primary) and $Y_2$ (e.g., geophysical measurement) be two jointly modeled fields. Use the covariance blocks

$$\mathbf C=\begin{bmatrix}\mathbf C_{11}&\mathbf C_{12}\\\mathbf C_{21}&\mathbf C_{22}\end{bmatrix}.$$

Known-mean simple co-kriging is joint Gaussian conditioning:

$$\hat Y_{1,0}=m_{1,0}+\mathbf C_{1,0,d}\mathbf C_{dd}^{-1}(\mathbf d-\mathbf m_d).$$

For ordinary co-kriging, apply appropriate mean/unbiasedness constraints. Cross-covariances cannot be fitted arbitrarily: the *whole* multivariate covariance must remain valid.

A valid **linear model of coregionalization (LMC)** is

$$C_{ij}(\mathbf h)=\sum_{k}B_{ij}^{(k)}\rho_k(\mathbf h),$$

where $\rho_k$ are valid spatial correlations and each $\mathbf B^{(k)}$ is positive semidefinite.

**Example:** Borehole $\log K$ is sparse but geophysical resistivity is dense. Co-kriging may improve inference if joint correlation is supported. Be careful: resistivity is also influenced by pore-water chemistry, lithology, and saturation.

### Questions with answers

1. **When does a secondary variable help?** When it provides nonredundant, reliable information about the primary variable under a valid joint model.
2. **Why can't we fit arbitrary cross-variograms separately?** The assembled joint covariance may cease to be positive semidefinite.
3. **How does regression kriging differ?** It uses predictors for the mean trend and kriges residuals, whereas co-kriging models cross-spatial dependence explicitly.
4. **Does `pyrga-ht` provide a co-kriging demonstration in this repository?** No; this repository presents co-kriging theoretically, while the runnable lab implements a univariate kriging step and an RGA head inversion.

## 5. Groundwater forward and inverse modeling

### 5.1 Governing equation

For saturated transient flow with scalar $K$ (or tensor $\mathbf K$), a common continuum formulation is

$$S_s\frac{\partial h}{\partial t}=\nabla\cdot(\mathbf K\nabla h)+q,$$

where $S_s$ is specific storage and $q$ is a consistent volumetric source density. For steady conditions, $\nabla\cdot(\mathbf K\nabla h)+q=0$. Boundary conditions, geometry, units, and source normalization matter.

Let $\mathbf m=\log\mathbf K$ and let $\mathcal G(\mathbf m)$ map parameters to measured heads (including numerical flow solution and observation extraction). Then

$$\mathbf d=\mathcal G(\mathbf m)+\boldsymbol\epsilon.$$

### 5.2 Misfit and regularization

A common inverse objective is

$$J(\mathbf m)=\frac12\|\mathcal G(\mathbf m)-\mathbf d\|_{\mathbf R^{-1}}^2+\frac12\|\mathbf m-\mathbf m_0\|_{\mathbf C_m^{-1}}^2.$$

Here $\mathbf R$ is measurement-error covariance and $\mathbf C_m$ is prior covariance. Under additive Gaussian errors and Gaussian prior, this is the negative log-posterior up to constants; minimization yields a **maximum a posteriori (MAP)** estimate.

Sparse head observations rarely identify thousands of independent conductivities. Spatial priors, parameter reduction, pumping experiments, and additional flux/tracer data can reduce—but rarely eliminate—ambiguity.

### 5.3 Linear-Gaussian result

For $\mathbf d=\mathbf H\mathbf m+\boldsymbol\epsilon$ with independent Gaussian prior and errors,

$$\mathbf C_{post}=(\mathbf C_m^{-1}+\mathbf H^T\mathbf R^{-1}\mathbf H)^{-1},$$

$$\mathbf m_{post}=\mathbf m_0+\mathbf C_m\mathbf H^T(\mathbf H\mathbf C_m\mathbf H^T+\mathbf R)^{-1}(\mathbf d-\mathbf H\mathbf m_0).$$

This is connected to Gaussian conditioning, kriging, and Kalman updates. A nonlinear groundwater posterior need not be Gaussian, and its MAP need not equal its mean.

### 5.4 What makes hydraulic tomography informative?

Changing which well pumps changes the head sensitivity patterns. Multiple source/observation configurations probe different aspects of subsurface structure. Nonetheless, parameter null spaces remain possible and model-error effects should be assessed.

### Questions with answers

1. **Why can excellent head fit coexist with poor conductivity recovery?** Nonuniqueness, noise, model error, sparse coverage, and regularization bias.
2. **Why use log-$K$ parameters?** Positivity and often improved statistical scaling.
3. **How is kriging distinct from inversion?** Kriging uses covariance and direct spatial measurements; inversion incorporates a PDE connecting $K$ to heads.
4. **What is the role of the prior?** It constrains plausible magnitudes, correlations, and structures that head data cannot uniquely determine.

## 6. Optimization

For residual $\mathbf r(\mathbf m)=\mathcal G(\mathbf m)-\mathbf d$ and sensitivity Jacobian $\mathbf J_G=\partial\mathcal G/\partial\mathbf m$, Gauss–Newton solves an approximate second-order system. For the MAP objective above,

$$\left(\mathbf J_G^T\mathbf R^{-1}\mathbf J_G+\mathbf C_m^{-1}\right)\Delta\mathbf m=-\mathbf J_G^T\mathbf R^{-1}\mathbf r-\mathbf C_m^{-1}(\mathbf m-\mathbf m_0).$$

The update $\mathbf m\leftarrow\mathbf m+\Delta\mathbf m$ may require line search/trust region. Levenberg–Marquardt adds damping; BFGS approximates curvature; adjoints efficiently compute derivatives with respect to many parameters.

For discretized $\mathbf A(\mathbf m)\mathbf h=\mathbf b(\mathbf m)$, define the adjoint by $\mathbf A^T\boldsymbol\lambda=(\partial J/\partial\mathbf h)^T$. Then

$$\nabla_{\mathbf m}J=\left(\frac{\partial J}{\partial\mathbf m}\right)_{\mathbf h}+\left[\frac{\partial\mathbf b}{\partial\mathbf m}-\frac{\partial(\mathbf A\mathbf h)}{\partial\mathbf m}\right]^T\boldsymbol\lambda.$$

A scalar objective can require one adjoint solve rather than a forward solve per unknown parameter, subject to implementation details.

### Questions with answers

1. **Why is Gauss–Newton attractive for head-data fitting?** It exploits residual sensitivities and least-squares structure.
2. **What does the Jacobian mean physically?** Each element describes the sensitivity of one predicted observation to one parameter.
3. **Does optimization guarantee a physically correct field?** No. It optimizes the chosen objective; identifiability and model adequacy must also be checked.
4. **Why use regularization?** To stabilize ill-posed estimation and encode prior plausibility.

## 7. pyRGA hands-on lab

Run [the accompanying Python demo](examples/pyrga_groundwater_demo.py) after following [README.md](README.md). It follows the PyPI `pyrga-ht` v0.1.4 public example interface:

1. `generate_synthetic_field`: create a positive heterogeneous $K$ and reduced spatial basis.
2. Empirical isotropic semivariogram from sampled pairs of $\log K$.
3. Simple kriging on a subset of **synthetic direct conductivity samples** using a **separately chosen illustrative Gaussian covariance** (not claimed to be pyRGA's exact generator covariance).
4. `prepare_physical_domain` and `hydraulic_tomography`: simulate pumping-well flow responses.
5. `observation_operator` and `add_noise`: construct noisy head observations.
6. `forward_model` and `gauss_newton_dynamic_lambda`: estimate reduced coefficients through the package's RGA optimization.
7. Diagnostic figures and result archive for comparison.

**Important limits:** This is a synthetic tutorial, not calibration of real site data. The exact flattening and units should be checked against pyRGA implementation. The demo evaluates **observation-space fit**; do not claim accurate recovered spatial $K$ solely from head RMSE. Co-kriging is discussed in Section 4 but is not implemented in the demo.

### Experiments

- Change the field's `lx` and `ly` and compare structure and sample variograms.
- Increase/decrease synthetic borehole count and plot kriging standard deviation.
- Compare kriging error against the synthetic field in log space; repeat across seeds.
- Change the observation noise level and examine inversion error and stability.
- Change the number of retained modes and consider representation error versus conditioning.
- Contrast initial and fitted head predictions; analyze where model predictions are biased.

### Lab questions with model answers

**L1. Does a low kriging prediction variance mean the covariance model is correct?** No. It reports uncertainty conditional on an assumed covariance and data geometry.

**L2. Why is the sample variogram noisy?** One realization, limited sampled pairs, lag binning, and statistical variability.

**L3. Does optimizing reduced coefficients use direct $K$ samples?** Not in the pyRGA inversion step used here; it uses hydraulic-head observations, while the kriging exercise uses separately sampled log-$K$ values.

**L4. Why might inversion heads fit better without reconstructing the true field?** Multiple fields can generate similar response patterns; reducing basis dimension adds approximation bias.

**L5. Does decreasing the regularization coefficient always improve the result?** It may improve in-sample fit but harm parameter plausibility or out-of-sample prediction.

## 8. Ten-question final assessment (with solutions)

1. **Derive $\gamma(h)=C(0)-C(h)$.** Expand the squared two-point difference and use second-order stationarity.
2. **For $C(h)=2e^{-h/10}$, find $\gamma(10)$.** $2(1-e^{-1})\approx1.2642$.
3. **What makes a covariance valid?** Every finite covariance matrix must be symmetric positive semidefinite.
4. **Why is the kriging mean smoother than a field realization?** It integrates out residual conditional uncertainty.
5. **What changes under observation noise?** Gaussian conditioning uses $C_{oo}+R$ for independent additive noise.
6. **What constraint matters for cross-covariance?** The full joint covariance must remain valid/positive semidefinite.
7. **Define the inverse problem.** Estimate $K$ or reduced parameters that explain observed heads through the forward groundwater model.
8. **When does quadratic regularization equal Gaussian-prior MAP?** With Gaussian errors and a Gaussian prior (properly weighted).
9. **How can head fit be perfect but $K$ wrong?** Nonidentifiability and limited observations.
10. **When are adjoints useful?** Many parameters and few scalar objectives, when derivatives of a PDE-based objective are required.

## 9. Suggested four-week teaching plan

| Week | Focus | Assignment |
|---|---|---|
| 1 | Variogram, stationarity, anisotropy, GRFs | Generate fields and calculate directional/omnidirectional variograms |
| 2 | Conditional Gaussian, kriging, co-kriging | Run kriging and interpret standard deviation; derive simple co-kriging equations |
| 3 | Flow PDE, observation operator, inversions | Run pyRGA tomography and explain observation sensitivities |
| 4 | RGA, Gauss–Newton, regularization | Complete inversion, analyze fit and identifiability, present limitations |

## References and attribution

- [pyRGA package / published v0.1.4 example](https://pypi.org/project/pyrga-ht/). API function calls are adapted from its public quick-start/optimization example; this repository's explanatory code and exercises are separately authored.
- Zhao, Y., & Luo, J. (2020), *Reformulation of Bayesian Geostatistical Approach on Principal Components*, **Water Resources Research**, 56(4), e2019WR026732, https://doi.org/10.1029/2019WR026732 .
- Cressie, N., *Statistics for Spatial Data*.
- Chilès, J.-P., & Delfiner, P., *Geostatistics: Modeling Spatial Uncertainty*.
- Anderson, M. P., Woessner, W. W., & Hunt, R. J., *Applied Groundwater Modeling*.

