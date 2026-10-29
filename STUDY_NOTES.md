# Gaussian Random Fields and Gauss–Newton Inverse Modeling for Groundwater

*A focused study guide with derivations, worked examples, questions and answers, and a reproducible `pyrga-ht` laboratory.*

**Audience:** graduate students familiar with linear algebra, probability, optimization, and basic groundwater flow.  
**Main emphasis:** Gaussian random field (GRF) construction → physical inverse problem → **Gauss–Newton (GN)** → accuracy and scalability experiments.  
**Software:** [pyRGA / `pyrga-ht`](https://pypi.org/project/pyrga-ht/) ([package example](https://pypi.org/project/pyrga-ht/), [related paper](https://doi.org/10.1029/2019WR026732)).

## Learning outcomes

By the end, you should be able to (1) specify and validate a Gaussian spatial prior; (2) generate unconditional and conditional conductivity fields; (3) derive a regularized inverse problem and the GN linear system; (4) explain how eigenmode truncation changes computational cost and bias; and (5) **run a controlled pyRGA experiment and argue whether the optimizer is accurate, robust, and scalable**.

## 0. Foundations: spatial statistics and geostatistics

### 0.1 Random variables, spatial data, and notation

We observe a property at coordinates $\mathbf x=(x,y)$ or $\mathbf x=(x,y,z)$. For a groundwater example, let $K(\mathbf x)>0$ denote hydraulic conductivity and let

$$Y(\mathbf x)=\ln K(\mathbf x).$$

At each location, $Y(\mathbf x)$ is a **random variable**; across the domain, the collection $\{Y(\mathbf x)\}$ is a **random field**. A realization is one full spatial map sampled from the field's probability distribution. A measurement is one observed value at a known spatial location.

In two-dimensional grids, distinguish (i) **physical coordinates** such as meters; (ii) **grid indices** $(i,j)$; and (iii) **flattened vector indices**. A correlation length of 20 meters is *not* automatically 20 grid cells. Coordinate units and ordering must be consistent in every covariance calculation.

### 0.2 Mean, variance, covariance, and correlation

Define the mean function, variance, and cross-location covariance:

$$m(\mathbf x)=\mathbb E[Y(\mathbf x)],\quad \sigma^2(\mathbf x)=\operatorname{Var}[Y(\mathbf x)],$$

$$C(\mathbf x_1,\mathbf x_2)=\mathbb E[(Y(\mathbf x_1)-m(\mathbf x_1))(Y(\mathbf x_2)-m(\mathbf x_2))].$$

The **correlation coefficient** is dimensionless:

$$\rho(\mathbf x_1,\mathbf x_2)=\frac{C(\mathbf x_1,\mathbf x_2)}{\sigma(\mathbf x_1)\sigma(\mathbf x_2)}.$$

Covariance is measured in squared units of the random variable; correlation lies in $[-1,1]$. Under second-order stationarity, $m(\mathbf x)=\mu$, $\sigma^2(\mathbf x)=\sigma^2$, and $C(\mathbf x_1,\mathbf x_2)=C(\mathbf h)$ for separation vector $\mathbf h=\mathbf x_2-\mathbf x_1$. Under isotropy it depends only on $\|\mathbf h\|$. **Stationarity does not imply isotropy.**

### 0.3 Semivariograms and structural interpretation

The semivariogram measures expected squared differences:

$$\gamma(\mathbf h)=\tfrac12\mathbb E[(Y(\mathbf x+\mathbf h)-Y(\mathbf x))^2].$$

For a second-order stationary field,

$$\boxed{\gamma(\mathbf h)=C(\mathbf 0)-C(\mathbf h).}$$

If $C(\mathbf 0)=\sigma^2$, a high correlation means small semivariance. The **sill** is an asymptotic semivariance level; the **nugget** is the limit of semivariance as nonzero lag approaches zero, which can represent unresolved microscale variation or an error contribution. A **range** characterizes decay of correlation; exponential and Gaussian kernels approach zero asymptotically, so their quoted “practical range” depends on a threshold convention. A directional variogram helps reveal anisotropy.

For $N(h)$ pairs in a distance bin around $h$, the empirical estimator is

$$\hat\gamma(h)=\frac{1}{2|N(h)|}\sum_{(i,j)\in N(h)}(y_i-y_j)^2.$$

This estimator does **not** prove Gaussianity and can be noisy with few pairs. Inspect directional bins, sampling bias, outliers, and measurement support before selecting a valid theoretical covariance model.

### 0.4 Kriging and co-kriging in context

**Simple kriging** assumes a known mean and covariance and predicts missing values by conditioning a joint Gaussian model:

$$\hat{\mathbf y}_u=\boldsymbol\mu_u+C_{uo}C_{oo}^{-1}(\mathbf y_o-\boldsymbol\mu_o),$$

$$C_{u\mid o}=C_{uu}-C_{uo}C_{oo}^{-1}C_{ou}.$$

For noisy measurements with independent error covariance $R$, replace $C_{oo}$ by $C_{oo}+R$. **Ordinary kriging** treats an unknown constant mean and imposes an unbiasedness constraint (weights sum to one). **Co-kriging** models jointly correlated variables using auto- and cross-covariances; its joint covariance must be valid (positive semidefinite). These tools use statistics of directly or indirectly measured properties; the later groundwater inverse model also couples conductivity to hydraulic heads through the flow equations.

### Foundation questions with answers

1. **What is the difference between an expected field and a realization?** The expectation $m(\mathbf x)$ is the across-realization mean; one realization includes a spatially correlated random departure from that mean.
2. **If two locations are 1 meter apart, must their values be almost equal?** No: similarity depends on correlation length, kernel, and nugget, not on distance alone.
3. **Does a zero variogram at zero lag mean there is no measurement noise?** No: $\gamma(0)=0$ by definition for the same random variable; a noise nugget concerns *distinct measurements* at very short separation.
4. **Why not independently draw Gaussian values at each grid cell?** That generates a white-noise field rather than the desired spatially correlated field.
5. **What is the practical difference between kriging and inverse modeling?** Kriging uses spatial dependence of measured variables; inversion conditions on data related to the unknown field by a physical simulator.

---

# Part I. Gaussian random fields

## 1. Random-field statistics

A field $Y(\mathbf x)$ is Gaussian when *every finite collection* $\mathbf Y=[Y(\mathbf x_1),\ldots,Y(\mathbf x_n)]^T$ is multivariate Gaussian:

$$\mathbf Y\sim\mathcal N(\boldsymbol\mu,C),\qquad C_{ij}=\operatorname{Cov}(Y(\mathbf x_i),Y(\mathbf x_j)).$$

The mean $m(\mathbf x)$ controls the expected spatial trend; variance $\sigma^2(\mathbf x)=C(\mathbf x,\mathbf x)$ controls pointwise dispersion; covariance controls spatial dependence. A stationary field has a constant mean and $C(\mathbf x,\mathbf x')=C(\mathbf x'-\mathbf x)$. Isotropy further reduces dependence to $\|\mathbf x'-\mathbf x\|$.

**Important distinction:** a *Gaussian-shaped covariance* does not guarantee a *Gaussian random field*. Gaussianity refers to joint distributions, not just the kernel.

For groundwater, choose $Y=\log K$ and set $K=\exp Y>0$. If $Y\sim\mathcal N(\mu,\sigma^2)$ at a location, then

$$\mathbb E[K]=e^{\mu+\sigma^2/2},\quad \operatorname{median}(K)=e^\mu,\quad \operatorname{Var}(K)=(e^{\sigma^2}-1)e^{2\mu+\sigma^2}.$$

Thus **$e^\mu$ is not the arithmetic mean of $K$**. State whether your field uses $\ln K$ or $\log_{10}K$ and maintain consistent conductivity units.

### Questions and answers

1. **What makes a covariance model admissible?** For every selection of spatial locations, the resulting matrix must be symmetric positive semidefinite: $\mathbf a^TC\mathbf a\ge0$ for any $\mathbf a$.
2. **Why work with log conductivity?** Positivity and frequently more tractable marginal statistics, while recognizing that lognormality is an assumption.
3. **Does variance determine correlation length?** No. Variance is local amplitude; correlation length describes separation dependence.
4. **What happens as $\sigma^2$ increases with other parameters fixed?** Log-conductivity becomes more variable, and the resulting $K$ distribution becomes more right-skewed.

## 2. Covariance functions and structural parameters

A Gaussian random field's finite-dimensional distribution is completely specified by its **mean vector** and **covariance matrix**. A kernel tells us how to compute covariance for **any pair of coordinates**. A useful general representation is

$$\boxed{C(\mathbf x_i,\mathbf x_j)=\sigma^2\rho(d_A(\mathbf x_i,\mathbf x_j))}$$

where $\rho(0)=1$ and $d_A$ is an anisotropy-adjusted separation. For a 2D axis-aligned Gaussian kernel, define

$$d_A^2=\left(\frac{x_i-x_j}{a_x}\right)^2+\left(\frac{y_i-y_j}{a_y}\right)^2,$$

$$\boxed{C(\mathbf x_i,\mathbf x_j)=\sigma^2\exp(-d_A^2).}$$

**Note:** the mean $\mu$ does *not* appear in the covariance formula. It shifts the field's expected value, whereas $\sigma^2$ and the kernel shape determine variability around the mean. For this kernel, $C(\mathbf x_i,\mathbf x_i)=\sigma^2$, and for any two different locations $C$ decreases as the standardized separation grows.

### 2.1 Common covariance families

| Model | $\rho$ (isotropic, distance $h$) | Interpretation |
|---|---|---|
| Exponential | $\exp(-h/a)$ | Rougher spatial realizations; 95% practical correlation range $\approx3a$ |
| Squared exponential / Gaussian | $\exp[-(h/a)^2]$ | Very smooth realizations; 95% correlation range $\approx1.73a$ |
| Matérn | $\frac{2^{1-\nu}}{\Gamma(\nu)}(\sqrt{2\nu}h/\ell)^\nu K_\nu(\sqrt{2\nu}h/\ell)$ | Smoothness tuned by $\nu$ |

Here $a$ and $\ell$ are scale parameters under the written conventions, not universal definitions of “range.” In anisotropic models use an anisotropy-adjusted distance or transform the coordinate differences. A Gaussian *covariance kernel* can be used in a Gaussian random field, but kernel shape alone does not imply Gaussian joint distributions.

### 2.2 Structural parameters: which values must be specified?

| Parameter | Meaning | If increased, holding other parameters fixed |
|---|---|---|
| Mean $\mu$ | Expected **log conductivity**, $E[Y]$, if working with $Y=\ln K$ | Shifts every expected $Y$ upward; raises typical $K$ |
| Variance $\sigma^2$ | Pointwise variance of $Y$ (units of log-conductivity squared) | Larger amplitude of spatial fluctuations |
| $a_x$ | Correlation-length scale along $x$ in coordinate units | More lateral continuity along $x$ |
| $a_y$ | Correlation-length scale along $y$ | More continuity along $y$ |
| Orientation $\theta$ | Rotation of principal correlation axes | Rotates elongated spatial structures |
| Kernel / Matérn $\nu$ | Small-lag shape and smoothness | Changes roughness and differentiability |
| Nugget / error variance | Microscale variability and/or observation error | More short-separation disagreement; effects depend on exact model |

**Physical interpretation example.** Choose $Y=\ln K$ where $K$ is measured in m/s, and let $\mu=\ln(10^{-4})\approx-9.2103$, $\sigma^2=1$, $a_x=40$ m, $a_y=10$ m. The **median** conductivity at each location is $10^{-4}$ m/s, while the **mean** conductivity is $e^{\mu+\sigma^2/2}\approx1.6487\times10^{-4}$ m/s. The larger $x$ correlation length means elongated structures in the $x$ direction. All units and transforms must be stated explicitly.

### 2.3 Pseudocode: evaluate covariance for two coordinates

Use the following pseudocode for the **2D anisotropic Gaussian kernel**. It is the conceptual operation repeated to assemble a covariance matrix.

```text
FUNCTION GaussianCovariance(point_i, point_j, variance, ax, ay):
    REQUIRE variance >= 0, ax > 0, ay > 0
    dx = point_i.x - point_j.x
    dy = point_i.y - point_j.y
    squared_scaled_distance = (dx / ax)^2 + (dy / ay)^2
    correlation = exp(-squared_scaled_distance)
    covariance = variance * correlation
    RETURN covariance
```

**Worked pair.** With $\mathbf x_1=(10,20)$ m, $\mathbf x_2=(30,30)$ m, $a_x=40$ m, $a_y=10$ m and $\sigma^2=1.5$,

$$d_A^2=(20/40)^2+(10/10)^2=1.25,$$

$$\rho=e^{-1.25}\approx0.2865,\qquad C(\mathbf x_1,\mathbf x_2)=1.5e^{-1.25}\approx0.4298.$$

At a single location, the covariance is $C(\mathbf x_i,\mathbf x_i)=1.5$. The mean $\mu$ never enters these two-coordinate calculations.

If the principal axes are rotated through angle $\theta$, first compute

$$\begin{bmatrix}u\\v\end{bmatrix}=\begin{bmatrix}\cos\theta&\sin\theta\\-\sin\theta&\cos\theta\end{bmatrix}\begin{bmatrix}\Delta x\\\Delta y\end{bmatrix}$$

and then use $d_A^2=(u/a_x)^2+(v/a_y)^2$. This convention rotates the displacement into the principal-axis coordinate frame; define the angle convention explicitly in code.

### 2.4 Pseudocode: build a covariance matrix for many locations

For $n$ grid coordinates, repeat the pair operation for every $(i,j)$:

```text
FUNCTION BuildCovariance(points[0..n-1], variance, ax, ay):
    C = zeros(n, n)
    FOR i = 0 TO n-1:
        FOR j = i TO n-1:
            value = GaussianCovariance(points[i], points[j], variance, ax, ay)
            C[i,j] = value
            C[j,i] = value     # symmetry
    RETURN C
```

For the two-point example, the covariance matrix is

$$C=\begin{bmatrix}1.5&0.4298\\0.4298&1.5\end{bmatrix}.$$

It must be symmetric **positive semidefinite**; a strictly positive-definite kernel on distinct points can be Cholesky factored, subject to numerical precision. For identical coordinates or degenerate setups the matrix may be singular. A small diagonal jitter can improve numerical factorization but represents a changed discrete covariance, not a free mathematical operation.

### 2.5 From covariance to a sampled field: algorithm

```text
INPUT: grid coordinates, mean mu, variance, ax, ay, RNG seed
C = BuildCovariance(grid coordinates, variance, ax, ay)
L = Cholesky(C)                     # approximately; stabilize if needed
z = independent_standard_normals(n)
y = mu * ones(n) + L @ z           # correlated log-conductivity
K = exp(y)                         # if y = natural log conductivity
OUTPUT: y, K
```

**Why does it work?** $E[z]=0$, $\operatorname{Cov}(z)=I$, so $E[y]=\mu\mathbf 1$ and $\operatorname{Cov}(y)=LIL^T=C$. Repeating with different seeds creates different spatial realizations with the *same statistical model*. On large grids, dense $C$ costs $O(n^2)$ memory and Cholesky costs $O(n^3)$ operations; this motivates the low-rank and spectral methods discussed next.

### 2.6 Minimal executable Python example

A ready-to-run, pyRGA-independent fundamentals script is included in [`examples/covariance_grf_fundamentals.py`](examples/covariance_grf_fundamentals.py). The core operation is:

```python
import numpy as np

def gaussian_covariance(p, q, variance, ax, ay):
    dx, dy = np.asarray(p, dtype=float) - np.asarray(q, dtype=float)
    return variance * np.exp(-(dx/ax)**2 - (dy/ay)**2)

print(gaussian_covariance((10, 20), (30, 30), 1.5, 40, 10))
# approximately 0.4298
```

The complete script also constructs a covariance matrix, checks its eigenvalues, generates reproducible GRF samples, and visualizes the impact of correlation length and variance. It runs *without* pyRGA, making it a good first exercise before exploring `generate_synthetic_field()`.

### Worked questions and answers

1. **If the mean doubles, do covariance and correlation change?** Not when only a constant mean shift is changed. The random deviations and covariance remain the same.
2. **If variance changes from 1.5 to 3.0, what happens to covariance for the worked coordinate pair?** It doubles from approximately 0.4298 to 0.8595; correlation stays approximately 0.2865.
3. **If $a_y$ grows from 10 to 40 m, what happens to covariance for $(10,20)$ and $(30,30)$?** $d_A^2=0.25+0.0625=0.3125$ and covariance grows to $1.5e^{-0.3125}\approx1.0974$.
4. **Why is a covariance matrix not a matrix of pairwise distances?** Distances are inputs to a kernel; covariance entries quantify expected co-fluctuation and must satisfy positive-semidefinite constraints.
5. **Does a high covariance imply high correlation?** Not necessarily across models with different variances; normalize by standard deviations to compare correlations.
6. **What is the difference between statistical anisotropy and hydraulic tensor anisotropy?** One specifies spatial covariance geometry of a field; the other specifies directional flow response under a hydraulic gradient.

## 3. How to generate a Gaussian random field

### 3.1 Dense Cholesky simulation

Discretize at $n$ cells, construct $C_{ij}=C(\mathbf x_i,\mathbf x_j)$ and factor $C=LL^T$. Sample $\boldsymbol\xi\sim N(0,I)$ and set

$$\boxed{\mathbf y=\boldsymbol\mu+L\boldsymbol\xi},\qquad \operatorname{Cov}(\mathbf y)=LL^T=C.$$

Cholesky storage is $O(n^2)$ and factorization is $O(n^3)$ for a dense matrix: usually unsuitable for very large 2D/3D grids. A small diagonal jitter may stabilize numerical factorization but changes the model slightly.

### 3.2 Eigen/Karhunen–Loève or principal-component expansion

For $C=U\Lambda U^T$, truncate to the first $r$ modes:

$$\boxed{\mathbf y\approx\boldsymbol\mu+U_r\Lambda_r^{1/2}\boldsymbol\xi_r},\quad\boldsymbol\xi_r\sim N(0,I_r),\quad r\ll n.$$

The covariance is approximated by $U_r\Lambda_rU_r^T$; the fraction of total discrete variance captured is

$$f_r=\frac{\sum_{i=1}^r\lambda_i}{\sum_{i=1}^{n}\lambda_i}.$$

This creates a tradeoff: a small $r$ accelerates inversion and stabilizes the problem but may exclude true fine-scale structure. In pyRGA the published example generates an ensemble and returns basis-related arrays `V`, `alpha`, and `beta`; **verify their numerical normalization before identifying them with this exact textbook eigenbasis**.

### 3.3 Alternatives

Sequential Gaussian simulation, FFT/circulant embedding, spectral methods, and sparse-precision (GMRF/SPDE) methods can be preferable for large domains or different covariance assumptions. Each approach has assumptions about stationarity, domain boundaries, and numerical approximation.

### 3.4 Conditional simulation

Given direct observations, first compute conditional mean $m_c$ and conditional covariance $C_c$, then sample

$$\mathbf y^{(s)}=m_c+L_c\boldsymbol\xi^{(s)},\qquad L_cL_c^T=C_c.$$

This produces multiple fields consistent with the specified data-error model, unlike a single kriging mean.

### Questions and answers

1. **Why can two GRF realizations with the same $\mu,C$ look different?** Covariance specifies a distribution, not a unique field.
2. **Why can a low-rank basis speed inverse modeling?** It reduces independent coefficients and often makes the Jacobian much narrower.
3. **Does a large explained-variance fraction guarantee good inverse reconstruction?** No; physically observable modes may not coincide with the highest-variance prior modes.
4. **What changes when correlation length increases?** Coherent structures become larger; typically fewer leading modes capture a fixed fraction of variance on the same domain.
5. **How can a wrong covariance prior affect inversion?** It can bias the solution toward unrealistic structures and understate uncertainty even when data fit is good.

---

# Part II. Groundwater inverse problems

## 4. Forward model, observations and unknowns

For saturated flow with suitable units and source convention,

$$S_s\,\frac{\partial h}{\partial t}=\nabla\cdot(K(\mathbf x)\nabla h)+q(\mathbf x,t).$$

Under steady-state conditions $\nabla\cdot(K\nabla h)+q=0$. For tensor conductivity, replace scalar $K$ by $\mathbf K$. A discretized forward solver is

$$A(\mathbf m)\mathbf h=\mathbf b(\mathbf m),\qquad\mathbf d_{\rm pred}=H\mathbf h=\mathcal G(\mathbf m),$$

where $\mathbf m$ parameterizes $\log K$, and $H$ extracts well-head (or other) observations. **Forward problem:** given $K$ and boundary/pumping conditions, solve for $h$. **Inverse problem:** estimate $K$ from the measured hydraulic response.

Hydraulic tomography combines multiple pumping configurations, providing more informative responses than a single pumping test, although identifiability is never automatic.

## 5. Formulating the objective function

Let $\mathbf d=\mathcal G(\mathbf m_*)+\boldsymbol\epsilon$, with observation-error covariance $R$. A standard regularized weighted least-squares objective is

$$\boxed{\Phi(\mathbf m)=\tfrac12[\mathcal G(\mathbf m)-\mathbf d]^TR^{-1}[\mathcal G(\mathbf m)-\mathbf d]+\tfrac\lambda2\|L(\mathbf m-\mathbf m_0)\|_2^2}.$$

- **Data residual:** difference between predicted and observed heads.
- **Weights $R^{-1}$:** account for unequal and correlated measurement errors.
- **Reference $\mathbf m_0$:** the center of a prior/regularization model.
- **Regularizer $L$ and strength $\lambda$:** constrain poorly identified directions.

With a Gaussian prior $m\sim N(m_0,C_m)$ and Gaussian errors, the maximum-a-posteriori (MAP) objective uses $L^TL=C_m^{-1}$ and $\lambda=1$ (or an equivalent rescaling). **Do not assume a package parameter named `lambda` necessarily implements this precise prior penalty:** it may be an algorithmic damping parameter. Inspect the implementation.

### Identifiability, bias and uncertainty

An inverse problem can have many nearly equivalent solutions because different $K$ fields yield similar heads. A smaller training residual does not imply a more accurate conductivity field. Assess **observation fit**, **parameter/field recovery**, and **prediction at withheld pumping configurations** separately.

**Worked conceptual example:** With prescribed hydraulic heads at both ends of a 1D steady aquifer and no known flux, scaling all conductivities by the same factor leaves the head profile unchanged. Absolute conductivity scale cannot be inferred from those head observations alone.

### Questions and answers

1. **Why log-parameterize $K$?** It guarantees positivity and often improves statistical modeling.
2. **Why do we regularize?** Sparse heads may leave many combinations of field parameters unconstrained.
3. **What is a posterior?** The distribution of possible parameters given the observations and a probabilistic prior/noise model.
4. **Why use a separate validation experiment?** A solver can overfit observed pumping tests without recovering transferable physical structure.

---

# Part III. Gauss–Newton optimization (main focus)

## 6. From nonlinear least squares to the GN step

Define residual $r(m)=\mathcal G(m)-d$ and sensitivity/Jacobian $J(m)=\partial\mathcal G/\partial m\in\mathbb R^{n_d\times n_p}$. Near the current iterate $m_k$,

$$r(m_k+\delta)\approx r_k+J_k\delta.$$

Minimize the local quadratic approximation

$$q_k(\delta)=\tfrac12\|R^{-1/2}(r_k+J_k\delta)\|_2^2+\tfrac\lambda2\|L(m_k+\delta-m_0)\|_2^2.$$

Set its gradient to zero to obtain the **regularized Gauss–Newton normal equations**:

$$\boxed{(J_k^TR^{-1}J_k+\lambda L^TL)\delta_k=-J_k^TR^{-1}r_k-\lambda L^TL(m_k-m_0).}$$

Then update $m_{k+1}=m_k+\alpha_k\delta_k$, with step length $\alpha_k$ possibly chosen by line search.

### Derivation using gradients and Hessians

For the data term $\Phi_d(m)=\tfrac12r^TR^{-1}r$,

$$\nabla\Phi_d=J^TR^{-1}r.$$

Its exact Hessian is

$$\nabla^2\Phi_d=J^TR^{-1}J+\sum_{i} (R^{-1}r)_i\nabla^2\mathcal G_i$$

(for constant symmetric $R$). GN **drops the residual-weighted second-derivative term**, obtaining a positive-semidefinite curvature approximation. This is most reliable near a good fit or when residual nonlinearities are modest.

**Numerical practice:** do not explicitly invert $J^TJ$. Prefer QR/SVD, a least-squares solver, or Krylov methods on the regularized system. Scaling and conditioning matter.

## 7. Gauss–Newton versus related methods

| Algorithm | Approximate step matrix | Main tradeoff |
|---|---|---|
| Gradient descent | $I$ times a step-size heuristic | Simple; may be slow when ill-conditioned |
| Gauss–Newton | $J^TR^{-1}J$ (plus prior) | Uses residual sensitivity; can fail far from solution |
| Levenberg–Marquardt (LM) | $J^TR^{-1}J+\eta D$ | Damping/trust-region improves robustness |
| Full Newton | Exact Hessian | More curvature information; costly / possibly indefinite |

**GN regularization versus LM damping:** $\lambda L^TL$ in a statistical inverse objective is part of the model; LM's $\eta D$ is a numerical device to stabilize the *step*. Although algebraically similar, they should not be conflated. Some RGA implementations may blend these roles; read the package source for its exact meaning.

### Conditioning and rank

If $J$ has nearly dependent columns, $J^TJ$ has large condition number and unstable steps. Singular-value decomposition $J=U\Sigma V^T$ exposes weakly identifiable parameter combinations. Small singular values create large unregularized corrections and noise amplification.

If there are $n_p=r+1$ reduced parameters, $n_d$ observations and $n$ grid cells, a dense Jacobian stores $O(n_dn_p)$ values; a dense GN linear solve nominally costs $O(n_dn_p^2+n_p^3)$, **in addition to forward-model and Jacobian evaluation**. Do not report this as the total computational complexity without timing the PDE solver and derivative method.

## 8. Computing sensitivities

Finite differences approximate column $j$ by

$$J_{:,j}\approx\frac{\mathcal G(m+\epsilon e_j)-\mathcal G(m-\epsilon e_j)}{2\epsilon}.$$

Central differences use roughly $2n_p$ forward model evaluations per full Jacobian, plus baseline evaluations depending on the implementation. Step size involves truncation versus roundoff error; test sensitivity to $\epsilon$. Automatic differentiation, analytic sensitivities, and adjoints may be cheaper or more accurate when available.

### Adjoint state — brief overview only

For the forward residual $F(h,m)=0$, solve an adjoint equation for $\lambda$ to evaluate objective gradients without one separate sensitivity solve per parameter. This is useful for many unknowns and few scalar objectives, but **adjoint derivation and implementation are outside this lab's scope**.

## 9. A GN debugging checklist

1. Verify sign convention: residual $\mathcal G(m)-d$ versus $d-\mathcal G(m)$ changes the right-hand side.
2. Check forward-model output and observational units, ordering, and dimensions.
3. Check Jacobian with a directional finite-difference test: $(\mathcal G(m+tv)-\mathcal G(m))/t\approx Jv$.
4. Plot objective, head RMSE, $\|\delta\|$, and damping/regularization parameter by iteration.
5. Try several initial guesses; inspect stagnation and divergence.
6. Vary noise and the number of retained GRF modes; report validation error, not only fitting error.
7. Separate time for field generation, forward simulation, and optimization; report memory if possible.
8. Compare fixed budgets or convergence tolerances fairly; more iterations can trivially lower residual but raise runtime.

### Worked GN example (one scalar parameter)

Consider $\mathcal G(m)=m^2$, observation $d=9$, starting $m_0=2$. For unweighted, unregularized GN, $r=4-9=-5$ and $J=2m_0=4$:

$$\delta=-\frac{r}{J}=\frac54=1.25,\qquad m_1=3.25.$$

The new residual is $3.25^2-9=1.5625$ rather than $-5$. Repeated iterations approach $m=3$ from a positive initial guess. The problem also has $m=-3$ as another solution: initialization matters. At $m=0$, the Jacobian vanishes and the basic GN step is undefined.

### Questions and answers

1. **Why not use the exact Hessian?** Second derivatives of a costly PDE forward model can be expensive; GN exploits least-squares structure.
2. **What if $J$ is rank deficient?** There may be unidentifiable directions; use regularization, truncation, or a robust least-squares solver.
3. **Why can GN increase the objective?** Its quadratic approximation may be inaccurate for a large step; damping or a line search may be necessary.
4. **What does a small step norm mean?** Possibly convergence, poor scaling, strong damping, or stagnation. Check the residual and gradient too.
5. **Is low training head RMSE proof of good conductivity reconstruction?** No; evaluate known synthetic truth and held-out heads.
6. **Why does truncating modes help but also hurt?** Fewer parameters reduce cost and overfitting, but omit structures the data might otherwise recover.

---

# Part IV. Required laboratory: inverse modeling with `pyrga-ht`

## 10. Reproduce the published pyRGA example

The [PyPI example](https://pypi.org/project/pyrga-ht/) uses `generate_synthetic_field`, `prepare_physical_domain`, `hydraulic_tomography`, `observation_operator`, `add_noise`, `forward_model`, and `gauss_newton_dynamic_lambda`. The distribution is **`pyrga-ht`**, imported as **`pyrga`**. The published example uses a $64\times64$ grid, 50 retained modes, 400 realizations, Gaussian covariance, variance 1.0, `lx=0.15`, `ly=0.2`, and 10 GN iterations. Interpret the *units and normalization* of `lx` and `ly` only after examining the installed implementation.

```bash
python -m pip install -r requirements.txt
python examples/pyrga_groundwater_demo.py
python examples/optimizer_benchmark.py --grid 64 --modes 20 --noise 0.05 --iterations 8 --seed 17
```

`examples/pyrga_groundwater_demo.py` supplies the introductory visualizations. The separate benchmark script makes configuration and metrics explicit and writes machine-readable results. Depending on hardware and implementation, a full tomography inversion can be expensive; start with one case.

## 11. Design controlled experiments

For the **accuracy study**, vary one factor at a time while holding others fixed:

| Experiment | Suggested values | Core question |
|---|---|---|
| Noise level | 0, 0.01, 0.05, 0.10 | When does head fitting cease to imply recovery? |
| Retained modes | 10, 20, 50 | What is the accuracy–runtime tradeoff? |
| Initial coefficient scale | 0, random small, random larger | How initialization-sensitive is GN? |
| GN iterations | 3, 6, 10, 20 | Does objective plateau before the budget? |
| Spatial structure | isotropic versus $l_x\ne l_y$ | How does the prior affect recoverability? |
| Grid size | 32, 64, 96 (if supported) | How does computational cost grow? |

**Experimental design cautions:** rerun with multiple seeds, record the actual number of observations and pumping settings, and avoid varying random truth/noise when the goal is to isolate one hyperparameter. Check that each grid and mode choice is supported by the installed pyRGA release. For scalability, report wall time and peak RSS where available, and discuss whether elapsed time includes ensemble generation, forward simulations, Jacobian evaluations, and inversion.

### Metrics: separate three questions

**(A) Observation fit:**

$$\mathrm{RMSE}_{\rm head}=\sqrt{\frac1{n_d}\sum_i(\hat h_i-h_i^{\rm obs})^2},\qquad \mathrm{NRMSE}=\frac{\|\hat{\mathbf d}-\mathbf d\|_2}{\|\mathbf d\|_2}.$$

Compare initial to fitted predictions. For a noisy experiment also compare against **noise-free synthetic head truth** when available.

**(B) Conductivity reconstruction:** if the package's $V$ orientation and transform from estimated coefficients back to $K$ have been verified, compute

$$\mathrm{RMSE}_{\log K}=\sqrt{\frac1n\|\log\hat K-\log K_{\rm true}\|_2^2},\qquad r_{\log K}=\operatorname{corr}(\log\hat K,\log K_{\rm true}).$$

**Do not claim conductivity recovery from head fit alone.** Until reconstruction is verified against the package API, the benchmark intentionally reports head-fit metrics rather than guessing how to reshape $V$ or interpret `beta`.

**(C) Efficiency/scalability:** record total elapsed time, time in each stage, final data fit, parameter dimension, observation count and (optionally) peak memory. Plot elapsed time versus grid cells and versus retained modes on log–log axes; fit exploratory scaling exponents only with enough reliably measured cases.

### Deliverable: mini research report

Write a **2–4 page report** including (1) the exact pyRGA version, hardware, seed, covariance parameters and grid; (2) initial versus fitted scatter plots and loss/convergence curves; (3) a results table with head RMSE, runtime, and modes; (4) a reconstruction plot and log-$K$ RMSE **if validated**; and (5) a thoughtful discussion of nonuniqueness, regularization/damping, computational bottlenecks, and whether more modes actually improve **validation** performance.

## 12. Discussion questions (answer after running the experiment)

1. **Why does increasing the number of GRF modes improve representation but potentially worsen GN conditioning?**  
   **Expected answer:** a larger basis can capture more true structure, but introduces weakly constrained directions and a larger Jacobian/linear solve.
2. **If training head RMSE falls while $\mathrm{RMSE}_{\log K}$ worsens, what happened?**  
   **Expected answer:** possible noise fitting, nonuniqueness, inadequate regularization, or parameterization mismatch. Validate with held-out pumping tests.
3. **What part dominates scaling: covariance generation, forward PDE solves, Jacobian evaluation, or the small GN system?**  
   **Expected answer:** measurement, not a universal assumption, determines the dominant step. The answer depends on grid size, mode count, derivative implementation and solver.
4. **How would you distinguish a bad optimizer from an unidentifiable problem?**  
   **Expected answer:** inspect gradients and Jacobian singular values, vary initialization, check synthetic truth and ideal/noiseless data, and compare residual stagnation against an appropriate regularized solution.
5. **Why could a truncated prior give deceptively low posterior variance?**  
   **Expected answer:** excluded modes are treated as impossible rather than uncertain, unless truncation error is explicitly modeled.
6. **When would you choose LM-like damping over a raw GN step?**  
   **Expected answer:** near singular sensitivity matrices, poor initial conditions, or strongly nonlinear regions with unreliable full steps.

## References

- Zhao, Y. & Luo, J. (2020). *Reformulation of Bayesian Geostatistical Approach on Principal Components*. *Water Resources Research*, 56, e2019WR026732. https://doi.org/10.1029/2019WR026732
- pyRGA PyPI API and examples: https://pypi.org/project/pyrga-ht/
- Cressie, N. (1993). *Statistics for Spatial Data*.
- Chilès, J.-P. & Delfiner, P. (2012). *Geostatistics: Modeling Spatial Uncertainty*.
- Nocedal, J. & Wright, S. (2006). *Numerical Optimization*.
