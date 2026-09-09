# R3: Physics-Informed Loss Formulation & Optimization Stability Proofs

**Document Version:** 1.0.0  
**Target Environment:** VS13 Dataset, PyTorch 2.x, NVIDIA Tesla T4 (16GB VRAM)  
**Target Metric:** Ensemble RMSE $< 6.5\text{ km/h}$  
**Author:** `worker_docs_theory`  

---

## 1. Executive Summary

This document establishes the theoretical formulation, physical kinematics constraints, and mathematical proofs of optimization stability for the loss functions used in the vehicle speed estimation pipeline. The legacy baseline relies on standard Mean Squared Error (MSE), which suffers from unbounded gradient growth in the presence of heavy-tailed acoustic anomalies and road recording noise (e.g., wind turbulence, tire splash, acoustic reflections, and horn honks). As documented in the VS13 dataset survey, vehicle classes with high frontal areas such as the Renault Scenic ($10.31\text{ km/h}$ baseline RMSE) and Citroen C4 Picasso ($9.37\text{ km/h}$ baseline RMSE) suffer disproportionately from outlier residuals that destabilize MSE gradient descent.

We formulate a multi-objective **Physics-Informed Robust Loss** comprising:
1. A **Smooth Cauchy Loss (Lorentzian M-estimator)** with scale parameter $\gamma = 5.0\text{ km/h}$, engineered with a strictly bounded, redescending influence function that completely rejects catastrophic outliers;
2. A **Kinematic Acceleration Regularizer** that penalizes unphysical vehicle speed accelerations exceeding the maximum tire-road friction limit ($a_{\max} = 30.0\text{ (km/h)/s}$);
3. A **Domain Boundary Penalty** enforcing highway domain bounds ($v \in [10, 140]\text{ km/h}$).

We provide complete, step-by-step mathematical proofs demonstrating:
- Unboundedness and gradient explosion vulnerability of standard MSE ($\lim_{|e|\to\infty} \|\nabla \mathcal{L}_{\text{MSE}}\| = \infty$);
- Exact bound of the Cauchy influence function ($\sup_e |\psi_{\text{Cauchy}}(e)| = \frac{\gamma}{2}$ at $e = \pm \gamma$) and its redescending property ($\lim_{|e|\to\infty} \psi_{\text{Cauchy}}(e) = 0$);
- Lipschitz continuity of the Huber loss gradient ($\|\nabla \mathcal{L}_{\text{Huber}}\| \le \delta$) and its optimization descent guarantee.

---

## 2. Physics-Informed Loss Formulation

### 2.1 Kinematic Vehicle Acceleration Constraints

In classical Newtonian mechanics, the motion of a road vehicle of mass $m$ is constrained by the tractive and braking forces generated at the tire-pavement contact patches. According to Coulomb's law of friction:
$$F_{\text{friction}} \le \mu \cdot F_N = \mu \cdot m \cdot g$$
where $\mu$ is the effective tire-road coefficient of friction, $F_N = mg$ is normal gravitational force, and $g \approx 9.81\text{ m/s}^2$ is gravitational acceleration.

Applying Newton's second law of motion ($F = m \cdot a$):
$$m \cdot |a(t)| \le \mu \cdot m \cdot g \implies |a(t)| = \left| \frac{dv(t)}{dt} \right| \le \mu \cdot g$$

For dry, high-traction asphalt roadways under standard ambient conditions, the maximum achievable longitudinal friction coefficient for passenger tires is typically $\mu \approx 0.85$. Under panic braking with Anti-lock Braking Systems (ABS), maximum deceleration reaches:
$$a_{\max} = \mu \cdot g \approx 0.85 \times 9.81\text{ m/s}^2 \approx 8.3385\text{ m/s}^2$$

Converting to vehicle speed units of kilometers per hour per second ($\text{(km/h)/s}$):
$$a_{\max} = 8.3385\text{ m/s}^2 \times \left( \frac{3600\text{ s/h}}{1000\text{ m/km}} \right) = 8.3385 \times 3.6 \approx 30.0186\text{ (km/h)/s}$$

**Physical Constraint:** No commercial road vehicle in the VS13 dataset can physically exceed a longitudinal acceleration or deceleration magnitude of:
$$a_{\max} = 30.0\text{ (km/h)/s}$$

---

### 2.2 Temporal Discretization of Kinematic Regularizer $\mathcal{L}_{\text{physics}}$

Consider an acoustic speed estimator that outputs either:
1. An auxiliary sequence of frame-level velocity predictions $\hat{\mathbf{v}} = [\hat{v}_1, \hat{v}_2, \dots, \hat{v}_M]^T$ across $M$ temporal frames, or
2. Local trajectory estimates computed across overlapping sub-windows of duration $\Delta t$.

Let the STFT hop length be $H = 512$ samples at sampling rate $f_s = 16,000\text{ Hz}$. The physical time interval between consecutive temporal frames is:
$$\Delta t = \frac{H}{f_s} = \frac{512}{16000} = 0.032\text{ s} \quad (32\text{ ms})$$

The maximum physically permissible velocity change between adjacent frames $m$ and $m+1$ is:
$$\Delta v_{\max} = a_{\max} \cdot \Delta t = 30.0\text{ (km/h)/s} \times 0.032\text{ s} = 0.96\text{ km/h per frame}$$

The discrete kinematic acceleration regularizer is formulated as a one-sided quadratic penalty:
$$\mathcal{L}_{\text{physics}}(\hat{\mathbf{v}}) = \frac{1}{M-1} \sum_{m=1}^{M-1} \max\left( 0, \ \left| \frac{\hat{v}_{m+1} - \hat{v}_m}{\Delta t} \right| - a_{\max} \right)^2$$
Equivalently, expressed in terms of frame delta threshold $\Delta v_{\max}$:
$$\mathcal{L}_{\text{physics}}(\hat{\mathbf{v}}) = \frac{1}{M-1} \sum_{m=1}^{M-1} \operatorname{ReLU}\left( |\hat{v}_{m+1} - \hat{v}_m| - \Delta v_{\max} \right)^2$$

#### Subgradient of Kinematic Regularizer:
For frame $m$, let $\delta_m = \hat{v}_{m+1} - \hat{v}_m$. The gradient with respect to $\hat{v}_{m+1}$ is:
$$\frac{\partial \mathcal{L}_{\text{physics}}}{\partial \hat{v}_{m+1}} = \begin{cases}
2 (|\delta_m| - \Delta v_{\max}) \operatorname{sgn}(\delta_m), & \text{if } |\delta_m| > \Delta v_{\max} \\
0, & \text{if } |\delta_m| \le \Delta v_{\max}
\end{cases}$$
When the predicted velocity trajectory conforms to Newtonian vehicle kinematics, this loss is identically zero ($\mathcal{L}_{\text{physics}} = 0$) and contributes zero gradient. When high-frequency noise induces erratic frame fluctuations ($|\delta_m| > \Delta v_{\max}$), the quadratic penalty provides restorative gradient pull proportional to the physical violation.

---

### 2.3 Physical Domain Boundary Constraint $\mathcal{L}_{\text{bound}}$

All recordings in the VS13 dataset correspond to highway and suburban vehicle passes with verified cruise-control speeds ranging from $30\text{ km/h}$ to $105\text{ km/h}$. Under extreme noise, an unregularized regression head can predict unphysical negative speeds ($\hat{v} < 0$) or supersonic velocities ($\hat{v} > 200\text{ km/h}$).

We define the admissible physical speed interval $[v_{\min}, v_{\max}] = [10.0, 140.0]\text{ km/h}$. The domain boundary regularizer is:
$$\mathcal{L}_{\text{bound}}(\hat{v}) = \operatorname{ReLU}(v_{\min} - \hat{v})^2 + \operatorname{ReLU}(\hat{v} - v_{\max})^2$$

---

## 3. Robust Statistical Loss Formulation

### 3.1 Failure Modes of Standard Mean Squared Error (MSE)

Let $y \in \mathbb{R}^+$ denote the ground truth vehicle pass-by speed, and let $\hat{y} = f_\theta(\mathbf{x})$ denote the neural network prediction. The residual error is defined as:
$$e = y - \hat{y}$$

Standard Mean Squared Error is defined as:
$$\mathcal{L}_{\text{MSE}}(e) = \frac{1}{2} e^2 = \frac{1}{2} (y - \hat{y})^2$$

In real-world acoustic field recordings, the empirical error distribution is non-Gaussian and heavy-tailed. Sources of acoustic outliers include:
1. Sudden aerodynamic wind gusts striking the microphone diaphragm;
2. Heavy roadside acoustic reverberation from passing tall commercial trucks;
3. Diffuse acoustic scattering from multi-purpose vehicles (e.g. Renault Scenic, Citroen C4 Picasso);
4. Erroneous acoustic triggers where tire noise is obscured by background vehicle passes.

Under these conditions, a single recording with an error of $|e| = 50\text{ km/h}$ incurs an MSE penalty of:
$$\mathcal{L}_{\text{MSE}} = \frac{1}{2} (50)^2 = 1250$$
which is $625\times$ larger than a standard error of $|e| = 2\text{ km/h}$ ($\mathcal{L}_{\text{MSE}} = 2$). In mini-batch stochastic gradient descent, the outlier sample completely dominates the batch gradient vector, triggering weight displacement and destroying learned acoustic representations.

---

### 3.2 Smooth Cauchy Loss (Lorentzian M-Estimator)

In robust M-estimation (Huber, 1981), an estimator minimizes an objective $\sum_i \rho(e_i)$, where the influence function $\psi(e) = \rho'(e)$ governs the sensitivity of the parameter estimate to individual residuals.

We formulate the **Smooth Cauchy Loss** (Lorentzian loss):
$$\mathcal{L}_{\text{Cauchy}}(e; \gamma) = \frac{\gamma^2}{2} \ln\left( 1 + \frac{e^2}{\gamma^2} \right) = \frac{\gamma^2}{2} \ln\left( 1 + \left(\frac{y - \hat{y}}{\gamma}\right)^2 \right)$$
where $\gamma > 0$ is a scale parameter in units of speed ($\text{km/h}$). For the VS13 dataset, we calibrate $\gamma = 5.0\text{ km/h}$.

#### Asymptotic Properties of Cauchy Loss:
1. **Near-Zero Residuals ($|e| \ll \gamma$):**  
   Taylor expansion of $\ln(1 + u)$ for $u = e^2/\gamma^2 \ll 1$:
   $$\mathcal{L}_{\text{Cauchy}}(e) = \frac{\gamma^2}{2} \left[ \frac{e^2}{\gamma^2} - \frac{1}{2} \left(\frac{e^2}{\gamma^2}\right)^2 + \mathcal{O}\left( \frac{e^6}{\gamma^6} \right) \right] \approx \frac{1}{2} e^2$$
   The Cauchy loss behaves exactly like standard MSE for small residuals, ensuring rapid quadratic convergence in the vicinity of the optimal solution.
2. **Large Outlier Residuals ($|e| \gg \gamma$):**  
   $$\mathcal{L}_{\text{Cauchy}}(e) \approx \frac{\gamma^2}{2} \ln\left( \frac{e^2}{\gamma^2} \right) = \gamma^2 \ln\left( \frac{|e|}{\gamma} \right)$$
   The penalty grows logarithmically rather than quadratically, curbing the dominance of large residuals.

---

### 3.3 Huber Loss Formulation

The Huber loss provides a smooth transition between quadratic loss and linear loss:
$$\mathcal{L}_{\text{Huber}}(e; \delta) = \begin{cases}
\frac{1}{2} e^2, & \text{if } |e| \le \delta \\
\delta \left( |e| - \frac{1}{2} \delta \right), & \text{if } |e| > \delta
\end{cases}$$
where $\delta > 0$ is the transition threshold, chosen as $\delta = 5.0\text{ km/h}$.

---

## 4. Mathematical Proofs of Bounds and Gradient Stability

### 4.1 Influence Function Definition

In robust statistics, the **influence function** $\psi(e)$ measures the marginal influence of an infinitesimal observation error $e$ on the objective gradient:
$$\psi(e) = \frac{\partial \mathcal{L}(e)}{\partial e}$$
The gradient of the loss with respect to the network prediction $\hat{y}$ satisfies:
$$\nabla_{\hat{y}} \mathcal{L} = \frac{\partial \mathcal{L}}{\partial e} \frac{\partial e}{\partial \hat{y}} = -\psi(e)$$

---

### 4.2 Theorem 1: Unbounded Gradient Instability of MSE

**Theorem 1 (MSE Gradient Divergence).**  
*The gradient of the standard Mean Squared Error loss is unbounded on $\mathbb{R}$. Specifically:*
$$\|\nabla_{\hat{y}} \mathcal{L}_{\text{MSE}}\| = |e|$$
$$\lim_{|e| \to \infty} \|\nabla_{\hat{y}} \mathcal{L}_{\text{MSE}}\| = \infty$$
*Consequently, standard MSE has an unbounded influence function and does not satisfy global Lipschitz continuity of the loss.*

#### Proof:
By definition:
$$\mathcal{L}_{\text{MSE}}(e) = \frac{1}{2} e^2$$
Differentiating with respect to residual $e$:
$$\psi_{\text{MSE}}(e) = \frac{d\mathcal{L}_{\text{MSE}}}{de} = e$$
The gradient with respect to prediction $\hat{y}$ is:
$$\nabla_{\hat{y}} \mathcal{L}_{\text{MSE}} = -e = -(y - \hat{y})$$
The Euclidean norm of the gradient is:
$$\|\nabla_{\hat{y}} \mathcal{L}_{\text{MSE}}\| = |e|$$
Taking the supremum over all possible residuals $e \in \mathbb{R}$:
$$\sup_{e \in \mathbb{R}} \|\nabla_{\hat{y}} \mathcal{L}_{\text{MSE}}\| = \sup_{e \in \mathbb{R}} |e| = \infty$$

If a corrupt audio recording yields a residual $|e| \to \infty$, the gradient norm diverges to infinity. Under gradient descent with learning rate $\eta > 0$, the parameter update $\Delta \theta = -\eta \nabla_\theta \mathcal{L} = \eta \cdot e \cdot \nabla_\theta \hat{y}$ is unbounded, leading to numerical divergence or severe destabilization of learned weights. $\blacksquare$

---

### 4.3 Theorem 2: Bounded & Redescending Influence of Smooth Cauchy Loss

**Theorem 2 (Cauchy Gradient Bounds & Redescending Property).**  
*Let $\mathcal{L}_{\text{Cauchy}}(e; \gamma) = \frac{\gamma^2}{2} \ln(1 + e^2/\gamma^2)$ with scale $\gamma > 0$. Then:*
1. *The influence function $\psi_{\text{Cauchy}}(e) = \frac{e}{1 + (e/\gamma)^2}$ is globally bounded on $\mathbb{R}$:*
   $$\sup_{e \in \mathbb{R}} |\psi_{\text{Cauchy}}(e)| = \frac{\gamma}{2}$$
   *with the extrema occurring at exactly $e^* = \pm \gamma$.*
2. *The influence function satisfies the redescending property:*
   $$\lim_{|e| \to \infty} \psi_{\text{Cauchy}}(e) = 0$$
3. *The second derivative (Hessian component) is uniformly bounded on $\mathbb{R}$:*
   $$-\frac{1}{8} \le \frac{d^2 \mathcal{L}_{\text{Cauchy}}}{de^2} \le 1$$

#### Proof:

**Part 1: Derivation of Extremum and Maximum Gradient**  
Differentiating $\mathcal{L}_{\text{Cauchy}}(e)$ with respect to $e$:
$$\psi_{\text{Cauchy}}(e) = \frac{d}{de} \left[ \frac{\gamma^2}{2} \ln\left( 1 + \frac{e^2}{\gamma^2} \right) \right] = \frac{\gamma^2}{2} \cdot \frac{1}{1 + \frac{e^2}{\gamma^2}} \cdot \frac{2e}{\gamma^2} = \frac{e}{1 + \frac{e^2}{\gamma^2}} = \frac{\gamma^2 e}{\gamma^2 + e^2}$$

To find the critical points of $\psi_{\text{Cauchy}}(e)$, compute its derivative $\frac{d\psi}{de}$:
$$\frac{d\psi}{de} = \frac{d}{de} \left[ \frac{e}{1 + e^2/\gamma^2} \right] = \frac{\left(1 + \frac{e^2}{\gamma^2}\right) \cdot 1 - e \cdot \left( \frac{2e}{\gamma^2} \right)}{\left( 1 + \frac{e^2}{\gamma^2} \right)^2} = \frac{1 + \frac{e^2}{\gamma^2} - \frac{2e^2}{\gamma^2}}{\left( 1 + \frac{e^2}{\gamma^2} \right)^2} = \frac{1 - \frac{e^2}{\gamma^2}}{\left( 1 + \frac{e^2}{\gamma^2} \right)^2}$$

Setting the derivative to zero to identify local extrema:
$$\frac{d\psi}{de} = 0 \iff 1 - \frac{e^2}{\gamma^2} = 0 \iff e^2 = \gamma^2 \iff e^* = \pm \gamma$$

Evaluating $\psi_{\text{Cauchy}}(e)$ at the critical points:
$$\psi_{\text{Cauchy}}(\gamma) = \frac{\gamma}{1 + \frac{\gamma^2}{\gamma^2}} = \frac{\gamma}{1 + 1} = \frac{\gamma}{2}$$
$$\psi_{\text{Cauchy}}(-\gamma) = \frac{-\gamma}{1 + \frac{(-\gamma)^2}{\gamma^2}} = \frac{-\gamma}{1 + 1} = -\frac{\gamma}{2}$$

Evaluating the asymptotic limits as $e \to \pm \infty$:
$$\lim_{e \to +\infty} \psi_{\text{Cauchy}}(e) = \lim_{e \to +\infty} \frac{e}{1 + e^2/\gamma^2} = \lim_{e \to +\infty} \frac{\frac{1}{e}}{\frac{1}{e^2} + \frac{1}{\gamma^2}} = \frac{0}{0 + 1/\gamma^2} = 0$$
$$\lim_{e \to -\infty} \psi_{\text{Cauchy}}(e) = 0$$

Because $\psi(e)$ is continuous, differentiable, odd ($\psi(-e) = -\psi(e)$), and vanishes at infinity, the global maximum occurs at $e = +\gamma$ and the global minimum occurs at $e = -\gamma$. Therefore:
$$\sup_{e \in \mathbb{R}} |\psi_{\text{Cauchy}}(e)| = \frac{\gamma}{2}$$
For $\gamma = 5.0\text{ km/h}$, the maximum gradient magnitude is:
$$\sup_{e \in \mathbb{R}} \|\nabla_{\hat{y}} \mathcal{L}_{\text{Cauchy}}\| = \frac{5.0}{2} = 2.50$$
Regardless of how massive the error $e$ is (even if $|e| = 100\text{ km/h}$), the gradient update can **never exceed $2.50$**.

---

**Part 2: Redescending Property**  
As shown above:
$$\lim_{|e| \to \infty} \psi_{\text{Cauchy}}(e) = \lim_{|e| \to \infty} \frac{e}{1 + e^2/\gamma^2} = 0$$

**Physical Consequence:** An observation corrupted by massive non-Gaussian noise ($|e| \gg \gamma$) exerts **zero gradient force** on the network parameters. Rather than pulling weights toward the outlier, the Cauchy estimator smoothly downweights and rejects it entirely.

---

**Part 3: Uniformly Bounded Hessian**  
The second derivative of $\mathcal{L}_{\text{Cauchy}}$ with respect to $e$ is:
$$\mathcal{H}_{\text{Cauchy}}(e) = \frac{d^2 \mathcal{L}_{\text{Cauchy}}}{de^2} = \frac{d\psi}{de} = \frac{1 - u}{(1 + u)^2}, \quad \text{where } u = \frac{e^2}{\gamma^2} \ge 0$$
Let $h(u) = \frac{1 - u}{(1 + u)^2}$ for $u \in [0, \infty)$.
1. At $u = 0$ ($e = 0$):
   $$h(0) = \frac{1 - 0}{(1 + 0)^2} = 1$$
2. Differentiating $h(u)$ with respect to $u$:
   $$h'(u) = \frac{-(1 + u)^2 - (1 - u) \cdot 2(1 + u)}{(1 + u)^4} = \frac{-(1 + u) - 2(1 - u)}{(1 + u)^3} = \frac{-1 - u - 2 + 2u}{(1 + u)^3} = \frac{u - 3}{(1 + u)^3}$$
   Setting $h'(u) = 0 \implies u = 3 \implies e = \pm \sqrt{3}\gamma$.
3. Evaluating $h(u)$ at the minimum $u = 3$:
   $$h(3) = \frac{1 - 3}{(1 + 3)^2} = \frac{-2}{16} = -\frac{1}{8} = -0.125$$
4. As $u \to \infty$: $\lim_{u \to \infty} h(u) = 0$.

Therefore, the second derivative is strictly bounded:
$$-\frac{1}{8} \le \frac{d^2 \mathcal{L}_{\text{Cauchy}}}{de^2} \le 1$$
This guarantees that the curvature of the loss surface is strictly controlled, preventing sudden gradient explosions during backpropagation. $\blacksquare$

---

### 4.4 Theorem 3: Gradient Bounds & Lipschitz Continuity of Huber Loss

**Theorem 3 (Huber Lipschitz Continuity & Convergence Stability).**  
*Let $\mathcal{L}_{\text{Huber}}(e; \delta)$ be defined as above. Then:*
1. *The gradient $\nabla_{\hat{y}} \mathcal{L}_{\text{Huber}}$ is globally bounded:*
   $$\sup_{e \in \mathbb{R}} \|\nabla_{\hat{y}} \mathcal{L}_{\text{Huber}}\| = \delta$$
2. *The gradient $\nabla_{\hat{y}} \mathcal{L}_{\text{Huber}}$ is globally Lipschitz continuous with Lipschitz constant $L = 1$:*
   $$\|\nabla \mathcal{L}_{\text{Huber}}(e_1) - \nabla \mathcal{L}_{\text{Huber}}(e_2)\| \le 1 \cdot |e_1 - e_2| \quad \forall e_1, e_2 \in \mathbb{R}$$
3. *Under gradient descent with step size $\eta < 2$, the sequence of loss values is monotonically non-increasing and converges to a stationary point.*

#### Proof:

**Part 1: Gradient Bound**  
Differentiating $\mathcal{L}_{\text{Huber}}(e)$ with respect to $e$:
$$\psi_{\text{Huber}}(e) = \frac{d\mathcal{L}_{\text{Huber}}}{de} = \begin{cases}
e, & \text{if } |e| \le \delta \\
\delta \operatorname{sgn}(e), & \text{if } |e| > \delta
\end{cases}$$
The gradient with respect to $\hat{y}$ is:
$$\nabla_{\hat{y}} \mathcal{L}_{\text{Huber}} = -\psi_{\text{Huber}}(e) = \begin{cases}
-e, & \text{if } |e| \le \delta \\
-\delta \operatorname{sgn}(e), & \text{if } |e| > \delta
\end{cases}$$
Its norm satisfies:
$$\|\nabla_{\hat{y}} \mathcal{L}_{\text{Huber}}\| = \begin{cases}
|e| \le \delta, & |e| \le \delta \\
\delta, & |e| > \delta
\end{cases} \implies \sup_{e \in \mathbb{R}} \|\nabla_{\hat{y}} \mathcal{L}_{\text{Huber}}\| = \delta$$
For $\delta = 5.0\text{ km/h}$, the gradient norm is globally capped at $5.0$.

---

**Part 2: Lipschitz Continuity of Gradient**  
Consider two arbitrary residuals $e_1, e_2 \in \mathbb{R}$. We examine the difference ratio:
$$\frac{|\psi(e_1) - \psi(e_2)|}{|e_1 - e_2|}$$
- Case 1: Both $|e_1| \le \delta$ and $|e_2| \le \delta$:
  $$|\psi(e_1) - \psi(e_2)| = |e_1 - e_2| \implies \frac{|\psi(e_1) - \psi(e_2)|}{|e_1 - e_2|} = 1$$
- Case 2: Both $e_1 > \delta$ and $e_2 > \delta$:
  $$\psi(e_1) = \delta, \ \psi(e_2) = \delta \implies |\psi(e_1) - \psi(e_2)| = 0 \le |e_1 - e_2|$$
- Case 3: $e_1 > \delta$ and $|e_2| \le \delta$:
  $$|\psi(e_1) - \psi(e_2)| = |\delta - e_2| = \delta - e_2 < e_1 - e_2 = |e_1 - e_2|$$
- Case 4: $e_1 > \delta$ and $e_2 < -\delta$:
  $$|\psi(e_1) - \psi(e_2)| = |\delta - (-\delta)| = 2\delta < e_1 - e_2 = |e_1 - e_2|$$

In all cases:
$$|\psi(e_1) - \psi(e_2)| \le 1 \cdot |e_1 - e_2| \quad \forall e_1, e_2 \in \mathbb{R}$$
Thus, $\psi(e)$ is Lipschitz continuous with Lipschitz constant $L = 1$.

---

**Part 3: Optimization Descent Guarantee**  
By the Descent Lemma (Nesterov, 2004; Bertsekas, 1999) for $L$-smooth functions:
$$\mathcal{L}(\theta_{k+1}) \le \mathcal{L}(\theta_k) + \langle \nabla \mathcal{L}(\theta_k), \theta_{k+1} - \theta_k \rangle + \frac{L}{2} \|\theta_{k+1} - \theta_k\|^2$$
Under a gradient descent update $\theta_{k+1} = \theta_k - \eta \nabla \mathcal{L}(\theta_k)$:
$$\mathcal{L}(\theta_{k+1}) \le \mathcal{L}(\theta_k) - \eta \|\nabla \mathcal{L}(\theta_k)\|^2 + \frac{L \eta^2}{2} \|\nabla \mathcal{L}(\theta_k)\|^2 = \mathcal{L}(\theta_k) - \eta \left( 1 - \frac{L \eta}{2} \right) \|\nabla \mathcal{L}(\theta_k)\|^2$$

For $L = 1$ and any learning rate $\eta \in (0, 2)$:
$$\eta \left( 1 - \frac{\eta}{2} \right) > 0$$
Hence:
$$\mathcal{L}(\theta_{k+1}) - \mathcal{L}(\theta_k) \le -\eta \left( 1 - \frac{\eta}{2} \right) \|\nabla \mathcal{L}(\theta_k)\|^2 \le 0$$
Every gradient descent step strictly decreases the loss until a stationary point $\|\nabla \mathcal{L}\| = 0$ is reached. This convergence guarantee is impossible under standard MSE with unbounded gradients. $\blacksquare$

---

## 5. Optimization Dynamics Comparison Matrix

The table below summarizes the analytical and optimization properties of the candidate loss functions:

| Property | Mean Squared Error (MSE) | Mean Absolute Error (MAE / L1) | Huber Loss ($\delta = 5.0$) | Smooth Cauchy Loss ($\gamma = 5.0$) |
| :--- | :---: | :---: | :---: | :---: |
| **Formula $\mathcal{L}(e)$** | $\frac{1}{2} e^2$ | $|e|$ | $\begin{cases} \frac{1}{2}e^2 & \|e\|\le\delta \\ \delta(\|e\|-\frac{1}{2}\delta) & \|e\|>\delta \end{cases}$ | $\frac{\gamma^2}{2} \ln\left(1 + \frac{e^2}{\gamma^2}\right)$ |
| **Gradient $\nabla_{\hat{y}} \mathcal{L}$** | $-e$ | $-\operatorname{sgn}(e)$ | $\begin{cases} -e & \|e\|\le\delta \\ -\delta\operatorname{sgn}(e) & \|e\|>\delta \end{cases}$ | $-\frac{e}{1 + (e/\gamma)^2}$ |
| **Influence Function $\psi(e)$** | $e$ | $\operatorname{sgn}(e)$ | $\operatorname{clip}(e, -\delta, \delta)$ | $\frac{e}{1 + (e/\gamma)^2}$ |
| **Maximum Influence $\sup \|\psi(e)\|$** | $\infty$ (Unbounded) | $1$ | $\delta = 5.0$ | $\frac{\gamma}{2} = 2.50$ |
| **Extremum Location $e^*$** | $e \to \infty$ | $\forall e \neq 0$ | $|e| \ge \delta$ | $e^* = \pm \gamma = \pm 5.0\text{ km/h}$ |
| **Asymptotic Limit $\lim_{\|e\|\to\infty} \psi(e)$** | $\infty$ (Explosion) | $\pm 1$ (Persistent) | $\pm \delta$ (Constant) | **$0$ (Redescending / Nullified)** |
| **Differentiability** | $C^\infty$ everywhere | Non-diff at $e=0$ | $C^1$ everywhere | $C^\infty$ everywhere |
| **Hessian $\frac{d^2\mathcal{L}}{de^2}$** | $1$ (Constant) | $0$ ($e \neq 0$), Dirac $\delta(0)$ | $\begin{cases} 1 & \|e\|<\delta \\ 0 & \|e\|>\delta \end{cases}$ | $\frac{1 - (e/\gamma)^2}{(1 + (e/\gamma)^2)^2} \in [-0.125, 1.0]$ |
| **Robustness to Road Acoustic Outliers** | ❌ Vulnerable ($0\%$) | ⚠️ Moderate | ✅ High | **🏆 Maximum (Zero-weight Outliers)** |

---

## 6. Multi-Objective Total Loss Formulation

Combining the robust regression criterion with kinematic physical regularization yields the complete training objective:
$$\mathcal{L}_{\text{total}}(\mathbf{y}, \hat{\mathbf{y}}, \hat{\mathbf{v}}) = \mathcal{L}_{\text{Cauchy}}(\mathbf{y}, \hat{\mathbf{y}}; \gamma) + \lambda_{\text{physics}} \mathcal{L}_{\text{physics}}(\hat{\mathbf{v}}) + \lambda_{\text{bound}} \mathcal{L}_{\text{bound}}(\hat{\mathbf{y}})$$

### Hyperparameter Specifications for `src/config.py`:
- Robust Loss Type: `LOSS_TYPE = 'cauchy'`
- Cauchy Scale: `CAUCHY_GAMMA = 5.0` ($\text{km/h}$)
- Huber Threshold (Alternative): `HUBER_DELTA = 5.0` ($\text{km/h}$)
- Kinematic Acceleration Weight: `PHYSICS_WEIGHT = 0.10`
- Maximum Acceleration Limit: `MAX_ACCELERATION = 30.0` ($\text{(km/h)/s}$)
- Speed Bounds: `SPEED_MIN = 10.0`, `SPEED_MAX = 140.0` ($\text{km/h}$)
- Boundary Penalty Weight: `BOUND_WEIGHT = 0.05`

### Implementation Blueprint for `src/losses.py`:
```python
# physics informed robust loss implementation
import torch
import torch.nn as nn

class SmoothCauchyLoss(nn.Module):
    def __init__(self, gamma: float = 5.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = pred - target
        return torch.mean(0.5 * (self.gamma ** 2) * torch.log(1.0 + (error / self.gamma) ** 2))

class KinematicAccelerationLoss(nn.Module):
    def __init__(self, max_accel: float = 30.0, hop_length: int = 512, sample_rate: int = 16000):
        super().__init__()
        self.dt = hop_length / sample_rate
        self.max_delta_v = max_accel * self.dt

    def forward(self, v_trajectory: torch.Tensor) -> torch.Tensor:
        # v_trajectory: [B, M]
        diffs = torch.abs(v_trajectory[:, 1:] - v_trajectory[:, :-1])
        violation = torch.relu(diffs - self.max_delta_v)
        return torch.mean(violation ** 2)
```

---

## 7. Summary of Loss Formulation Conclusions

1. Standard MSE is fundamentally fragile on the VS13 dataset because its influence function is unbounded ($\|\nabla\| = |e| \to \infty$), causing training instability on challenging classes such as the Renault Scenic.
2. The Smooth Cauchy Loss guarantees a strictly bounded gradient with supremum $\sup_e \|\nabla\| = \gamma / 2 = 2.50\text{ km/h}$, and possesses the redescending property ($\lim_{|e|\to\infty} \psi(e) = 0$), completely shielding network optimization from heavy-tailed acoustic recording anomalies.
3. The Kinematic Acceleration Regularizer incorporates Coulomb friction constraints ($|a| \le \mu g \approx 30\text{ (km/h)/s}$), penalizing unphysical frame-to-frame velocity discontinuities while contributing zero penalty when trajectory predictions respect physical vehicular kinematics.
