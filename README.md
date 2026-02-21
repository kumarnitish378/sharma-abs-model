The **Sharma Absolute Boundary Summation (ABS) Model** is a mathematically stable and computationally efficient multiplicative noise heuristic designed for physical simulations. Developed by **Nitish Sharma**, it is specifically engineered to replace expensive Gaussian approximations in resource-constrained environments such as embedded firmware (MCU), high-speed PCB testing, and real-time game engines.

### Mathematical Specification

The model generates a dimensionless noise factor, $\eta$, representing the ratio of a system's potential boundary range to its current stochastic state.

* 
**The Core Noise Factor ($\eta$):** 

$$ \eta = \frac{n(|\psi_{min}| + |\psi_{max}|)}{n(|\psi_{min}| + |\psi_{max}|) + \sum_{i=1}^{n} r_i} $$

* 
**$\psi_{min}, \psi_{max}$**: The per-sample lower and upper integer bounds defining the "Randomness Horizon".


* 
**$n$ (Complexity)**: The number of random samples per cycle; increasing $n$ narrows the variance.


* 
**$r_i$**: A random integer sampled from the uniform distribution $[\psi_{min}, \psi_{max}]$.


* 
**Numerator**: Acts as a stabilizing constant to prevent division by zero and ensure the factor oscillates around 1.0.




* **Perturbation Modulation:** To control noise intensity on a deterministic signal, an influence coefficient $\alpha$ is applied:

$$ v_{real} = v_{ideal} \cdot [1 + \alpha(\eta - 1)] $$

This allows a transition from "Perfectly Deterministic" ($\alpha = 0$) to "Maximum Turbulence" ($\alpha = 1$).



---

### Implementation and Logic

When implementing the model, developers should adhere to specific architectural requirements:

* 
**Input Validation**: Ensure $n > 0$.


* 
**Stability Guard**: Always check if the denominator equals zero; if it does, default to 1.0.


* 
**Scaling**: Results should be returned as high-precision floats or fixed-point integers for MCUs.



### Manual & Usage Guide

#### 1. Core C/Embedded Library
Includes a single-header implementation `src/sharma_abs.h`.
```c
#include "src/sharma_abs.h"

// Generate a noise factor (min=-10, max=10, complexity=12)
float eta = compute_eta(-10, 10, 12);
// Apply noise to an ideal value with influence alpha = 0.5
float noisy_val = apply_sharma_noise(100.0f, eta, 0.5f);
```

#### 2. Python Core Library
A class-based Python implementation `src/sharma_abs.py`.
```python
import sys
sys.path.append('src')
from sharma_abs import SharmaABS

# Initialize with min=-10, max=10, complexity=12
model = SharmaABS(-10, 10, 12)
# Apply to deterministic data
noisy_data = model.generate_noisy_dataset([1.0, 2.0, 3.0], alpha=0.5)
```

#### 3. CLI Data Generator
A versatile command-line utility `tools/abs_gen.py` tailored for processing deterministic data arrays.
```bash
python tools/abs_gen.py ideal_data.csv noisy_output.csv --min -10 --max 10 -n 12 -a 0.5 --col 1 --plot
```

#### 4. Web Dashboard
Open `public/index.html` in any modern web browser to interact with the real-time simulation dashboard. Features include:
* Interactive sliders for parameters like bounds, complexity, and influence $\alpha$.
* Multiple physical presets (e.g., Free Fall, Damped Oscillation, Financial Shocks).
* A live Math Box displaying formula breakdowns and realtime statistical distortions.

---

### Use Case Applications

1. **Embedded Signal Jitter (ADCS/IMUs)**: Used to simulate sensor noise for testing firmware error-handling. The ABS model ensures simulated noise does not push voltages into impossible hardware states, such as negative voltage on a 0-5V rail.


2. **Gaming Physics**: Used for wind gusts or recoil patterns. Multiplicative noise scales with velocity, providing a more realistic feel than static additive noise.


3. **Financial "Black Swan" Simulation**: Models market shocks by using a dynamic $n$. Decreasing $n$ increases variance to simulate localized volatility spikes without breaking underlying trends.



### Statistical Phenomenon: The "Magic 11"

The model identifies a unique **Quantization Bottleneck**. When complexity is low ($n=12$) and bounds are symmetric ($\psi_{min} = -12, \psi_{max} = 11$), the resulting values of $\eta$ occupy only 11–19 unique states over thousands of trials. This "pixelated" randomness is useful for "Lo-Fi" digital simulations or retro-game physics.

---

### Metadata

* 
**Author**: Nitish Sharma 


* 
**viXra ID**: 17827727 


* 
**License**: MIT License 


* **Citation**: Sharma, N. (2025). *Stochastic Normalization via Absolute Boundary Summation (ABS): The Sharma Multiplicative Noise Model for Physical Simulations*.

