# Independent Research Projects: Fluid Dynamics & Bio-Calculus
### Bridging IAL Physics Unit 1 (Mechanics & Materials) and IAL Biology Unit 5 (Energy, Exercise and Coordination)

Confirmed from your uploaded books:
- **phy.pdf** = Pearson Edexcel IAS/IAL Physics Book 1 — Topic 1 (Mechanics: 1A Motion, 1B Energy, 1C Momentum) and Topic 2 (Materials: 2A Fluids, 2B Solid Material Properties). This is **Unit 1: Mechanics and Materials**.
- **bio.pdf** = Pearson Edexcel IAL Biology Book 2 — Topic 7 (Respiration, Muscles and the Internal Environment) and Topic 8 (Coordination and Control). This is **Unit 5**.

**[Code available on GitHub](https://github.com/morshedrafi9/hpv_research_visualisation-01.git)**

Each project below deliberately sits at the overlap of **2A Fluids** (density, viscosity, Stokes' law, terminal velocity — spec refs in the "2A" chapter of phy.pdf) and **Topic 7** biology (cellular respiration, gas exchange, blood, muscle) so that the calculus you use is doing real biological work, not decoration.

---

## 0. How to use this document

For each project you get:
1. **Research question** (make this your own title — examiners/supervisors like a question, not a topic)
2. **Syllabus anchors** — exact sections to cite from your two books
3. **The calculus/physics derivation** ("bio-calculus") — do this by hand first, in your lab notebook
4. **Data you need** and where to get it for free (or how to collect it yourself, which is usually better for an IAL-style project since Papers 1–3 reward your own practical skills)
5. **Python code** — a working starting script (numpy/scipy/matplotlib), which you extend with your own data
6. **Extension / evaluation ideas** — for the discussion section of your report

Install once:
```bash
pip install numpy scipy matplotlib pandas
```

---

## Project 1 — Poiseuille's Law: why does artery radius matter so much?

**Research question:** *"How does small-vessel radius govern blood flow rate, and what does this predict about the physiological effect of atherosclerosis?"*

### Syllabus anchors
- Physics: 2A.1 "Fluids, density and upthrust", 2A.3 "Viscosity" (viscous drag, Stokes' law logic extends to flow in a pipe)
- Physics: 1A.7 Resolving vectors / 1B Work and power (relating pressure gradient to work done pushing fluid)
- Biology: Topic 7 — internal environment, transport of oxygen in blood, cellular respiration's dependency on oxygen supply rate

### The bio-calculus
Poiseuille derived the volumetric flow rate of a viscous fluid in laminar flow through a cylindrical vessel by integrating the parabolic velocity profile across the cross-section:

For radius from the centre, r, the velocity profile in a tube of radius R under pressure gradient ΔP/L and viscosity η is:

v(r) = (ΔP / 4ηL)(R² − r²)

Flow rate Q is the integral of v(r) over the circular cross-section (area element 2πr dr):

Q = ∫₀ᴿ v(r)·2πr dr = (π ΔP R⁴) / (8ηL)

**This is the calculus step you should show fully in your report** — differentiate/integrate it yourself symbolically (or verify with `sympy`) before you trust the simulation.

Key biological consequence: Q ∝ R⁴. A 19% reduction in vessel radius (typical of moderate atherosclerotic plaque) roughly **halves** blood flow at constant pressure gradient — link this explicitly to Topic 7's discussion of oxygen delivery and cellular respiration rate.

### Data you need
- Typical human vessel radii (aorta ~1.25 cm, arterioles ~15 μm, capillaries ~4 μm)
- Blood viscosity (~3–4 × 10⁻³ Pa·s, rises with haematocrit)
- Free literature values (no login needed):
  - **NIH/NCBI PubMed Central (PMC)** — open-access physiology papers, e.g. search "blood viscosity haematocrit PMC" (CC-BY licensed articles are reusable with attribution): https://www.ncbi.nlm.nih.gov/pmc/
  - **PhysioNet** (open, ODC-BY licensed physiological datasets, no signup for the open tier): https://physionet.org/about/database/ — good for real arterial pressure waveforms if you want to go further
  - You can also **measure viscosity yourself** using the falling-ball method exactly as in phy.pdf's Core Practical CP2 (2A "viscosity" section) with glycerol/golden syrup as a blood-analogue, then scale using the Stokes'-law relationship you already derive there.

### Python starting code
```python
import numpy as np
import matplotlib.pyplot as plt

def poiseuille_Q(delta_p, R, eta, L):
    """Volumetric flow rate (m^3/s) through a cylindrical vessel."""
    return (np.pi * delta_p * R**4) / (8 * eta * L)

# --- parameters (edit with your own literature/measured values) ---
eta = 3.5e-3      # Pa s, blood viscosity
L = 0.02           # m, vessel segment length
delta_p = 1000     # Pa, pressure drop along the segment

radii = np.linspace(0.5e-3, 2e-3, 200)   # small artery range, m
Q = poiseuille_Q(delta_p, radii, eta, L)

plt.figure()
plt.plot(radii*1e3, Q*1e6)   # mm vs mL/s
plt.xlabel("Vessel radius / mm")
plt.ylabel("Flow rate Q / mL s$^{-1}$")
plt.title("Poiseuille flow: Q ∝ R⁴")
plt.grid(True)
plt.show()

# quantify the effect of a 19% radius reduction (typical stenosis)
R0 = 1.0e-3
R_narrowed = 0.81 * R0
ratio = poiseuille_Q(delta_p, R_narrowed, eta, L) / poiseuille_Q(delta_p, R0, eta, L)
print(f"Flow falls to {ratio*100:.1f}% of normal for a 19% radius reduction")
```
Verify the R⁴ derivation symbolically:
```python
import sympy as sp
r, R, dP, eta_, L_ = sp.symbols('r R dP eta L', positive=True)
v = (dP/(4*eta_*L_)) * (R**2 - r**2)
Q = sp.integrate(v * 2*sp.pi*r, (r, 0, R))
print(sp.simplify(Q))
```

### Extension
Combine with 1B (Work and Power): calculate the **power** the heart must deliver to maintain a given flow rate as vessels narrow, using P = Q·ΔP, and relate this to cardiac work discussed in Topic 8 (coordination and control of heart rate/blood pressure).

---

## Project 2 — Stokes' Law and the Erythrocyte Sedimentation Rate (ESR)

**Research question:** *"Can Stokes' Law, verified using a school-laboratory falling-ball viscometer, be scaled to model the sedimentation of red blood cells used as a clinical inflammation marker (ESR)?"*

### Syllabus anchors
- Physics: 2A.3 Viscosity, 2A.4 Terminal velocity (CP2 core practical — "Use a falling-ball method to determine the viscosity of a liquid")
- Biology: Topic 7 — blood as tissue, cellular composition, aggregation of erythrocytes under inflammation (rouleaux formation increases effective particle radius)

### The bio-calculus
Newton's second law for a sphere falling in a viscous fluid, including buoyancy:

m(dv/dt) = mg − ρ_fluid·V·g − 6πηrv

This is a first-order linear ODE in v(t). Solve analytically:

v(t) = v_T (1 − e^(−t/τ)), where v_T = 2r²g(ρ_particle − ρ_fluid)/(9η), τ = m/(6πηr)

**Do the analytical solution by hand** (separation of variables/integrating factor) — this is your calculus evidence. Then verify it numerically in Python with `scipy.integrate.solve_ivp` and overlay both curves.

Biology link: ESR clinically uses exactly this equation (Stokes' Law) — during inflammation, fibrinogen causes red blood cells to clump (rouleaux), increasing effective r, which increases v_T ∝ r² — hence a raised ESR indicates inflammation. This is real bio-calculus, not analogy.

### Data you need
- Your own CP2 falling-ball data (steel ball in glycerol) — **primary data, use this**
- Real RBC dimensions: diameter ≈ 7–8 μm, density ≈ 1125 kg/m³, plasma viscosity ≈ 1.2 × 10⁻³ Pa·s, plasma density ≈ 1025 kg/m³
- Free reference ESR datasets:
  - **Kaggle "Blood Test / CBC" open datasets** (many are CC0-licensed): https://www.kaggle.com/datasets — search "ESR" or "complete blood count"
  - **UCI Machine Learning Repository** (free, cited academic use): https://archive.ics.uci.edu/

### Python starting code
```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

g = 9.81
def rbc_ode(t, v, r, eta, rho_p, rho_f):
    m = (4/3)*np.pi*r**3*rho_p
    drag = 6*np.pi*eta*r*v
    buoyancy = rho_f*(4/3)*np.pi*r**3*g
    weight = m*g
    return (weight - buoyancy - drag)/m

# single RBC vs rouleaux (clumped, effective radius x3)
params_single = dict(r=4e-6, eta=1.2e-3, rho_p=1125, rho_f=1025)
params_clumped = dict(r=12e-6, eta=1.2e-3, rho_p=1125, rho_f=1025)

t_span = (0, 3600)   # seconds (ESR measured over 1 hour clinically)
t_eval = np.linspace(*t_span, 500)

sol1 = solve_ivp(rbc_ode, t_span, [0], args=tuple(params_single.values()), t_eval=t_eval)
sol2 = solve_ivp(rbc_ode, t_span, [0], args=tuple(params_clumped.values()), t_eval=t_eval)

plt.plot(sol1.t, sol1.y[0]*1e6, label="single RBC")
plt.plot(sol2.t, sol2.y[0]*1e6, label="rouleaux (inflammation)")
plt.xlabel("time / s"); plt.ylabel("settling velocity / µm s$^{-1}$")
plt.legend(); plt.title("Stokes' Law: ESR mechanism"); plt.grid(True)
plt.show()

# compare numerical terminal velocity to the analytical formula
def v_terminal(r, eta, rho_p, rho_f):
    return 2*r**2*g*(rho_p-rho_f)/(9*eta)
print("analytical v_T (single):", v_terminal(**params_single))
print("analytical v_T (clumped):", v_terminal(**params_clumped))
```

### Extension
Use your CP2 raw data (ball radius, distances, times) to first **verify Stokes' Law experimentally** with a scatter/line-of-best-fit graph (as instructed in phy.pdf p.21's "straight-line graph" method), extract η for your test liquid, then only *afterwards* rescale the same equation to biological parameters. This two-step structure (verify → apply) is exactly what A03 (experimental skills) rewards.

---

## Project 3 — Fick's Law and Diffusion Across the Alveolar Membrane

**Research question:** *"Using Fick's First and Second Laws, how does alveolar membrane thickness and surface area limit the rate of oxygen diffusion into the blood, and how does this constrain the maximum rate of aerobic cellular respiration (Topic 7A)?"*

### Syllabus anchors
- Biology: 7A Cellular Respiration (oxygen demand for oxidative phosphorylation), gas exchange discussion in Topic 7
- Physics: 1A Motion (rate of change concepts — you're applying the same "gradient = rate" logic from d–t graphs, 1A.2, to concentration–distance graphs), 2A Fluids (diffusion is a fluid/molecular transport process)

### The bio-calculus
Fick's First Law (steady state):

J = −D (dC/dx)

Fick's Second Law (non-steady state, a PDE) governs how the concentration profile evolves with time:

∂C/∂t = D (∂²C/∂x²)

This is the diffusion equation — you solve it numerically using a **finite-difference method** (this is genuinely first university-level calculus applied at school level — an excellent "stretch and challenge" component for a Rice application-style project).

### Data you need
- Alveolar membrane thickness (~0.2–0.6 μm), diffusion coefficient of O₂ in tissue (~1–2 × 10⁻⁹ m²/s), alveolar surface area (~70 m² total across ~480 million alveoli)
- Free datasets:
  - **NASA/physiology open datasets** are scarce for this; better to cite published open-access values from **PMC** (as above) or **PhysoNet**
  - **WHO Global Health Observatory** (free, open license) for population-level respiratory data if you want an epidemiological extension: https://www.who.int/data/gho
  - Best primary data: measure **your own lung function** with a simple peak-flow meter or spirometer if your school has one, and use it to sanity-check your diffusion model's implied maximum O₂ uptake rate against measured VO₂.

### Python starting code
```python
import numpy as np
import matplotlib.pyplot as plt

D = 1.5e-9        # m^2/s, O2 diffusion coefficient in tissue
L = 0.5e-6         # m, membrane thickness
C_alveolar = 0.28   # arbitrary conc. units (partial pressure proxy), alveolar side
C_blood = 0.10       # capillary side (lower - being removed by blood flow)

nx = 100
dx = L/nx
dt = 0.4 * dx**2 / D     # stability criterion for explicit finite differences
nt = 2000

C = np.linspace(C_alveolar, C_blood, nx)   # initial linear guess

for step in range(nt):
    C_new = C.copy()
    C_new[1:-1] = C[1:-1] + D*dt/dx**2 * (C[2:] - 2*C[1:-1] + C[:-2])
    C_new[0] = C_alveolar   # boundary condition: fixed alveolar concentration
    C_new[-1] = C_blood     # boundary condition: fixed capillary concentration
    C = C_new
    if step % 400 == 0:
        plt.plot(np.linspace(0, L, nx)*1e6, C, label=f"t step {step}")

plt.xlabel("distance across membrane / µm")
plt.ylabel("O$_2$ concentration (arb. units)")
plt.legend()
plt.title("Fick's Second Law: diffusion reaching steady state")
plt.show()

flux = -D * (C[-1]-C[-2])/dx
print(f"Steady-state flux J ≈ {flux:.3e} units/(m^2 s)")
```

### Extension
Multiply your steady-state flux by total alveolar surface area to estimate maximum whole-lung O₂ uptake, then compare with literature VO₂max values, and discuss in Biology terms why this diffusion-limited step matters for the rate of the Krebs cycle/oxidative phosphorylation (link explicitly to fig C on the "Respiration in Cells" page of your bio.pdf).

---

## Project 4 — Differentiating and Integrating Real Respirometer Data (rate of respiration, oxygen debt)

**Research question:** *"How can numerical differentiation and integration of respirometer volume–time data be used to quantify instantaneous respiration rate and total oxygen debt after exercise?"*

### Syllabus anchors
- Biology: Topic 7A — "investigating the rate of respiration [using a respirometer]" (explicitly listed as a maths skill in your bio.pdf: "Calculate rate of change from a graph showing a linear relationship")
- Physics: 1A.2 Motion graphs (the exact same skill — gradient of a d–t graph = speed; area under a v–t graph = distance) — you are directly transferring an A-Level Physics skill into Biology, which is a strong "interdisciplinary methodology" angle for a personal statement/interview

### The bio-calculus
- **Rate of respiration** at any instant = dV/dt, found by differentiating your volume–time respirometer curve (numerically: central differences)
- **Total oxygen debt** after exercise = ∫(rate during recovery − resting rate) dt, i.e. the *extra* area under the oxygen-consumption curve above baseline — directly analogous to the "area under a v–t graph = distance" method your Physics book teaches in 1A.2 (fig C, boat example)

### Data you need
This project is ideally **self-collected**: build/use a simple respirometer (a core practical setup in most IAL Biology labs, using soda lime, a manometer, and a small organism such as woodlice/germinating peas) and log gas volume vs time before/during/after a controlled activity change.
- If you cannot access a respirometer, free open respiration/metabolic-rate datasets:
  - **PhysioNet** (open, no login for the open databases): https://physionet.org/about/database/ — e.g. gait/metabolic datasets
  - **Kaggle** "oxygen consumption" / "VO2" datasets (check each dataset's licence tag — many are CC0): https://www.kaggle.com/datasets
  - **Dryad** (open-access biological datasets, CC0 by default): https://datadryad.org — search "respirometry" or "oxygen consumption"

### Python starting code
```python
import numpy as np
import matplotlib.pyplot as plt

# Example respirometer data: replace with your own logged (t, V) pairs
t = np.array([0,1,2,3,4,5,6,7,8,9,10])       # minutes
V = np.array([0,0.9,1.7,2.4,3.0,4.8,6.4,7.6,8.4,8.9,9.2])  # mL O2 consumed (cumulative)

# --- numerical differentiation: instantaneous rate ---
rate = np.gradient(V, t)   # mL O2 per min

plt.figure()
plt.plot(t, V, 'o-', label="cumulative O$_2$ consumed")
plt.xlabel("time / min"); plt.ylabel("volume O$_2$ / mL")
plt.title("Respirometer data"); plt.legend(); plt.show()

plt.figure()
plt.plot(t, rate, 's-', color='crimson')
plt.xlabel("time / min"); plt.ylabel("rate dV/dt / mL min$^{-1}$")
plt.title("Instantaneous respiration rate"); plt.grid(True); plt.show()

# --- numerical integration: oxygen debt during recovery ---
resting_rate = rate[:2].mean()          # baseline before exercise
excess = np.clip(rate - resting_rate, 0, None)
oxygen_debt = np.trapz(excess, t)       # area above resting-rate line
print(f"Resting rate ≈ {resting_rate:.2f} mL/min")
print(f"Oxygen debt ≈ {oxygen_debt:.2f} mL O2")
```

### Extension
Fit an exponential recovery model (`scipy.optimize.curve_fit`) to the post-exercise decay of respiration rate back to baseline, and relate the recovery time-constant to anaerobic/lactate physiology described in Topic 7's "effect of lactate on muscle tissues" section.

---

## Project 5 (bonus, higher difficulty) — Bernoulli's Principle and Arterial Stenosis

**Research question:** *"Does the Bernoulli/continuity relationship correctly predict the pressure drop and velocity increase measured across an arterial narrowing, and what does this imply for the risk of vessel collapse (Venturi effect) in atherosclerosis?"*

### Syllabus anchors
- Physics: 1B Energy (conservation of energy — Bernoulli is literally energy-per-volume conservation, connecting directly to your gravitational PE/KE section), 2A Fluids
- Biology: Topic 8 (Coordination and Control — blood pressure regulation, baroreceptors)

### The bio-calculus
Continuity: A₁v₁ = A₂v₂ (conservation of mass/volume flow, itself derivable by integrating flux over cross-sectional area, as in Project 1)
Bernoulli: P₁ + ½ρv₁² = P₂ + ½ρv₂² (derived from the work–energy theorem, i.e. directly from your 1B "Work and Power" chapter, extended to a fluid element)

```python
import numpy as np
rho = 1060   # kg/m3, blood density
A1, A2 = 3.14e-6, 1.5e-6   # m^2, normal vs stenosed cross-section
v1 = 0.3      # m/s, typical arterial velocity
v2 = v1 * A1/A2
P1 = 12000    # Pa (arbitrary reference)
P2 = P1 + 0.5*rho*(v1**2 - v2**2)
print(f"v2 = {v2:.2f} m/s, pressure drop = {P1-P2:.1f} Pa")
```

---

## Summary table for your report's introduction

| Project | Physics 2A/1B concept | Biology Topic 7/8 concept | Calculus tool |
|---|---|---|---|
| 1. Poiseuille | Viscosity, fluid flow | Blood transport, O₂ delivery | Integration (velocity profile → flow rate) |
| 2. Stokes'/ESR | Terminal velocity, CP2 | Blood composition, inflammation marker | 1st-order ODE (analytical + numerical) |
| 3. Fick's Law | Diffusion as transport | Gas exchange, respiration rate limit | PDE (finite-difference solution) |
| 4. Respirometer | Motion-graph gradient/area method (1A.2) | Rate of respiration, oxygen debt, lactate | Numerical differentiation & integration |
| 5. Bernoulli | Energy conservation (1B) | Blood pressure regulation (Topic 8) | Algebraic energy conservation |

## General note on data licensing
Always record and cite: (a) source URL, (b) licence (CC0, CC-BY, ODC-BY, etc.), (c) access date. For a school research project, **your own primary data from the Core Practicals in phy.pdf (CP2) and the respirometer practical in bio.pdf is the strongest and safest source** — it needs no licence, demonstrates the A03 experimental skills objective directly, and every one of the five projects above is written so that self-collected data is a valid, often preferable, substitute for the online datasets listed.
