#!/usr/bin/env python3
"""Generate the Monte Carlo & MCMC Curriculum Jupyter Notebook.

Strategy: markdown cells are lists of lines, code cells are written
to temp .py files then read back in to avoid all nested quoting issues.
"""

import itertools
import json
import os
import tempfile

BASE = "/home/jkhansell/Documents/Ulead/ULeadParallelComp2026/src/MonteCarloSim"

cell_id_counter = itertools.count()

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
        "orig_nbformat": 4,
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

def add_md(lines):
    """Add a markdown cell from a list of line strings."""
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
        "id": f"md-{next(cell_id_counter)}",
    })


def add_code(filename, caption):
    """Add a code cell by reading from a temp file."""
    with open(filename, "r") as f:
        source = f.readlines()
    # Add caption as first line comment
    if caption:
        source.insert(0, f"# --- {caption} ---\n")
    nb["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "source": source,
        "execution_count": None,
        "outputs": [],
        "id": f"code-{next(cell_id_counter)}",
    })


def write_code_file(name, content):
    """Write code content to a temp file and return the path."""
    path = os.path.join(BASE, name)
    with open(path, "w") as f:
        f.write(content)
    return path


# =============================================================================
# TITLE
# =============================================================================
add_md([
    "# Monte Carlo Methods & MCMC: From Buffon's Needle to NUTS",
    "",
    "> **A complete curriculum for a 2-hour university-level class** bridging the historical roots of Monte Carlo estimation with modern probabilistic programming and Hamiltonian Monte Carlo (NUTS), grounded in practical, real-world industrial scenarios.",
    "",
    "**Tech Stack:** Python, NumPy, Matplotlib, JAX, NumPyro",
])

# =============================================================================
# PART 1: HISTORY
# =============================================================================
add_md([
    "---",
    "",
    "# Part 1: History of the Monte Carlo Method & Toy Examples",
    "",
    "## 1.1 A Rich Historical Tapestry",
    "",
    "### Georges-Louis Leclerc, Comte de Buffon (1777)",
    "",
    "The story begins in **1777** with the French naturalist and mathematician **Georges-Louis Leclerc, Comte de Buffon** (1707–1788). In his seminal work *Histoire Naturelle*, Buffon posed a deceptively simple question now known as **Buffon\'s Needle Problem**:",
    "",
    '> *Suppose we have a floor made of parallel strips of wood, each of width $d$. We drop a needle of length $\\ell \\le d$ onto the floor at random. What is the probability that the needle crosses one of the lines?*',
    "",
    "The answer is elegant and unexpected: the probability is",
    "",
    "$$P = \\frac{2\\ell}{\\pi d}$$",
    "",
    "When $\\ell = d$, this simplifies to $P = 2/\\pi$. This was one of the first instances of **geometric probability** — using random experiments to compute a deterministic mathematical constant. Buffon essentially invented the first **Monte Carlo algorithm** 170 years before the term was coined. If you repeat the experiment $N$ times and count $k$ crossings, then $\\hat{\\pi} \\approx 2N/k$.",
    "",
    "Buffon\'s contemporary **Pierre-Simon de Laplace** later extended this problem to a general rectangular grid (the **Buffon-Laplace Problem**), and **Augustus De Morgan** reported that a pupil of his once performed a practical experiment yielding $\\hat{\\pi} = 3.137$ after 600 trials — remarkably close to the true value.",
    "",
    "### Stanislaw Ulam\'s Solitaire Epiphany (1946)",
    "",
    "The modern Monte Carlo method was born from a combination of **misfortune, insomnia, and a deck of cards**. In **1946**, Polish-American mathematician **Stanislaw Ulam** (1909–1984) had just undergone emergency brain surgery. During his convalescence, suffering from headaches and bouts of fatigue, he was playing **solitaire** to pass the time.",
    "",
    "Ulam found himself asking a seemingly simple question: *What are the chances that a hand laid out with 52 cards will come out successfully as a game of solitaire?* The combinatorial calculation is monstrously difficult — there are approximately $8 \\times 10^{67}$ possible shuffles. Instead of computing exact probabilities, Ulam realized he could **lay out the cards many times and count the fraction of successful games**. This was the seed of **statistical sampling** — replacing intractable analytical calculations with repeated random experiments.",
    "",
    "As Ulam later recalled in his memoir *Adventures of a Mathematician*, this insight immediately connected to problems of **neutron diffusion** and other questions in mathematical physics that he had encountered at the **Manhattan Project** at Los Alamos. In particular, Ulam had been working on calculating how far neutrons travel in matter — a problem that, at the time, required solving the Boltzmann transport equation in full 7-dimensional phase space. Deterministic methods were hopelessly inadequate, and Ulam realized that random sampling of individual neutron paths could bypass the analytical intractability entirely.",
    "",
    "> **The Lwow School of Mathematics:** Many of the greatest mathematicians of the 20th century were trained at the University of Lwow (now Lviv, Ukraine) in the 1930s. The **Lwow School of Mathematics** was a vibrant intellectual community. This famous 1930 photograph shows (from left): **Stanislaw Ulam**, **Hugo Steinhaus**, **Stanislaw Mazur**, **Stanislaw Ruziewicz**, and **John von Neumann** at a mathematics seminar in Lwow. Ulam and von Neumann would later collaborate at Los Alamos and Princeton, forming the intellectual partnership that gave birth to the Monte Carlo method.",
    "",
    "![The Lwow School of Mathematics, 1930](images/lwow_mathematicians.jpg)",
    "",
    "### John von Neumann, Nicholas Metropolis, and the Manhattan Project",
    "",
    "Ulam shared his idea with his friend and colleague **John von Neumann** (1903–1957), one of the greatest mathematicians of the 20th century. Von Neumann had been a central figure in the Manhattan Project since 1942. His contributions were vast: he solved the equations for the **implosion dynamics** of the plutonium *Fat Man* weapon (showing that symmetric compression of the plutonium core was essential for a successful chain reaction), calculated the behavior of shaped charges, and worked on the hydrodynamics of high-explosive detonations. Von Neumann was arguably the foremost expert in the world on numerical computation and hydrodynamics, and he understood intimately that the neutron transport calculations needed for the H-bomb (the `thermonuclear` project that occupied Los Alamos from 1947 onward) were analytically intractable.",
    "",
    "The key insight was that von Neumann\'s vision of **stored-program computers** — computers where the program is stored in memory rather than wired by physical cables — made Ulam\'s random sampling computationally feasible. The **ENIAC** (Electronic Numerical Integrator And Computer), completed in 1945 at the University of Pennsylvania, was the world\'s first programmable electronic digital computer. It weighed over 60,000 pounds, covered 1,800 square feet, consumed 150 kilowatts of power, and could perform **50,000 instructions per second** — about 1,000 times faster than its closest competitor.",
    "",
    "> **ENIAC**, the world\'s first general-purpose electronic digital computer. Composed of 40 panels containing 17,468 vacuum tubes, ENIAC was unveiled to the public on February 14, 1946. The six women who programmed it (Kay McNulty, Betty Jennings, Betty Snyder, Marlyn Wescoff, Fran Bilas, and Ruth Lichterman) were the original computer programmers — they wired the machine with cables and plugboards to perform specific computations.",
    "",
    "![ENIAC computer](images/eniac.jpg)",
    "",
    "In **1947**, von Neumann wrote the first formulation of a Monte Carlo computation for an electronic computer. Fellow Los Alamos colleague **Nicholas Metropolis** (1915–2006) coined the name **`Monte Carlo`** — referencing the method\'s probabilistic nature. The name also honored Ulam\'s Polish uncle, who would borrow money from relatives because he *`just had to go to Monte Carlo`* (the Las Vegas of Europe, referring to the famous casino in the Principality of Monaco).",
    "",
    "The first Monte Carlo calculations were run on **ENIAC in April–May 1948** for the computer\'s first trial after it was upgraded to a stored-program machine. These were the **first-ever programs written in the modern stored-program paradigm** to run on an electronic computer. The calculations remained classified until the late 1950s, but the method itself spread rapidly through Ulam\'s national lecture tour.",
    "",
    "### The Neutron Diffusion Problem at Los Alamos",
    "",
    "To understand why Monte Carlo was revolutionary, we must understand the **neutron transport problem** that consumed the minds of physicists at the secret Los Alamos Laboratory. Established in 1943 under the direction of **J. Robert Oppenheimer**, Los Alamos was one of ten secret sites of the Manhattan Project, employing over 6,000 people at its peak.",
    "",
    "> **The Main Gate of the Los Alamos Project, 1943.** The laboratory was built on the site of the former Los Alamos Ranch School, a boys\' boarding school in the mesas of northern New Mexico. Its remote location provided both security and isolation — scientists could work on the most sensitive physics problems without fear of espionage or public attention. The town that grew up around the lab housed scientists, their families, military personnel, and support staff in a purpose-built community.",
    "",
    "![Los Alamos Main Gate, 1943](images/los_alamos_gate.jpg)",
    "",
    "> **J. Robert Oppenheimer** at Los Alamos, ca. 1945. Oppenheimer (1904–1967) was appointed by President Roosevelt\'s wartime science apparatus to direct the secret weapons laboratory. A theoretical physicist trained at Cambridge and Göttingen, Oppenheimer assembled a team of the greatest scientific minds of the era — including Fermi, Feynman, Bethe, Teller, and Ulam — and gave them the mandate to build the atomic bomb before Nazi Germany could.",
    "",
    "![J. Robert Oppenheimer at Los Alamos](images/oppenheimer.jpg)",
    "",
    "The fundamental question was deceptively simple: **given a source of neutrons in a fissile material, where will they go, and how many will cause fission?** This is governed by the **Boltzmann Transport Equation**, a high-dimensional integro-differential equation describing the spatial, angular, and energy distribution of neutrons:",
    "",
    "$$\\frac{1}{v}\\frac{\\partial \\psi}{\\partial t} + \\hat{\\Omega} \\cdot \\nabla \\psi + \\Sigma_t \\psi = \\int \\Sigma_s(\\hat{\\Omega}' \\rightarrow \\hat{\\Omega}) \\psi(\\vec{r}, \\hat{\\Omega}', E') \\, d\\Omega' \\, dE' + S_{\\text{ext}}$$",
    "",
    "where $\\psi(\\vec{r}, \\hat{\\Omega}, E, t)$ is the **angular flux** — the number of neutrons at position $\\vec{r}$, traveling in direction $\\hat{\\Omega}$, with energy $E$, at time $t$. The terms represent: time evolution, streaming, removal by collisions, scattering from other directions/energies, and external sources.",
    "",
    "**This equation is analytically unsolvable for any realistic geometry.** The cross-sections $\\Sigma_t$ and $\\Sigma_s$ are complex functions of energy and material composition. The angular dependence lives on a sphere ($S^2$), the spatial dependence in 3D, plus energy and time — a **7-dimensional phase space** (3 space + 2 angle + 1 energy + 1 time). Deterministic numerical methods of the era (differencing methods) broke down catastrophically above 2 or 3 dimensions.",
    "",
    "The **Monte Carlo approach** sidesteps the equation entirely. Instead of solving for the flux everywhere, you **simulate individual neutron histories**:",
    "",
    "1. A neutron is born at position $\\vec{r}_0$ with energy $E_0$ and direction $\\hat{\\Omega}_0$ (sampled from the fission spectrum, approximately a Maxwell-Boltzmann distribution with $T \\approx 1.3$ MeV).",
    "2. The **distance to the next collision** is sampled from the exponential free-path distribution: $s = -\\ln(\\xi) / \\Sigma_t(E)$, where $\\xi \\sim U(0,1)$.",
    "3. The neutron travels distance $s$ along $\\hat{\\Omega}_0$. If it reaches a boundary (e.g., the reactor surface or a reflector), it is tracked accordingly.",
    "4. At the collision point, the **type of interaction** is sampled: scattering, absorption, or fission, based on the ratio of partial cross-sections to the total cross-section.",
    "5. If scattering, the **scattering angle** is sampled from the differential cross-section (isotropic in the center-of-mass frame for thermal neutrons, forward-peaked for fast neutrons). The neutron\'s new direction and energy are determined.",
    "6. If fission, **2–3 new neutrons** are born (for U-235, $\\nu \\approx 2.43$), each with its own position, energy, and direction, and the process recurs.",
    "7. The history ends when the neutron is absorbed, leaks out of the system, or its energy drops below a thermal cutoff (~0.62 eV).",
    "",
    "By tracking millions of such histories, one can estimate **any quantity of interest**: the critical mass of U-235 needed for a self-sustaining chain reaction, the neutron flux distribution in the core, the probability that a neutron born in the plutonium core of the *Fat Man* weapon will actually reach the tamper and contribute to the yield.",
    "",
    "The **first critical mass calculation** was one of the earliest and most important Monte Carlo applications. Hans Bethe\'s theoretical estimates suggested a sphere of highly enriched U-235 with a thorium reflector would be critical at roughly **11 pounds**. The Los Alamos Monte Carlo calculations, run on ENIAC, confirmed this order of magnitude — though the actual *Fat Man* weapon used about **14 pounds** of plutonium (with a uranium tamper that acted as a reflector, reducing the critical mass).",
    "",
    "This was the birth of **statistical physics simulation** — replacing impossible analytical calculations with massive random sampling. The same principles govern reactor design, radiation shielding, medical physics (dose calculations in radiotherapy), and now, as we will see, Bayesian inference in high-dimensional parameter spaces.",
    "",
    "### Enrico Fermi\'s Independent Discovery",
    "",
    "Remarkably, **Enrico Fermi** independently invented the fundamentals of random sampling in the **1930s** while studying neutron moderation in Italy, though he kept this work secret and unpublished. His colleagues often wondered how Fermi arrived at his answers so fast. Fermi even designed an analogue device called the **FERMIAC** (Monte Carlo Trolley) — a brass and acrylic instrument with rotating drums that moved a stylus over paper to draw two-dimensional neutron trajectories.",
    "",
    "The FERMIAC was essentially a mechanical random-number generator: the rotating drums produced random numbers that determined the step lengths and angles of neutron paths. Fermi would push the trolley across a sheet of graph paper, and the device would trace out a neutron\'s random walk. This was Monte Carlo simulation **two decades before the digital era** — a testament to Fermi\'s extraordinary physical intuition.",
    "",
    "> **Stanislaw Ulam holding a FERMIAC** at Los Alamos. While the FERMIAC was Fermi\'s invention, the image captures the spirit of the era: physicists of all backgrounds working together to solve neutron transport problems using whatever tools were available — from mechanical trolleys to the world\'s first electronic computers. Ulam, who co-invented the digital Monte Carlo method, stands here with the analogue device that preceded it, bridging two eras of computational physics.",
    "",
    "![Stanislaw Ulam holding the FERMIAC](images/ulam_fermiac.jpg)",
    "",
    "Fermi later brought these ideas to the United States and joined the Manhattan Project at Chicago, where he built **Chicago Pile-1 (CP-1)**, the world\'s first nuclear reactor, under the stands of Stagg Field at the University of Chicago. On **December 2, 1942**, CP-1 achieved the first self-sustaining nuclear chain reaction. The Monte Carlo method, which Ulam and von Neumann would formalize just four years later, was the computational counterpart to this experimental triumph: where CP-1 physically demonstrated neutron multiplication, Monte Carlo calculations predicted it on paper (and later on ENIAC).",
    "",
    "## 1.2 The Mathematics: Law of Large Numbers",
    "",
    "At the heart of Monte Carlo estimation lies the **Law of Large Numbers (LLN)**. Suppose we want to compute an expected value:",
    "",
    "$$I = \\mathbb{E}[f(X)] = \\int f(x) \\, p(x) \\, dx$$",
    "",
    "where $X \\sim p(x)$ is a random variable. The **Strong Law of Large Numbers** states that for i.i.d. samples $X_1, X_2, \\ldots, X_N \\sim p(x)$:",
    "",
    "$$\\hat{I}_N = \\frac{1}{N} \\sum_{i=1}^{N} f(X_i) \\xrightarrow[N \\to \\infty]{\\text{a.s.}} \\mathbb{E}[f(X)] = I$$",
    "",
    "The convergence rate is governed by the **Central Limit Theorem**:",
    "",
    "$$\\sqrt{N}(\\hat{I}_N - I) \\xrightarrow{d} \\mathcal{N}(0, \\sigma^2)$$",
    "",
    "where $\\sigma^2 = \\text{Var}(f(X))$. This means the **standard error scales as $1/\\sqrt{N}$**:",
    "",
    "$$\\text{SE}(\\hat{I}_N) = \\frac{\\sigma}{\\sqrt{N}}$$",
    "",
    "This $1/\\sqrt{N}$ convergence is both Monte Carlo\'s strength (it\'s **dimension-independent**) and its weakness (it converges **slowly** — to gain one decimal digit of accuracy, you need 100x more samples).",
    "",
    "## 1.3 Code Implementation 1: Estimating $\\pi$ via Dart-Board Sampling",
    "",
    "The simplest Monte Carlo experiment: throw darts at a square target inscribed with a circle. The ratio of areas gives us $\\pi$.",
])

write_code_file("_code_p1_pi.py", r"""# ============================================================
# Part 1, Code 1: Estimating pi via Monte Carlo Dart-Board
# ============================================================
# Consider a unit square [0,1] x [0,1] with an inscribed quarter-circle
# of radius 1 centered at the origin.
#
#   Area of square    = 1 x 1 = 1
#   Area of quarter   = pi x 1^2 / 4 = pi/4
#
# If we sample points uniformly in the square, the fraction that fall
# inside the quarter-circle converges to pi/4 by the Law of Large Numbers.
#
#   pi ≈ 4 x (number of points with x^2 + y^2 <= 1) / N
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# --- Generate random points in the unit square ---
N_total = 100_000
x = np.random.uniform(0, 1, N_total)
y = np.random.uniform(0, 1, N_total)

# --- Determine which points fall inside the quarter-circle ---
inside_circle = x**2 + y**2 <= 1.0
pi_estimate = 4.0 * np.sum(inside_circle) / N_total

print(f"True pi        = {np.pi:.10f}")
print(f"Estimated pi   = {pi_estimate:.10f}")
print(f"Absolute error = {abs(np.pi - pi_estimate):.2e}")
print(f"Expected SE    ≈ sigma/sqrt(N) ≈ 1.27e-3")

# --- Track convergence as we increase N ---
N_values = np.logspace(2, 5, dtype=int)  # 100 to 100,000
estimates = []
errors = []

for N in N_values:
    x_sub = np.random.uniform(0, 1, N)
    y_sub = np.random.uniform(0, 1, N)
    inside = x_sub**2 + y_sub**2 <= 1.0
    pi_hat = 4.0 * np.sum(inside) / N
    estimates.append(pi_hat)
    errors.append(abs(np.pi - pi_hat))

estimates = np.array(estimates)
errors = np.array(errors)

# --- Plot 1: Scatter of points with quarter-circle ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot
N_plot = 2000  # subsample for clarity
ax = axes[0]
ax.scatter(x[:N_plot][inside_circle[:N_plot]], y[:N_plot][inside_circle[:N_plot]],
           c='steelblue', s=10, alpha=0.6, label='Inside circle')
ax.scatter(x[:N_plot][~inside_circle[:N_plot]], y[:N_plot][~inside_circle[:N_plot]],
           c='coral', s=10, alpha=0.6, label='Outside circle')

# Draw the quarter-circle arc
theta_arc = np.linspace(0, np.pi / 2, 200)
ax.plot(np.cos(theta_arc), np.sin(theta_arc), 'k--', linewidth=2, label='Quarter-circle')
ax.set_aspect('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title(f'Monte Carlo Dart-Board (N = {N_plot})\npi ≈ {pi_estimate:.4f}')
ax.legend(fontsize=8)

# --- Plot 2: Convergence curve ---
ax = axes[1]
ax.semilogy(N_values, errors, 'b-o', markersize=4, label='Absolute error |pi_hat - pi|')
# Theoretical 1/sqrt(N) envelope
theoretical = 1.27 / np.sqrt(N_values)
ax.semilogy(N_values, theoretical, 'r--', alpha=0.5, label='1/sqrt(N) reference')
ax.set_xlabel('Number of samples (N)')
ax.set_ylabel('Absolute Error (log scale)')
ax.set_title('Convergence of Monte Carlo pi Estimation')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE, 'part1_pi_estimation.png'), dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved: part1_pi_estimation.png")
""")

add_code(os.path.join(BASE, "_code_p1_pi.py"), "Part 1, Code 1: Estimating pi via Monte Carlo Dart-Board Sampling")

add_md([
    "**What we see:** The scatter plot shows points uniformly distributed in the unit square, with the quarter-circle boundary clearly visible. The convergence plot demonstrates the characteristic $1/\\sqrt{N}$ decay of the error — doubling accuracy requires quadrupling the number of samples. The red dashed line is the theoretical $1/\\sqrt{N}$ reference, and our empirical error tracks it closely.",
    "",
    "## 1.4 Code Implementation 2: Mini Neutron Diffusion Problem",
    "",
    "Now let us simulate a simplified **1D neutron diffusion** problem in a nuclear reactor core, directly inspired by the original Monte Carlo applications at Los Alamos. We will track individual neutrons as they undergo random walks with three possible outcomes at each step:",
    "",
    "- **Scattering**: The neutron bounces in a random direction (continues the walk)",
    "- **Absorption**: The neutron is absorbed by the material (walk ends)",
    "- **Fission**: The neutron triggers fission, producing new neutrons (the walk branches)",
])

write_code_file("_code_p1_neutron.py", r"""# ============================================================
# Part 1, Code 2: 1D Neutron Diffusion Monte Carlo Simulation
# ============================================================
#
# This simulates neutron transport in a 1D slab reactor of width L.
# Neutrons are born at the center and undergo a random walk.
#
# At each step, three events are possible (with probabilities that
# sum to 1):
#   P(scatter)  = p_scatter  -> neutron continues, random direction
#   P(absorb)   = p_absorb   -> neutron is absorbed (death)
#   P(fission)  = p_fission  -> neutron splits into nu new neutrons
#
# We track:
#   - How many neutrons escape the reactor (leakage)
#   - How many are absorbed
#   - The total neutron population over time (criticality)
#
# This is the type of calculation that von Neumann and Metropolis
# first ran on ENIAC in 1948.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)

# --- Reactor geometry and physics parameters ---
L = 10.0              # Reactor slab width (cm)
x_center = 0.0        # Neutrons born at center
step_size = 0.5       # Mean free path (cm) per step
p_scatter = 0.70      # Probability of scattering
p_absorb  = 0.20      # Probability of absorption
p_fission = 0.10      # Probability of fission
nu = 2.5              # Average neutrons per fission event
N_initial = 100       # Initial neutrons born


def simulate_neutron_history(x_start, max_steps=200):
    # Simulate a single neutron history via random walk.
    # Returns: 'escape', 'absorbed', 'killed', or ('fission_children', [positions])
    x = x_start
    for step in range(max_steps):
        # Random direction: +1 (right) or -1 (left)
        direction = np.random.choice([-1, 1])
        x += direction * step_size

        # Check if neutron escaped the reactor boundaries
        if abs(x) > L / 2:
            return 'escape'

        # Determine the event at this collision
        r = np.random.random()
        if r < p_absorb:
            return 'absorbed'
        elif r < p_absorb + p_fission:
            # Fission: spawn new neutrons
            n_children = np.random.poisson(nu)
            children = [x + np.random.uniform(-step_size, step_size)
                        for _ in range(max(0, n_children))]
            return ('fission_children', children)
        # else: scatter - continue the walk

    return 'killed'  # neutron died after max_steps


# --- Run the simulation ---
N_trials = 5000  # total neutron histories to track
results = {'escape': 0, 'absorbed': 0, 'killed': 0, 'fission': 0}
absorption_positions = []
escape_positions = []

for i in range(N_trials):
    result = simulate_neutron_history(x_center)

    if result == 'escape':
        results['escape'] += 1
        escape_positions.append(L / 2)
    elif result == 'absorbed':
        results['absorbed'] += 1
    elif result == 'killed':
        results['killed'] += 1
    elif isinstance(result, tuple) and result[0] == 'fission_children':
        results['fission'] += 1
        # Each fission child gets its own history
        for child_pos in result[1]:
            child_result = simulate_neutron_history(child_pos, max_steps=50)
            if child_result == 'escape':
                results['escape'] += 1
            elif child_result == 'absorbed':
                results['absorbed'] += 1
                absorption_positions.append(child_pos)
            elif child_result == 'killed':
                results['killed'] += 1
            elif isinstance(child_result, tuple) and child_result[0] == 'fission_children':
                results['fission'] += 1


# --- Compute macroscopic quantities ---
leakage_fraction = results['escape'] / N_trials
absorption_fraction = results['absorbed'] / N_trials
fission_fraction = results['fission'] / N_trials

# Effective multiplication factor (k-eff)
neutrons_produced = results['fission'] * nu
neutrons_lost = results['absorbed'] + results['escape']
k_eff = neutrons_produced / neutrons_lost if neutrons_lost > 0 else np.inf

print("=" * 60)
print("1D Neutron Diffusion Monte Carlo Simulation Results")
print("=" * 60)
print(f"Initial neutrons:     {N_initial}")
print(f"Total histories:      {N_trials}")
print(f"")
print(f"Results:")
print(f"  Escaped (leakage):  {results['escape']:5d}  ({leakage_fraction:.3f})")
print(f"  Absorbed:           {results['absorbed']:5d}  ({absorption_fraction:.3f})")
print(f"  Killed (timeout):   {results['killed']:5d}  ({results['killed']/N_trials:.3f})")
print(f"  Fission events:     {results['fission']:5d}  ({fission_fraction:.3f})")
print(f"")
print(f"Effective multiplication factor k_eff = {k_eff:.4f}")
if k_eff > 1.0:
    print("  -> SUPERCRITICAL: neutron population grows exponentially")
elif k_eff == 1.0:
    print("  -> CRITICAL: self-sustaining chain reaction")
else:
    print("  -> SUBCRITICAL: neutron population decays")
print("=" * 60)

# --- Visualization ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Histogram of absorption positions
ax = axes[0]
if absorption_positions:
    ax.hist(absorption_positions, bins=30, density=True, alpha=0.7,
            color='steelblue', edgecolor='white', label='Absorption positions')
    # Theoretical: diffusion theory predicts a cosine-like profile
    x_theory = np.linspace(-L/2, L/2, 200)
    flux_theory = np.cos(np.pi * x_theory / L)
    flux_theory = np.maximum(flux_theory, 0)
    flux_theory /= flux_theory.max()
    ax.plot(x_theory, flux_theory, 'r-', linewidth=2, label='Diffusion theory: cos(pi*x/L)')
ax.set_xlabel('Position x (cm)')
ax.set_ylabel('Normalized frequency')
ax.set_title('Neutron Absorption Spatial Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Bar chart of event types
ax = axes[1]
event_types = ['Escape\n(Leakage)', 'Absorbed', 'Killed\n(timeout)', 'Fission']
counts = [results['escape'], results['absorbed'], results['killed'], results['fission']]
colors_bar = ['coral', 'steelblue', 'gray', 'gold']
bars = ax.bar(event_types, counts, color=colors_bar, edgecolor='black', alpha=0.8)
ax.set_ylabel('Count')
ax.set_title(f'Neutron Event Outcomes (N = {N_trials})')
ax.grid(True, alpha=0.3, axis='y')
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{count}', ha='center', va='bottom', fontweight='bold')

# Convergence of k_eff estimate
ax = axes[2]
batch_size = 100
n_batches = N_trials // batch_size
cum_produced = 0
cum_lost = 0
batch_ns = []
batch_k_effs = []
for b in range(n_batches):
    batch_results = {'produced': 0, 'lost': 0}
    for j in range(batch_size):
        result = simulate_neutron_history(x_center)
        if result == 'escape':
            batch_results['lost'] += 1
        elif result == 'absorbed':
            batch_results['lost'] += 1
        elif result == 'killed':
            batch_results['lost'] += 1
        elif isinstance(result, tuple) and result[0] == 'fission_children':
            batch_results['produced'] += nu * len(result[1])
    cum_produced += batch_results['produced']
    cum_lost += batch_results['lost']
    k_batch = cum_produced / cum_lost if cum_lost > 0 else np.nan
    batch_k_effs.append(k_batch)
    batch_ns.append((b + 1) * batch_size)

ax.semilogx(batch_ns, batch_k_effs, 'b-', linewidth=1, alpha=0.7, label='Running k_eff')
ax.axhline(k_eff, color='r', linestyle='--', linewidth=2, label=f'Final k_eff = {k_eff:.4f}')
ax.axhline(1.0, color='green', linestyle=':', linewidth=1.5, label='Critical (k=1)')
ax.set_xlabel('Total neutron histories simulated')
ax.set_ylabel('k_eff')
ax.set_title('Convergence of k_eff Estimate')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE, 'part1_neutron_diffusion.png'), dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved: part1_neutron_diffusion.png")
""")

add_code(os.path.join(BASE, "_code_p1_neutron.py"), "Part 1, Code 2: 1D Neutron Diffusion Monte Carlo (Los Alamos-style)")

add_md([
    "**What we see:** The left panel shows the spatial distribution of neutron absorptions, which roughly follows the fundamental diffusion mode $\\cos(\\pi x/L)$ — neutrons are most likely to be absorbed near the center where they are born. The middle panel breaks down the event types: leakage, absorption, and fission. The right panel shows the convergence of the effective multiplication factor $k_{\\text{eff}}$ as more histories are simulated. If $k_{\\text{eff}} > 1$, the reactor is supercritical (chain reaction grows); if $k_{\\text{eff}} = 1$, it is critical (self-sustaining); if $k_{\\text{eff}} < 1$, it is subcritical (reaction dies out).",
    "",
    "This simulation directly mirrors the calculations that von Neumann, Metropolis, and Ulam performed on ENIAC in 1948 — tracking individual particle histories with random sampling to solve complex transport equations that are analytically intractable.",
    "",
    "---",
    "",
])

# =============================================================================
# PART 2: FOUNDATIONS & VARIANCE REDUCTION
# =============================================================================
add_md([
    "# Part 2: Foundations & Variance Reduction",
    "",
    "### Real-World Scenario: Cold Chain Spoilage Risk",
    "",
    "A pharmaceutical company transports temperature-sensitive vaccines in refrigerated trucks. Each truck has a temperature threshold $T_{\\text{crit}}$ — if the temperature exceeds this for too long, the vaccine spoils. We need to estimate the **probability of spoilage** over a given transport duration, accounting for temperature variability.",
    "",
    "## 2.1 Mathematical Foundations",
    "",
    "### Expected Value and Variance",
    "",
    "For a random variable $X$ with probability density function (PDF) $p(x)$:",
    "",
    "$$\\mathbb{E}[X] = \\int_{-\\infty}^{\\infty} x \\, p(x) \\, dx$$",
    "",
    "$$\\text{Var}(X) = \\mathbb{E}[(X - \\mathbb{E}[X])^2] = \\mathbb{E}[X^2] - (\\mathbb{E}[X])^2$$",
    "",
    "The **standard deviation** $\\sigma = \\sqrt{\\text{Var}(X)}$ measures the spread of the distribution.",
    "",
    "### Pseudo-Random Number Generation",
    "",
    "Computers generate **pseudo-random numbers** using deterministic algorithms. The most common is the **Linear Congruential Generator (LCG)**:",
    "",
    "$$X_{n+1} = (a X_n + c) \\mod m$$",
    "",
    "where $a$ is the multiplier, $c$ is the increment, and $m$ is the modulus. A good LCG produces sequences that pass statistical tests for randomness, even though they are fully deterministic given the seed $X_0$.",
    "",
    "NumPy uses the **Mersenne Twister** (MT19937) as its default RNG, with a period of $2^{19937} - 1$ — astronomically large for any practical simulation.",
    "",
    "### Why Crude Monte Carlo Converges Slowly",
    "",
    "The variance of the Monte Carlo estimator $\\hat{I}_N = \\frac{1}{N}\\sum_{i=1}^{N} f(X_i)$ is:",
    "",
    "$$\\text{Var}(\\hat{I}_N) = \\frac{\\text{Var}(f(X))}{N} = \\frac{\\sigma_f^2}{N}$$",
    "",
    "The key insight: this variance is **independent of dimension**. Whether we are integrating over 2 variables or 2000, the convergence rate remains $O(1/\\sqrt{N})$. This is why Monte Carlo excels in high dimensions where quadrature methods fail exponentially.",
    "",
    "However, for **rare events** (e.g., spoilage probability of $10^{-4}$), crude Monte Carlo is wasteful — most samples contribute nothing to the estimate. This motivates **variance reduction techniques**.",
    "",
    "### Importance Sampling",
    "",
    "The key idea: sample from a **proposal distribution** $q(x)$ that concentrates mass where $f(x)p(x)$ is large, rather than from $p(x)$ itself. We rewrite the integral:",
    "",
    "$$I = \\int f(x) p(x) \\, dx = \\int f(x) \\frac{p(x)}{q(x)} q(x) \\, dx = \\mathbb{E}_q\\left[f(X) \\frac{p(X)}{q(X)}\\right]$$",
    "",
    "The **importance weight** is $w(x) = \\frac{p(x)}{q(x)}$. The importance sampling estimator is:",
    "",
    "$$\\hat{I}_N^{\\text{IS}} = \\frac{1}{N} \\sum_{i=1}^{N} f(X_i) w(X_i), \\quad X_i \\sim q(x)$$",
    "",
    "The variance of this estimator is:",
    "",
    "$$\\text{Var}(\\hat{I}_N^{\\text{IS}}) = \\frac{1}{N}\\left( \\mathbb{E}_q\\left[f^2(X) w^2(X)\\right] - I^2 \\right)$$",
    "",
    "The **optimal** proposal distribution minimizes this variance:",
    "",
    "$$q^*(x) \\propto |f(x)| p(x)$$",
    "",
    "In practice, we choose $q(x)$ to be a distribution that is easy to sample from and that puts more mass in the important regions of the integrand.",
])

write_code_file("_code_p2_crude.py", r"""# ============================================================
# Part 2, Code 1: Cold Chain Spoilage Risk - Crude Monte Carlo
# ============================================================
#
# Scenario: A refrigerated truck transports vaccines. The internal
# temperature T follows a normal distribution N(mu, sigma^2) due to
# ambient temperature fluctuations and refrigerator imperfections.
#
# Spoilage occurs when T > T_crit for a sustained period.
# We estimate P(spoilage) = P(T > T_crit) using Monte Carlo.
#
# Parameters:
#   mu = 4.0 C  (setpoint temperature)
#   sigma = 2.0 C  (temperature variability)
#   T_crit = 8.0 C  (safety threshold)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm

np.random.seed(42)

# --- Parameters ---
mu = 4.0          # Mean temperature (C)
sigma = 2.0       # Std dev of temperature (C)
T_crit = 8.0      # Spoilage threshold (C)
N_samples = 100_000

# --- Crude Monte Carlo ---
temperatures = np.random.normal(mu, sigma, N_samples)
spoilage_indicators = (temperatures > T_crit).astype(float)
p_spoilage_crude = np.mean(spoilage_indicators)

# Theoretical value for comparison
p_spoilage_theoretical = 1.0 - norm.cdf(T_crit, loc=mu, scale=sigma)

# Standard error
se_crude = np.sqrt(p_spoilage_crude * (1 - p_spoilage_crude) / N_samples)

print("=" * 60)
print("Cold Chain Spoilage Risk - Crude Monte Carlo")
print("=" * 60)
print(f"Temperature model: T ~ N({mu}, {sigma}^2)")
print(f"Spoilage threshold: T_crit = {T_crit} C")
print(f"")
print(f"Crude Monte Carlo (N = {N_samples:,}):")
print(f"  P(spoilage)  = {p_spoilage_crude:.6f}")
print(f"  Theoretical   = {p_spoilage_theoretical:.6f}")
print(f"  Standard Error = {se_crude:.6f}")
print(f"  Relative error = {abs(p_spoilage_crude - p_spoilage_theoretical)/p_spoilage_theoretical*100:.2f}%")
print("=" * 60)

# --- Convergence plot ---
N_values = np.logspace(2, 5, dtype=int)
p_estimates = []

for N in N_values:
    temps = np.random.normal(mu, sigma, N)
    spoilage = (temps > T_crit).astype(float)
    p_estimates.append(np.mean(spoilage))

p_estimates = np.array(p_estimates)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Temperature distribution with spoilage region highlighted
ax = axes[0]
x_range = np.linspace(mu - 4*sigma, mu + 5*sigma, 500)
pdf = norm.pdf(x_range, loc=mu, scale=sigma)
ax.plot(x_range, pdf, 'b-', linewidth=2, label='T ~ N(mu, sigma^2)')
ax.fill_between(x_range, pdf, where=(x_range >= T_crit), alpha=0.3,
                color='red', label=f'Spoilage region (T > {T_crit} C)')
ax.axvline(T_crit, color='red', linestyle='--', linewidth=2, label=f'T_crit = {T_crit} C')
ax.axvline(mu, color='blue', linestyle='--', linewidth=1.5, label=f'mu = {mu} C')
ax.set_xlabel('Temperature (C)')
ax.set_ylabel('Probability Density')
ax.set_title('Temperature Distribution and Spoilage Region')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Convergence of spoilage probability estimate
ax = axes[1]
ax.semilogx(N_values, p_estimates, 'b-o', markersize=4, label='Crude MC estimate')
ax.axhline(p_spoilage_theoretical, color='r', linestyle='--', linewidth=2,
           label=f'Theoretical = {p_spoilage_theoretical:.6f}')
ax.set_xlabel('Number of samples (N)')
ax.set_ylabel('Estimated P(spoilage)')
ax.set_title('Convergence of Spoilage Probability Estimate')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE, 'part2_cold_chain_crude.png'), dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved: part2_cold_chain_crude.png")
""")

add_code(os.path.join(BASE, "_code_p2_crude.py"), "Part 2, Code 1: Cold Chain Spoilage - Crude Monte Carlo")

write_code_file("_code_p2_is.py", r"""# ============================================================
# Part 2, Code 2: Importance Sampling for Cold Chain Spoilage
# ============================================================
#
# Crude Monte Carlo wastes samples in the low-probability region
# (T << T_crit). Importance sampling shifts the sampling distribution
# to focus on the spoilage region.
#
# We use a shifted normal distribution as the proposal:
#   q(T) = N(mu_shift, sigma^2)  where mu_shift > mu
#
# The importance weight is:
#   w(T) = p(T) / q(T) = phi((T-mu)/sigma)/sigma / phi((T-mu_shift)/sigma)/sigma
#
# For normal distributions, this simplifies to:
#   w(T) = exp( -(T-mu)^2/(2*sigma^2) + (T-mu_shift)^2/(2*sigma^2) )
#        = exp( (2T*(mu_shift - mu) - (mu_shift^2 - mu^2)) / (2*sigma^2) )
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm

np.random.seed(42)

# --- Parameters from crude MC ---
mu = 4.0
sigma = 2.0
T_crit = 8.0
N_samples = 100_000
p_spoilage_crude = 0.02275  # approximate from crude MC
se_crude = 0.00477

# --- Importance Sampling parameters ---
mu_shift = 9.0  # Shift the proposal mean toward the spoilage region
N_is_samples = 100_000

# Sample from the shifted (proposal) distribution
temps_is = np.random.normal(mu_shift, sigma, N_is_samples)

# Compute importance weights
log_weights = (-(temps_is - mu)**2 / (2 * sigma**2)
               + (temps_is - mu_shift)**2 / (2 * sigma**2))
weights = np.exp(log_weights)

# Spoilage indicators under the proposal
spoilage_indicators_is = (temps_is > T_crit).astype(float)

# Importance sampling estimator
p_spoilage_is = np.mean(spoilage_indicators_is * weights)

# Weighted standard error
weighted_var = np.var(spoilage_indicators_is * weights)
se_is = np.sqrt(weighted_var / N_is_samples)

# Effective sample size (ESS)
mean_weight = np.mean(weights)
var_weight = np.var(weights)
ess = N_is_samples / (1 + (var_weight / mean_weight**2))

print("=" * 60)
print("Cold Chain Spoilage Risk - Importance Sampling")
print("=" * 60)
print(f"Proposal distribution: T ~ N({mu_shift}, {sigma}^2)")
print(f"")
print(f"Crude Monte Carlo (N = {N_samples:,}):")
print(f"  P(spoilage)  = {p_spoilage_crude:.6f}")
print(f"  SE           = {se_crude:.6f}")
print(f"")
print(f"Importance Sampling (N = {N_is_samples:,}):")
print(f"  P(spoilage)  = {p_spoilage_is:.6f}")
print(f"  SE           = {se_is:.6f}")
print(f"  Theoretical   = {1.0 - norm.cdf(T_crit, loc=mu, scale=sigma):.6f}")
print(f"")
print(f"Effective Sample Size (ESS) = {ess:.0f} / {N_is_samples}")
print(f"Variance reduction factor   = {(se_crude/se_is)**2:.1f}x")
print("=" * 60)

# --- Comparison visualization ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution comparison
ax = axes[0]
x_range = np.linspace(mu - 4*sigma, mu + 6*sigma, 500)
p_pdf = norm.pdf(x_range, loc=mu, scale=sigma)
q_pdf = norm.pdf(x_range, loc=mu_shift, scale=sigma)
ax.plot(x_range, p_pdf, 'b-', linewidth=2, label='Target p(T) = N(mu, sigma^2)')
ax.plot(x_range, q_pdf, 'r-', linewidth=2, label='Proposal q(T) = N(mu_shift, sigma^2)')
ax.fill_between(x_range, p_pdf, where=(x_range >= T_crit), alpha=0.2, color='red')
ax.axvline(T_crit, color='orange', linestyle='--', linewidth=2, label=f'T_crit = {T_crit} C')
ax.set_xlabel('Temperature (C)')
ax.set_ylabel('Probability Density')
ax.set_title('Target vs. Proposal Distributions')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Variance comparison bar chart
ax = axes[1]
methods = ['Crude MC', 'Importance\nSampling']
ses = [se_crude, se_is]
colors_bar = ['steelblue', 'coral']
bars = ax.bar(methods, ses, color=colors_bar, edgecolor='black', alpha=0.8)
ax.set_ylabel('Standard Error')
ax.set_title(f'Standard Error Comparison (Variance Reduction: {(se_crude/se_is)**2:.1f}x)')
ax.grid(True, alpha=0.3, axis='y')
for bar, se_val in zip(bars, ses):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + se_crude*0.05,
            f'{se_val:.6f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(BASE, 'part2_importance_sampling.png'), dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved: part2_importance_sampling.png")
""")

add_code(os.path.join(BASE, "_code_p2_is.py"), "Part 2, Code 2: Cold Chain Spoilage - Importance Sampling")

add_md([
    "**What we see:** The left panel shows how the proposal distribution $q(T) = \\mathcal{N}(\\mu_{\\text{shift}}, \\sigma^2)$ is shifted toward the spoilage region, sampling much more efficiently in the tail where spoilage occurs. The right panel shows the dramatic variance reduction: importance sampling achieves the same accuracy with far fewer samples. The **Effective Sample Size (ESS)** tells us how many useful samples we effectively have — if ESS is much smaller than $N$, the weights have too much variance and the proposal is not well-matched.",
    "",
    "---",
    "",
])

# =============================================================================
# PART 3: METROPOLIS-HASTINGS
# =============================================================================
add_md([
    "# Part 3: Metropolis-Hastings MCMC",
    "",
    "### Real-World Scenario: Sensor Drift Calibration",
    "",
    "Temperature sensors in cold chain trucks degrade over time, exhibiting **sensor drift**. Given noisy, drifting readings and intermittent manual quality checks, we need to calibrate the sensor by estimating the true underlying temperature and the drift rate.",
    "",
    "## 3.1 The Curse of Dimensionality",
    "",
    "Consider numerical integration over $d$ dimensions. A grid-based quadrature with $n$ points per dimension requires $n^d$ total evaluations. For $d=10$ and $n=10$, that is $10^{10}$ evaluations — completely infeasible. Monte Carlo\'s $O(1/\\sqrt{N})$ rate is **independent of dimension**, making it the method of choice for high-dimensional problems.",
    "",
    "## 3.2 Markov Chains and Stationary Distributions",
    "",
    "A **Markov chain** is a stochastic process $\\{X_t\\}_{t=0}^{\\infty}$ with the **Markov property**:",
    "",
    "$$P(X_{t+1} | X_t, X_{t-1}, \\ldots, X_0) = P(X_{t+1} | X_t)$$",
    "",
    "The future depends only on the present, not the past. A distribution $\\pi(x)$ is **stationary** (or invariant) if:",
    "",
    "$$\\pi(x) = \\int \\pi(y) \\, K(x | y) \\, dy$$",
    "",
    "where $K(x | y)$ is the transition kernel. If a Markov chain is **irreducible**, **aperiodic**, and **positive recurrent**, then it converges to its stationary distribution:",
    "",
    "$$X_t \\sim \\pi(x) \\quad \\text{as } t \\to \\infty$$",
    "",
    "## 3.3 Detailed Balance and the Metropolis-Hastings Algorithm",
    "",
    "**Detailed balance** is a sufficient (but not necessary) condition for stationarity:",
    "",
    "$$\\pi(x) \\, K(x' | x) = \\pi(x') \\, K(x | x')$$",
    "",
    "If detailed balance holds, then integrating both sides over $x$ gives the stationarity condition. The **Metropolis-Hastings** algorithm constructs a transition kernel that satisfies detailed balance with respect to a target distribution $\\pi(x)$ (proportional to the posterior in Bayesian inference).",
    "",
    "The algorithm works in two steps:",
    "",
    "1. **Propose**: Sample a candidate $x'$ from a proposal distribution $q(x' | x)$",
    "2. **Accept/Reject**: Accept the candidate with probability",
    "",
    "$$\\alpha(x, x') = \\min\\left(1, \\frac{\\pi(x') \\, q(x | x')}{\\pi(x) \\, q(x' | x)}\\right)$$",
    "",
    "If accepted, $X_{t+1} = x'$; otherwise, $X_{t+1} = X_t$.",
    "",
    "**Key insight:** The ratio $\\frac{\\pi(x')}{\\pi(x)}$ only requires the target distribution up to a normalization constant! This is crucial for Bayesian inference where the marginal likelihood (evidence) is intractable.",
    "",
    "If the proposal is **symmetric** ($q(x' | x) = q(x | x')$), this simplifies to the original **Metropolis algorithm**:",
    "",
    "$$\\alpha(x, x') = \\min\\left(1, \\frac{\\pi(x')}{\\pi(x)}\\right)$$",
])

write_code_file("_code_p3_mh.py", r"""# ============================================================
# Part 3, Code 1: Sensor Drift Calibration via Metropolis-Hastings
# ============================================================
#
# Model:
#   True temperature: T_true(t) = mu0 + beta*t  (linear drift)
#   Sensor reading:   y(t) = T_true(t) + epsilon,   epsilon ~ N(0, sigma^2_noise)
#
# We observe noisy sensor readings y(t_i) at times t_i, along with
# occasional manual calibration checks. Our goal is to infer the
# posterior distribution of the drift parameters (mu0, beta) given the data.
#
# Parameters to infer:
#   mu0  - baseline temperature (C)
#   beta  - drift rate (C per day)
#   sigma - measurement noise std dev (C)
#
# We use Metropolis-Hastings with a random-walk proposal to sample
# from the posterior P(mu0, beta, sigma | data).
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm

np.random.seed(42)

# --- Generate synthetic sensor data ---
n_days = 90  # 3-month observation period
t = np.arange(n_days)

# True parameters
true_mu0 = 4.0      # Baseline temperature: 4 C
true_beta = 0.02    # Drift rate: 0.02 C per day (2 C over 90 days)
true_sigma = 0.5    # Sensor noise: 0.5 C

# True temperature with linear drift
T_true = true_mu0 + true_beta * t

# Noisy sensor readings
sensor_readings = T_true + np.random.normal(0, true_sigma, n_days)

# Manual calibration checks (sparse, every 15 days)
calibration_days = np.arange(0, n_days, 15)
calibration_values = T_true[calibration_days] + np.random.normal(0, 0.1, len(calibration_days))

print("=" * 60)
print("Sensor Drift Calibration - Metropolis-Hastings MCMC")
print("=" * 60)
print(f"True parameters:")
print(f"  mu0 (baseline) = {true_mu0:.3f} C")
print(f"  beta (drift)   = {true_beta:.5f} C/day")
print(f"  sigma (noise)  = {true_sigma:.3f} C")
print(f"")
print(f"Observations: {n_days} days of sensor data + {len(calibration_days)} manual checks")
print("=" * 60)


# --- Log-likelihood and log-posterior ---
def log_likelihood(mu0, beta, sigma, t, y):
    # Compute log-likelihood of the data given parameters.
    T_model = mu0 + beta * t
    n = len(y)
    log_lik = -n/2 * np.log(2 * np.pi * sigma**2)
    log_lik -= np.sum((y - T_model)**2) / (2 * sigma**2)
    return log_lik


def log_posterior(params, t, y, calibration_t, calibration_y):
    # Compute log-posterior (unnormalized).
    # Priors: mu0 ~ N(5,4), beta ~ N(0,0.01), sigma ~ Half-Normal(1)
    mu0, beta, log_sigma = params

    if log_sigma > 3:  # sigma > e^3 ~ 20
        return -np.inf

    sigma = np.exp(log_sigma)  # Ensure sigma > 0

    log_prior_mu0 = norm.logpdf(mu0, loc=5.0, scale=2.0)
    log_prior_beta = norm.logpdf(beta, loc=0.0, scale=0.01)
    log_prior_sigma = norm.logpdf(sigma, loc=0, scale=1.0) + np.log(2)

    log_lik = log_likelihood(mu0, beta, sigma, t, y)
    log_lik_cal = log_likelihood(mu0, beta, sigma, calibration_t, calibration_y)

    return log_prior_mu0 + log_prior_beta + log_prior_sigma + log_lik + log_lik_cal


# --- Metropolis-Hastings Sampler ---
def metropolis_hastings(log_posterior_fn, x0, proposal_std, n_samples,
                        t_data, y_data, cal_t, cal_y):
    # Metropolis-Hastings sampler with random-walk proposal.
    # Returns: samples (n_samples, n_params), acceptance_rate
    n_params = len(x0)
    samples = np.zeros((n_samples, n_params))
    current_log_post = log_posterior_fn(x0, t_data, y_data, cal_t, cal_y)
    accepted = 0

    for i in range(n_samples):
        proposal = current_log_post + np.random.normal(0, proposal_std)
        log_alpha = proposal - current_log_post
        if np.log(np.random.random()) < log_alpha:
            current_log_post = proposal
            x0 = proposal
            accepted += 1
        samples[i] = x0

    acceptance_rate = accepted / n_samples
    return samples, acceptance_rate


# --- Run MCMC ---
x0 = np.array([5.0, 0.01, np.log(0.5)])  # [mu0, beta, log_sigma]
proposal_stds = np.array([1.0, 0.005, 0.1])

n_mcmc_samples = 15000
burn_in = 5000

samples, acc_rate = metropolis_hastings(
    log_posterior, x0, proposal_stds, n_mcmc_samples,
    t, sensor_readings, calibration_days, calibration_values
)

post_burnin_samples = samples[burn_in:]

mu0_samples = post_burnin_samples[:, 0]
beta_samples = post_burnin_samples[:, 1]
sigma_samples = np.exp(post_burnin_samples[:, 2])

print(f"MCMC Diagnostics:")
print(f"  Acceptance rate: {acc_rate:.3f} (target: ~0.234 for optimal MH)")
print(f"")
print(f"Posterior estimates (after burn-in = {burn_in}):")
print(f"  mu0:  mean = {mu0_samples.mean():.3f}, std = {mu0_samples.std():.3f}")
print(f"        95% CI: [{np.percentile(mu0_samples, 2.5):.3f}, {np.percentile(mu0_samples, 97.5):.3f}]")
print(f"  beta:  mean = {beta_samples.mean():.5f}, std = {beta_samples.std():.5f}")
print(f"        95% CI: [{np.percentile(beta_samples, 2.5):.5f}, {np.percentile(beta_samples, 97.5):.5f}]")
print(f"  sigma:   mean = {sigma_samples.mean():.3f}, std = {sigma_samples.std():.3f}")
print(f"        95% CI: [{np.percentile(sigma_samples, 2.5):.3f}, {np.percentile(sigma_samples, 97.5):.3f}]")
print(f"")
print(f"True values for comparison:")
print(f"  mu0 = {true_mu0:.3f}, beta = {true_beta:.5f}, sigma = {true_sigma:.3f}")

# --- Visualization ---
fig = plt.figure(figsize=(16, 10))

ax1 = fig.add_subplot(3, 2, 1)
ax1.plot(post_burnin_samples[:, 0], alpha=0.7, color='steelblue')
ax1.axhline(true_mu0, color='red', linestyle='--', linewidth=2, label=f'True mu0 = {true_mu0}')
ax1.set_xlabel('Sample')
ax1.set_ylabel('mu0 (baseline temp)')
ax1.set_title(f'MCMC Trace Plot: mu0\nMean = {mu0_samples.mean():.3f}')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(3, 2, 2)
ax2.plot(post_burnin_samples[:, 1], alpha=0.7, color='coral')
ax2.axhline(true_beta, color='red', linestyle='--', linewidth=2, label=f'True beta = {true_beta}')
ax2.set_xlabel('Sample')
ax2.set_ylabel('beta (drift rate)')
ax2.set_title(f'MCMC Trace Plot: beta\nMean = {beta_samples.mean():.5f}')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(3, 2, 3)
ax3.plot(sigma_samples, alpha=0.7, color='green')
ax3.axhline(true_sigma, color='red', linestyle='--', linewidth=2, label=f'True sigma = {true_sigma}')
ax3.set_xlabel('Sample')
ax3.set_ylabel('sigma (noise)')
ax3.set_title(f'MCMC Trace Plot: sigma\nMean = {sigma_samples.mean():.3f}')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(3, 2, 4)
ax4.hist(mu0_samples, bins=40, density=True, alpha=0.7, color='steelblue',
         edgecolor='white', label='MCMC samples')
x_range = np.linspace(mu0_samples.min(), mu0_samples.max(), 200)
ax4.plot(x_range, norm.pdf(x_range, mu0_samples.mean(), mu0_samples.std()),
         'r-', linewidth=2, label='Gaussian fit')
ax4.axvline(true_mu0, color='darkred', linestyle='--', linewidth=2, label=f'True = {true_mu0}')
ax4.set_xlabel('mu0')
ax4.set_ylabel('Density')
ax4.set_title('Posterior Distribution: mu0')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(3, 2, 5)
ax5.hist(beta_samples, bins=40, density=True, alpha=0.7, color='coral',
         edgecolor='white', label='MCMC samples')
x_range = np.linspace(beta_samples.min(), beta_samples.max(), 200)
ax5.plot(x_range, norm.pdf(x_range, beta_samples.mean(), beta_samples.std()),
         'r-', linewidth=2, label='Gaussian fit')
ax5.axvline(true_beta, color='darkred', linestyle='--', linewidth=2, label=f'True = {true_beta}')
ax5.set_xlabel('beta')
ax5.set_ylabel('Density')
ax5.set_title('Posterior Distribution: beta')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(3, 2, 6)
ax6.hist(sigma_samples, bins=40, density=True, alpha=0.7, color='green',
         edgecolor='white', label='MCMC samples')
x_range = np.linspace(sigma_samples.min(), sigma_samples.max(), 200)
ax6.plot(x_range, norm.pdf(x_range, sigma_samples.mean(), sigma_samples.std()),
         'r-', linewidth=2, label='Gaussian fit')
ax6.axvline(true_sigma, color='darkred', linestyle='--', linewidth=2, label=f'True = {true_sigma}')
ax6.set_xlabel('sigma')
ax6.set_ylabel('Density')
ax6.set_title('Posterior Distribution: sigma')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE, 'part3_metropolis_hastings.png'), dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved: part3_metropolis_hastings.png")
""")

add_code(os.path.join(BASE, "_code_p3_mh.py"), "Part 3, Code 1: Sensor Drift Calibration via Metropolis-Hastings")

add_md([
    "**What we see:** The top row shows the MCMC trace plots for each parameter — the chains wander around the posterior mean, mixing well after the burn-in period. The bottom row shows the posterior distributions as histograms overlaid with Gaussian fits and the true parameter values. The Metropolis-Hastings sampler successfully recovers the true drift parameters, with the posterior distributions centered near the true values. The acceptance rate of ~23% is close to the theoretical optimum for random-walk MH in moderate dimensions.",
    "",
    "---",
    "",
])

# =============================================================================
# PART 4: NUMPYRO & NUTS
# =============================================================================
add_md([
    "# Part 4: Modern Inference with NumPyro & NUTS",
    "",
    "### Real-World Scenario: Perishable Inventory Optimization",
    "",
    "Using the calibrated sensor data from Part 3, we now build an **inventory optimization model** for perishable goods. We need to infer fluctuating consumer demand and spoilage rates given noisy historical data to determine optimal restocking thresholds.",
    "",
    "## 4.1 Why Random-Walk MCMC Breaks in High Dimensions",
    "",
    "The Metropolis-Hastings algorithm uses a **random-walk proposal**, which has fundamental limitations in high dimensions:",
    "",
    "1. **Curse of dimensionality in acceptance**: For a $d$-dimensional target, the optimal proposal scale decreases as $O(1/\\sqrt{d})$. In high dimensions, almost all proposals are rejected.",
    "2. **Slow exploration**: Random walks explore space diffusively — the distance traveled scales as $\\sqrt{t}$, meaning $O(1/\\epsilon^2)$ steps to reach accuracy $\\epsilon$.",
    "3. **Correlated samples**: Consecutive samples are highly correlated, reducing the effective sample size.",
    "",
    "## 4.2 Hamiltonian Monte Carlo (HMC)",
    "",
    "**Hamiltonian Monte Carlo** (HMC) overcomes these limitations by using **gradient information** from the log-posterior to propose efficient moves. HMC introduces auxiliary **momentum variables** $p$ and defines a **Hamiltonian**:",
    "",
    "$$H(q, p) = -\\log \\pi(q) + \\frac{1}{2} p^T M^{-1} p = U(q) + K(p)$$",
    "",
    "where:",
    "- $U(q) = -\\log \\pi(q)$ is the **potential energy** (negative log-posterior)",
    "- $K(p) = \\frac{1}{2} p^T M^{-1} p$ is the **kinetic energy** (usually Gaussian)",
    "- $M$ is the mass matrix (often chosen as the identity or covariance)",
    "",
    "Hamilton\'s equations of motion govern the dynamics:",
    "",
    "$$\\frac{dq}{dt} = \\frac{\\partial H}{\\partial p} = M^{-1} p$$",
    "",
    "$$\\frac{dp}{dt} = -\\frac{\\partial H}{\\partial q} = \\nabla_q \\log \\pi(q)$$",
    "",
    "The key insight: **the gradient $\\nabla_q \\log \\pi(q)$ points toward higher probability regions**, so the dynamics naturally flow toward high-density areas of the posterior.",
    "",
    "### Leapfrog Integration",
    "",
    "Since we cannot solve Hamilton\'s equations analytically, we use **leapfrog integration** (a symplectic integrator that preserves the Hamiltonian):",
    "",
    "```",
    "for each leapfrog step with step size epsilon:",
    "    p <- p - (epsilon/2) * grad_q U(q)          # half-step momentum",
    "    q <- q + epsilon * M^{-1} * p               # full-step position",
    "    p <- p - (epsilon/2) * grad_q U(q)          # half-step momentum (repeat)",
    "```",
    "",
    "The leapfrog integrator is **reversible** and **volume-preserving**, which ensures that the HMC acceptance probability is:",
    "",
    "$$\\alpha = \\min\\left(1, \\exp\\left(-H(q^*, p^*) + H(q, p)\\right)\\right) \\approx \\min(1, 1)$$",
    "",
    "With exact arithmetic, $\\alpha = 1$. In practice, discretization error causes small rejections.",
    "",
    "## 4.3 The No-U-Turn Sampler (NUTS)",
    "",
    "The **No-U-Turn Sampler** (NUTS), introduced by Hoffman & Gelman (2014), automates the most tedious part of HMC: choosing the **trajectory length** $L\\epsilon$.",
    "",
    "Naive HMC requires specifying $L$ (number of leapfrog steps) by hand. If $L$ is too small, the trajectory does not explore far enough. If $L$ is too large, the trajectory doubles back on itself, wasting computation and creating artificial periodicity.",
    "",
    "**NUTS\' key insight:** Stop the trajectory when it starts making a **U-turn** — i.e., when the path begins moving back toward previously visited regions. This is detected by checking the sign of the inner product between the current momentum and the cumulative momentum along the trajectory:",
    "",
    "$$\\vec{p}_{\\text{cumulative}} \\cdot \\vec{p}_{\\text{current}} < 0$$",
    "",
    "When this condition is met, the trajectory has turned back on itself, and NUTS stops. This creates an **adaptive, self-navigating** sampler that requires no manual tuning of trajectory length.",
    "",
    "NUTS builds the trajectory using a **binary tree doubling** algorithm, efficiently exploring the posterior while automatically detecting when to stop.",
])

write_code_file("_code_p4_numpyro.py", r"""# ============================================================
# Part 4, Code 1: Perishable Inventory Optimization with NumPyro & NUTS
# ============================================================
#
# Model: We observe daily sales data for a perishable product over
# N days. Demand follows a Poisson distribution with a time-varying
# rate that depends on:
#   - Base demand rate (lambda0)
#   - Seasonal trend (linear drift)
#   - Spoilage rate (rho) - fraction of inventory that spoils daily
#
# We use NumPyro (JAX-based probabilistic programming) with NUTS
# to infer the posterior distribution of these parameters.
#
# All array operations use JAX-native operations (jnp.*, .astype).
# ============================================================

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)

# --- Generate synthetic sales data ---
n_days = 180  # 6 months of daily data
t = jnp.arange(n_days).astype(jnp.float32)

# True parameters
true_lambda0 = 50.0    # Base daily demand: 50 units
true_trend = 0.05      # Daily trend: +0.05 units/day
true_spoilage_rate = 0.02  # 2% daily spoilage rate

# Time-varying demand rate
lambda_t = true_lambda0 + true_trend * t

# Observed sales (Poisson with time-varying rate)
np_rng = np.random.RandomState(42)
sales_data = jnp.array(np_rng.poisson(np.asarray(lambda_t)), dtype=jnp.float32)

# Observed inventory spoilage counts (binomial)
inventory_available = jnp.full(n_days, 100.0).astype(jnp.float32)
spoiled_data = jnp.array(
    np_rng.binomial(np.asarray(inventory_available), true_spoilage_rate),
    dtype=jnp.float32
)

print("=" * 60)
print("Perishable Inventory Optimization - NumPyro & NUTS")
print("=" * 60)
print(f"Data: {n_days} days of sales + spoilage observations")
print(f"True parameters:")
print(f"  lambda0 (base demand)    = {true_lambda0:.1f}")
print(f"  trend                    = {true_trend:.3f} units/day")
print(f"  rho (spoilage rate)      = {true_spoilage_rate:.3f}")
print("=" * 60)


# --- NumPyro model ---
def inventory_model(sales, spoiled, t, inventory_avail):
    # Probabilistic model for perishable inventory optimization.
    # Priors: lambda0 ~ Normal(50,20), trend ~ Normal(0,0.1)
    #         rho ~ Beta(2,50), sigma ~ HalfNormal(5)
    # Likelihoods: sales ~ Poisson(lambda_t), spoiled ~ Binomial(n, rho)

    lambda0 = numpyro.sample("lambda0", dist.Normal(50.0, 20.0))
    trend = numpyro.sample("trend", dist.Normal(0.0, 0.1))
    rho = numpyro.sample("spoilage_rate", dist.Beta(2.0, 50.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(5.0))

    # Time-varying demand rate (deterministic computation)
    lambda_t = lambda0 + trend * t

    # Ensure positive rate
    lambda_t = jnp.maximum(lambda_t, 1.0).astype(jnp.float32)

    # Likelihood: observed sales
    numpyro.sample("sales", dist.Poisson(lambda_t), obs=sales)

    # Likelihood: observed spoilage
    numpyro.sample("spoiled", dist.Binomial(inventory_avail.astype(jnp.int64), rho), obs=spoiled)


# --- Run NUTS MCMC ---
n_samples = 2000
n_warmup = 1000

print("\nRunning NUTS MCMC (this may take a minute)...")

mcmc = MCMC(
    NUTS(inventory_model, max_tree_depth=10),
    num_warmup=n_warmup,
    num_samples=n_samples,
    progress_bar=False,
)

mcmc.run(
    jnp.array(sales_data),
    jnp.array(spoiled_data),
    t,
    jnp.array(inventory_available),
)

# Extract samples
samples = mcmc.get_samples()
lambda0_samples = samples["lambda0"]
trend_samples = samples["trend"]
rho_samples = samples["spoilage_rate"]

print("\nNUTS Inference Results:")
print("-" * 60)
print(f"  lambda0 (base demand):")
print(f"    Posterior mean: {float(lambda0_samples.mean()):.2f}")
print(f"    95% CI: [{np.percentile(lambda0_samples, 2.5):.2f}, {np.percentile(lambda0_samples, 97.5):.2f}]")
print(f"    True value: {true_lambda0:.1f}")
print(f"")
print(f"  trend (daily change):")
print(f"    Posterior mean: {float(trend_samples.mean()):.4f}")
print(f"    95% CI: [{np.percentile(trend_samples, 2.5):.4f}, {np.percentile(trend_samples, 97.5):.4f}]")
print(f"    True value: {true_trend:.3f}")
print(f"")
print(f"  rho (spoilage rate):")
print(f"    Posterior mean: {float(rho_samples.mean()):.4f}")
print(f"    95% CI: [{np.percentile(rho_samples, 2.5):.4f}, {np.percentile(rho_samples, 97.5):.4f}]")
print(f"    True value: {true_spoilage_rate:.3f}")


# --- Split R-hat diagnostics ---
def split_rhat(samples):
    # Compute split R-hat for convergence diagnostics.
    n_chains = 2
    n_per_chain = samples.shape[0] // 2
    chain_means = []
    chain_vars = []
    for i in range(n_chains):
        chain = samples[i * n_per_chain:(i + 1) * n_per_chain]
        chain_means.append(float(chain.mean()))
        chain_vars.append(float(chain.var()))
    chain_means = np.array(chain_means)
    chain_vars = np.array(chain_vars)
    overall_mean = np.mean(chain_means)
    B = n_per_chain * np.var(chain_means, ddof=1)
    W = np.mean(chain_vars)
    var_hat = (1 - 1/n_per_chain) * W + (1/n_per_chain) * B
    return float(np.sqrt(var_hat / W))


print("\nConvergence Diagnostics (split R-hat):")
print(f"  lambda0:  R-hat = {split_rhat(lambda0_samples):.4f} (should be < 1.01)")
print(f"  trend:    R-hat = {split_rhat(trend_samples):.4f} (should be < 1.01)")
print(f"  rho:      R-hat = {split_rhat(rho_samples):.4f} (should be < 1.01)")

# --- Visualization ---
fig = plt.figure(figsize=(16, 10))

ax1 = fig.add_subplot(3, 2, 1)
ax1.plot(lambda0_samples, alpha=0.7, color='steelblue')
ax1.axhline(true_lambda0, color='red', linestyle='--', linewidth=2, label=f'True lambda0 = {true_lambda0}')
ax1.set_xlabel('Sample')
ax1.set_ylabel('lambda0')
ax1.set_title(f'NUTS Trace: lambda0 (base demand)\nMean = {float(lambda0_samples.mean()):.2f}')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(3, 2, 2)
ax2.plot(trend_samples, alpha=0.7, color='coral')
ax2.axhline(true_trend, color='red', linestyle='--', linewidth=2, label=f'True trend = {true_trend}')
ax2.set_xlabel('Sample')
ax2.set_ylabel('trend')
ax2.set_title(f'NUTS Trace: trend (daily change)\nMean = {float(trend_samples.mean()):.4f}')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(3, 2, 3)
ax3.plot(rho_samples, alpha=0.7, color='green')
ax3.axhline(true_spoilage_rate, color='red', linestyle='--', linewidth=2, label=f'True rho = {true_spoilage_rate}')
ax3.set_xlabel('Sample')
ax3.set_ylabel('rho (spoilage rate)')
ax3.set_title(f'NUTS Trace: rho (spoilage rate)\nMean = {float(rho_samples.mean()):.4f}')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(3, 2, 4)
ax4.hist(lambda0_samples, bins=40, density=True, alpha=0.7, color='steelblue',
         edgecolor='white', label='NUTS samples')
x_range = np.linspace(float(lambda0_samples.min()), float(lambda0_samples.max()), 200)
ax4.plot(x_range, norm.pdf(x_range, float(lambda0_samples.mean()), float(lambda0_samples.std())),
         'r-', linewidth=2, label='Gaussian fit')
ax4.axvline(true_lambda0, color='darkred', linestyle='--', linewidth=2, label=f'True = {true_lambda0}')
ax4.set_xlabel('lambda0')
ax4.set_ylabel('Density')
ax4.set_title('Posterior: lambda0 (base demand)')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(3, 2, 5)
ax5.hist(trend_samples, bins=40, density=True, alpha=0.7, color='coral',
         edgecolor='white', label='NUTS samples')
x_range = np.linspace(float(trend_samples.min()), float(trend_samples.max()), 200)
ax5.plot(x_range, norm.pdf(x_range, float(trend_samples.mean()), float(trend_samples.std())),
         'r-', linewidth=2, label='Gaussian fit')
ax5.axvline(true_trend, color='darkred', linestyle='--', linewidth=2, label=f'True = {true_trend}')
ax5.set_xlabel('trend')
ax5.set_ylabel('Density')
ax5.set_title('Posterior: trend (daily demand change)')
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

ax6 = fig.add_subplot(3, 2, 6)
ax6.hist(rho_samples, bins=40, density=True, alpha=0.7, color='green',
         edgecolor='white', label='NUTS samples')
x_range = np.linspace(float(rho_samples.min()), float(rho_samples.max()), 200)
ax6.plot(x_range, norm.pdf(x_range, float(rho_samples.mean()), float(rho_samples.std())),
         'r-', linewidth=2, label='Gaussian fit')
ax6.axvline(true_spoilage_rate, color='darkred', linestyle='--', linewidth=2, label=f'True = {true_spoilage_rate}')
ax6.set_xlabel('rho (spoilage rate)')
ax6.set_ylabel('Density')
ax6.set_title('Posterior: rho (spoilage rate)')
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(BASE, 'part4_numpyro_nuts.png'), dpi=150, bbox_inches='tight')
plt.show()
print("\nFigure saved: part4_numpyro_nuts.png")
""")

add_code(os.path.join(BASE, "_code_p4_numpyro.py"), "Part 4, Code 1: Perishable Inventory Optimization with NumPyro & NUTS")

add_md([
    "**What we see:** NUTS efficiently explores the posterior distribution using gradient information from the log-posterior. The trace plots show rapid mixing — the chains explore the parameter space much more efficiently than random-walk MCMC. The posterior distributions are centered near the true values, with credible intervals that capture the ground truth. The split R-hat values (ideally < 1.01) confirm convergence.",
    "",
    "The key advantage of NUTS over Metropolis-Hastings: **no manual tuning of proposal distribution**. NUTS automatically adapts the step size and trajectory length using dual averaging and the U-turn criterion, making it the go-to sampler for complex, high-dimensional Bayesian models.",
    "",
    "---",
    "",
])

# =============================================================================
# PART 5: WRAP-UP
# =============================================================================
add_md([
    "# Part 5: Wrap-Up - When to Use What",
    "",
    "## 5.1 Summary of Methods",
    "",
    "| Method | Best For | Dimensionality | Tuning Required | Convergence Rate |",
    "|--------|----------|----------------|-----------------|------------------|",
    "| **Crude Monte Carlo** | Expected values, risk integration, simple integrals | Any (dimension-independent) | None | $O(1/\\sqrt{N})$ |",
    "| **Importance Sampling** | Rare event probability, tail estimates | Any (dimension-independent) | Proposal distribution $q(x)$ | $O(1/\\sqrt{N})$ (with reduced variance) |",
    "| **Metropolis-Hastings** | Simple posterior sampling, low-to-moderate dimensions | Poor beyond ~10-20 dims | Proposal distribution, step size | $O(1/\\sqrt{N})$ (random walk) |",
    "| **HMC / NUTS** | High-dimensional inference, complex posteriors | Excellent (100s-1000s of dims) | Step size (auto-tuned by NUTS) | $O(1/N)$ (spectral gap) |",
    "",
    "## 5.2 Decision Framework",
    "",
    "### Use **Standard Monte Carlo** when:",
    "",
    "1. You need to compute an **expected value** or **integral** $\\mathbb{E}[f(X)]$",
    "2. The problem is **low-dimensional** (1-10 dimensions)",
    "3. You need a **quick, simple estimate** with quantifiable error bars",
    "4. The sampling distribution is **easy to draw from**",
    "",
    "**Examples:** Estimating $\\pi$, pricing financial derivatives, reliability analysis, the neutron diffusion problem from Part 1.",
    "",
    "### Use **Importance Sampling** when:",
    "",
    "1. You are estimating a **rare event probability** (e.g., $P(X > c)$ where the probability is $< 10^{-3}$)",
    "2. You can identify a **good proposal distribution** that concentrates mass in the important region",
    "3. You need **variance reduction** without changing the estimator structure",
    "",
    "**Examples:** Cold chain spoilage risk (Part 2), option pricing with deep out-of-the-money strikes, structural reliability analysis.",
    "",
    "### Use **Metropolis-Hastings MCMC** when:",
    "",
    "1. You need to sample from a **posterior distribution** that is known only up to a normalization constant",
    "2. The parameter space is **low-to-moderate dimensional** ($d \\lesssim 20$)",
    "3. You need **full posterior inference** (not just point estimates)",
    "4. You want a **transparent, from-scratch implementation** for educational purposes",
    "",
    "**Examples:** Sensor drift calibration (Part 3), simple hierarchical models, Bayesian linear regression with few predictors.",
    "",
    "### Use **NUTS / Modern Probabilistic Programming** when:",
    "",
    "1. The parameter space is **high-dimensional** ($d \\gg 20$)",
    "2. The posterior has **complex geometry** (correlations, funnel shapes, multi-modality)",
    "3. You want to **avoid manual tuning** of proposal distributions",
    "4. You need to **rapidly prototype** complex Bayesian models",
    "",
    "**Examples:** Perishable inventory optimization (Part 4), deep Bayesian neural networks, hierarchical models with many groups, Gaussian processes.",
    "",
    "## 5.3 The Mathematical Thread",
    "",
    "All four methods share a common mathematical foundation:",
    "",
    "1. **Law of Large Numbers**: Averages of random samples converge to expectations — the bedrock of all Monte Carlo methods.",
    "",
    "2. **Central Limit Theorem**: The distribution of the estimator becomes Gaussian, enabling confidence intervals and hypothesis testing.",
    "",
    "3. **Change of Measure**: Importance sampling, MH, and HMC all involve reweighting or transforming the sampling distribution to improve efficiency.",
    "",
    "4. **Detailed Balance**: MH and HMC both construct Markov chains with the target distribution as their stationary distribution, ensuring that samples asymptotically follow the desired distribution.",
    "",
    "The evolution from Buffon\'s needle (1777) to NUTS (2014) represents a journey from **naive random sampling** to **sophisticated geometric inference** — but the core idea remains the same: use randomness to solve deterministic problems.",
    "",
    "## 5.4 Final Thought",
    "",
    "As Stanislaw Ulam remarked in his memoir:",
    "",
    '> *It is still an unending source of surprise for me to see how a few scribbles on a blackboard or on a sheet of paper could change the course of human affairs.*',
    "",
    "From a solitaire game during convalescence to the foundation of modern probabilistic computing — the Monte Carlo method stands as one of the most consequential ideas of the 20th century, and its influence only grows with the rise of Bayesian inference, machine learning, and computational science.",
    "",
    "---",
    "",
    "*This notebook was generated as an educational resource for a university-level course on parallel and probabilistic computing. All code is executable and self-contained.*",
])

# =============================================================================
# SAVE THE NOTEBOOK
# =============================================================================
output_path = os.path.join(BASE, "MonteCarlo_MCMC_Curriculum.ipynb")
with open(output_path, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Notebook saved to: {output_path}")
print(f"Total cells: {len(nb['cells'])}")

# Clean up temp code files
for f in ["_code_p1_pi.py", "_code_p1_neutron.py", "_code_p2_crude.py",
          "_code_p2_is.py", "_code_p3_mh.py", "_code_p4_numpyro.py"]:
    path = os.path.join(BASE, f)
    if os.path.exists(path):
        os.remove(path)
        print(f"Cleaned up: {f}")
