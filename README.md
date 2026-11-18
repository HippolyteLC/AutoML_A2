# Neural Architecture Search on NATS-Bench (Assignment 2)

> **Course:** Automated Machine Learning (AutoML) — Master's AI Track  
> **Institution:** Leiden University  
> **Author:** Hippolyte Le Carreres  
> **Hand-in Package:** Source Code, README (`README.md`), and Scientific Report (4 pages excl. sources)

---

## 📂 Repository Structure

The codebase follows a modular design pattern, separating core NAS optimization algorithms, configuration setups, experiment execution scripts, and plotting utilities.

```text
├── src/                        # Core codebase (Modular architecture)
│   ├── search_methods/         # Search algorithms (Random, BO, SH, Learning Curves)
│   │   ├── random_search.py    # Random Search & Random Sample implementations
│   │   ├── bo.py               # Bayesian Optimization (Matern and Weisfeiler-Lehman kernels)
│   │   ├── successive_halving.py # Successive Halving & Iterative Learning Curves (SH)
│   │   └── random_sampling.py  # Legacy code (Random Sample is handled in random_search.py)
│   ├── experiment.py           # Main experiment runner, GP surrogate models & UCB acquisition
│   ├── utils.py                # Plotting and general helper functions
│   ├── plotting.py             # Plotting script for generating final result figures
│   ├── nats.py                 # NATS-Bench API wrappers & metric logging
│   └── eval.py                 # Evaluation logic across CIFAR-10, CIFAR-100, and ImageNet16-120
├── results/                    # Saved logs and raw CSV metrics
├── figures/                    # Generated figures
├── requirements.txt            # Pinned Python dependencies
├── README.md                   # Project documentation
└── group_54_A2_report.pdf      # Final scientific report
```

---

## 💻 Installation

### 1. Prerequisites
- **Python:** 3.11+
- **Package Manager:** Recommended [`uv`](https://github.com/astral-sh/uv) package manager with virtual environment (`venv`) support.
- **Storage / RAM:** Sufficient RAM and disk space to run NATS-Bench with `FAST_MODE = False` (the only supported mode in this codebase).

### 2. Installation Steps

```bash
# Clone the repository
git clone https://github.com/HippolyteLC/AutoML_A2.git
cd AutoML_A2

# Set up virtual environment
uv venv .venv --python python3.11

# Activate virtual environment (Linux/macOS)
source .venv/bin/activate
# On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install --upgrade pip
uv pip install -r requirements.txt
```

### 3. NATS-Bench Data Setup
*(Adapted from the AutoML NATS-Bench tutorial `.ipynb`)*

NATS-Bench is distributed as data files from the `2020-08-31` release. Download the topology search space archive required for `FAST_MODE = False`:

| Mode | File | Download |
| :--- | :--- | :--- |
| `FAST_MODE = False` (Lab PC) | `NATS-tss-v1_0-3ffb9.pickle.pbz2` | [Google Drive](https://drive.google.com/file/d/1vzyK0UVH2D3fTpa1_dSWnp1gvGpAxRul/view?usp=sharing) |

**Setup Instructions:**
1. Download `NATS-tss-v1_0-3ffb9.pickle.pbz2`. On a lab PC, you can keep the file compressed—the NATS-Bench API reads it directly.
2. Save the file in an accessible directory. By default, the NATS-Bench README recommends placing it in `$TORCH_HOME`, where the API detects it automatically; otherwise, provide the path explicitly.
3. Open `src/nats.py` and set `PATH_LOCAL` to point to the downloaded file location.

> **Note:** If Google Drive blocks the download (which occasionally happens for large files), consult the official NATS-Bench repository at [github.com/D-X-Y/NATS-Bench](https://github.com/D-X-Y/NATS-Bench) for mirror links.

---

## 🔬 Reproducing Experiments and Figures

### 1. Running Full Experiments
Ensure you are in the root directory of the repository, then execute:

```bash
python -m src.experiment

# Or using uv:
uv run python -m src.experiment
```

Raw results and metric logs will be saved to the `results/` directory.

### 2. Customizing Seeds, Datasets, or Search Methods
To modify the experimental configuration, open `src/experiment.py` and edit the corresponding constants:

1. **Random Seeds:**
   ```python
   SEEDS = [1, 2, 3, 11, 24]
   ```
2. **Datasets** *(ensure valid NATS-Bench dataset strings are used)*:
   ```python
   DATASETS = ['cifar10', 'cifar100', 'ImageNet16-120']
   ```
3. **Search Methods** *(ensure valid method classes are imported and listed)*:
   ```python
   METHODS = [BO, NASBOWL, RandomSearch, SuccessiveHalving, LearningCurvesSH]
   ```

### 3. Generating Figures
Once experiments have completed and logs are stored in `results/`, generate the plots (saved to `figures/`) by running:

```bash
python -m src.plotting

# Or using uv:
uv run python -m src.plotting
```

---

## 📇 Key Implementation Details & Hyperparameters

| Component | Parameter / Setting | Value / Choice |
| :--- | :--- | :--- |
| **Search Space** | NATS-Bench Topology Space | $15,625$ total architectures ($a_i \in [0, 15624]$) |
| **Total Epoch Budget ($B$)** | Normalized training budget per run | $100,000$ epochs |
| **Random Seeds** | Fixed evaluation seeds | $5$ seeds (`[1, 2, 3, 11, 24]`) |
| **Random Search / Sample** | Evaluation fidelity ($e_i$) | Full fidelity ($200$ epochs; $1$ sample for Random Sample) |
| **Bayesian Optimization** | Initial Random Samples (DoE) | $10$ uniform random samples ($2,000$ epochs) |
| | Candidate Pool per Iteration | $500$ uniform random candidates (with replacement) |
| | GP Noise Variance ($\alpha_{GP}$) | $0.01$ |
| | Matern Kernel Smoothness ($\nu$) | $1.5$ (one-hot encoded architectures) |
| | WL Kernel Iterations ($n\_iter$) | $2$ (graph-encoded architectures, NASBOWL) |
| | Acquisition Function | Upper Confidence Bound (UCB, $\lambda = 2.0$) |
| **Successive Halving** | Initial Rung Budget ($e_i$) & Pool | $50$ epochs across $\|\mathcal{A}_{SH}\| = 1,000$ architectures |
| | Reduction Factor ($\eta$) & Rungs | $\eta = 2$, $s = \lceil \log_2(1000) \rceil = 10$ iterations (capped at $200$ epochs) |
| | Internal SH Budget ($B_{SH}$) | $\approx 500,000$ (calibrated via binary search to exhaust $B = 100,000$ with warm-starting) |
| **Learning Curves (SH)** | Power-Law Extrapolation Form | $f(x) = c - ax^{-\alpha}$ |
| | Parameter Initialization | $\hat{c} = y_{\max}$, $\hat{a} = \max(y_{\max} - y_{\min}, 0.01)$, $\hat{\alpha}_{LC} = 0.5$ |