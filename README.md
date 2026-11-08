# Neural Architecture Search on NATS-Bench (Assignment 2)

> **Course:** Automated Machine Learning (AutoML) — Master's AI Track  
> **Institution:** Leiden University  
> **Authors:** Hippolyte Le Carreres  
> **Hand-in Package:** Source Code, README (`README.md`), and Scientific Report (4 pages excl. sources)

---

## 📂 Repository Structure

The codebase follows a modular design pattern, separating core NAS optimization algorithms, configuration setups, experiment execution scripts, and plotting utilities.

```text
├── src/                    # Core codebase (Modular architecture)
│   ├── search_methods/         # Search algorithms (Random, BO, SH, Learning Curves)
│   │   ├── random_search.py
│   │   ├── bo.py # Contains Matern and Weisfeiler-Lehman (WL) kernels
│   │   ├── successive_halving.py
│   │   └── random_sampling.py # old legacy code (random sample is done with random search)
│   ├── experiment.py          # GP surrogate models & acquisition functions (UCB)
│   ├── utils.py              # plotting + hlper functions 
│   ├── plotting.py          # plotting code for final results
│   ├── nats.py              # NATS-Bench API wrappers & metric logging
│   └── eval.py       # Evaluation logic across CIFAR-10, CIFAR-100, ImageNet-16-120
├── results/                # Saved logs, raw CSV metrics
├── figures/                # generated figs
├── requirements.txt        # Pinned dependencies
├── README.md               # This documentation file
└── report.pdf              # The final scientific report
```

## 💻 Installation

### 1. prerequisites
- Python 3.11+
- Recommended uv package manager with venv.
- Sufficient space for Fast mode = False (this is the only supported mode in this code base)

### 2. installation steps

```bash
# get git repo
git clone https://github.com/HippolyteLC/AutoML_A2.git

# setup venv
uv venv .venv --python python3.11

# install deps
uv pip install --upgrade pip
uv pip install -r requirements.txt
```

### 3. Nats-bench data setup (from the NTS-bench tutorial ipynb)

NATS-Bench is distributed as data files from the 2020-08-31 release. Download the files for the mode you'll use (see section 3):

| Mode | File | Download |
|---|---|---|
| `FAST_MODE = False` (lab PC) | `NATS-tss-v1_0-3ffb9.pickle.pbz2` | [Google Drive](https://drive.google.com/file/d/1vzyK0UVH2D3fTpa1_dSWnp1gvGpAxRul/view?usp=sharing) |

Steps:
1. **Lab PC:** you can probably keep `NATS-tss-v1_0-3ffb9.pickle.pbz2` compressed. The API reads it directly. 
3. Save the files in a folder you can find again. The NATS-Bench README suggests `$TORCH_HOME`, where the API finds them automatically. Otherwise give the path explicitly.
4. In `src/nats.py`, set `PATH_LOCAL` to the file.

If Google Drive blocks the download (it sometimes does for large files), the project README at [github.com/D-X-Y/NATS-Bench](https://github.com/D-X-Y/NATS-Bench) lists mirrors.


##    🔬 Reproducing experiments and code

- The entire experiment can be run be simply doing (make sure to be in the root of the git repo):
```bash 
python -m src.experiment

# or 

uv run python -m src.experiment
```

- To set different seeds, datasets, or search methods:
    1. go inside the `src/experiment.py`.
    2. set desired seeds as 
    ```python
    SEEDS = [1, 2, ..., 10]
    ```
    3. data sets can be deleted or added inside:
    ```python
    # make sure to add valid dataset strings
    DATASETS = ['cifar10', 'cifar100', 'ImageNet16-120']
    ```
    4. Methods can be deleted or added inside: 

    ```python 
    # make sure to add valid method classes
    METHODS = [BO, NASBOWL, RandomSearch, SuccessiveHalving, LearningCurvesSH]
    ```

- to generate figures run:
```bash
python -m src.plotting
```

## 📇 Key implementation details and params

| Component | Parameter / Setting | Value / Choice |
| :--- | :--- | :--- |
| **Search Space** | NATS-Bench Topology Space | 15,625 total architectures |
| **Total Epoch Budget ($B$)** | Normalized training budget | $100,000$ epochs |
| **Random Seeds** | Fixed evaluation seeds | 5 seeds (`[1, 2, 3, 11, 24]` |
| **Bayesian Optimization** | Initial Random Samples | 10 samples ($2,000$ epochs) |
| | GP Noise Variance ($\alpha_{GP}$) | $0.01$ |
| | Matern Kernel Smoothness ($\nu$) | $1.5$ |
| | WL Kernel Iterations ($n\_iter$) | $2$ |
| | Acquisition Function | Upper Confidence Bound (UCB, $\lambda = 2.0$) |
| **Successive Halving** | Initial Rung Budget ($e_i$) | $50$ epochs across $1,000$ architectures |
| | Reduction Factor ($\eta$) | $2$ |
| **Learning Curves** | Power-Law Form | $f(x) = c - ax^{-\alpha}$ |


