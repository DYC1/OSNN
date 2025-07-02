# Solving Elliptic Optimal Control Problems via Neural Networks and Optimality System

This repository is the official implementation of the paper [**"Solving Elliptic Optimal Control Problems via Neural Networks and Optimality System"**](https://doi.org/10.1007/s10444-025-10241-z). 



## Installation

### Prerequisites
- Python 3.10 or higher
- CUDA (if using GPU acceleration, ensure the correct version of CUDA is installed)

### Install Dependencies

This project relies on the following main libraries:
- `flax>=0.10.1`
- `jax>=0.4.35` (ensure compatibility with the correct CUDA version)
- `matplotlib>=3.10.0`
- `optax>=0.2.4`
- `orbax>=0.1.9`
- `tqdm>=4.66.6`

It is recommended to install these libraries following the official installation guides, especially for JAX, as it has specific dependencies for CUDA versions.

1. First, clone the repository:
   ```bash
   git clone https://github.com/DYC1/OSNN.git
   cd OSNN
   ```

2. Install JAX following the instructions on the official [JAX](https://jax.readthedocs.io/en/latest/quickstart.html) website to ensure compatibility with your CUDA version. For example, to install with GPU support, use:
   ```bash
   pip install --upgrade pip
   pip install -U "jax[cuda12]"  # Ensure you install the correct version for your CUDA
   ```

3. Install other dependencies using `pip`:
   ```bash
   pip install flax>=0.10.1 optax>=0.2.4 matplotlib>=3.10.0 orbax>=0.1.9 tqdm>=4.66.6
   ```

### Verify Installation
Once dependencies are installed, you can verify your JAX installation by running:
```bash
python -c "import jax; print(jax.__version__)"
```

## Citation
If you use this code, please cite the following paper:

```tex
@article {MR4925182,
    AUTHOR = {Dai, Yongcheng and Jin, Bangti and Sau, Ramesh Chandra and
              Zhou, Zhi},
     TITLE = {Solving elliptic optimal control problems via neural networks
              and optimality system},
   JOURNAL = {Adv. Comput. Math.},
  FJOURNAL = {Advances in Computational Mathematics},
    VOLUME = {51},
      YEAR = {2025},
    NUMBER = {4},
     PAGES = {Paper No. 31},
      ISSN = {1019-7168,1572-9044},
   MRCLASS = {99-06},
  MRNUMBER = {4925182},
       DOI = {10.1007/s10444-025-10241-z},
       URL = {https://doi.org/10.1007/s10444-025-10241-z},
}
```
## Contact
If you have any questions or issues, feel free to reach out via email at dyc1go@outlook.com or open an issue in the repository.
