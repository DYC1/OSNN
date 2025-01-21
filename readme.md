# Solving Elliptic Optimal Control Problems via Neural Networks and Optimality System

This repository is the official implementation of the paper **"Solving Elliptic Optimal Control Problems via Neural Networks and Optimality System"**, see [*link*](https://arxiv.org/abs/2308.11925). 



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
   pip install jax[cuda12]  # Ensure you install the correct version for your CUDA
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

```
@misc{dai2024solvingellipticoptimalcontrol,
      title={Solving Elliptic Optimal Control Problems via Neural Networks and Optimality System}, 
      author={Yongcheng Dai and Bangti Jin and Ramesh Sau and Zhi Zhou},
      year={2024},
      eprint={2308.11925},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2308.11925}, 
}
```
## Contact
If you have any questions or issues, feel free to reach out via email at dyc1go@outlook.com or open an issue in the repository.