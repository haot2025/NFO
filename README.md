# Neural Fiber Operator (NFO)

Official implementation of **Neural Fiber Operators (NFO)** for diffusion MRI fiber orientation distribution (FOD) modeling.

NFO is a antipodal-symmetry-informed neural operator framework for diffusion MRI fiber reconstruction. Starting from a spherical fiber system, NFO derives a spectral operator solution, while preserving the antipodal structure of spherical fiber functions.

## Repository Structure

```text
NFO/
├── data/                   # Dataset loading and preprocessing utilities
├── models/                 # NFO and network architectures
├── options/                # Training and testing configurations
├── util/                   # Utility functions
│
├── train.py                # Model training
├── test.py                 # Whole-brain FOD reconstruction
│
├── requirements.txt        # Python dependencies
└── README.md
```

The core implementation of Neural Fiber Operators is located in `models/`.


## Training

The model can be trained using:

```bash
python train.py [options]
```

Training patches are randomly sampled from valid brain voxels. Each \(9\times9\times9\) LAR-FOD patch is used to predict the HAR-FOD at its center voxel.

Detailed training configurations and pretrained checkpoints will be released upon publication.

---

## Inference

Whole-brain reconstruction can be performed using:

```bash
python test.py [options]
```

During inference, valid patches are processed across the complete brain volume. To improve computational efficiency, patches along the first spatial dimension are grouped into the batch dimension and evaluated in parallel.

---

## Data Availability

The datasets used in this study are provided by their respective data owners and are subject to their individual data-use agreements.

- **HCP-YA:** Human Connectome Project, WU-Minn Consortium
- **HCP-A:** Human Connectome Project in Aging
- **MGH-HCP:** MGH-USC Human Connectome Project
- **CC:** microscopy-correlated ex-vivo corpus callosum dataset

This repository does **not** redistribute the original diffusion MRI datasets. Users should obtain access directly from the corresponding data providers and comply with their respective data-use terms.

---

## Citation

The manuscript is currently under review.

Citation information will be updated after the preprint becomes publicly available.
