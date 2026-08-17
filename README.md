# 3dxnaes_code

# 3D-XANES Analysis Code

Analysis pipeline for 3D-XANES tomography data used to characterize particle-size-dependent
H2/H3 phase separation in high-Ni NMC cathode materials.

## Pipeline overview

1. **`tomo_rotcen.py`** — Determines the rotation center for tomographic reconstruction.
2. **`tomo_recon.py`** — Reconstructs 3D tomograms from raw projection data.
3. **`registration(1).py`** — Registers/aligns reconstructed volumes (e.g., across energy points or scans).
4. **`roi fit xanes_final.py`** — Performs voxel-wise XANES spectral fitting within particle ROIs to extract
   white-line energy (Ni oxidation state) per voxel.
5. **`auto masking_2025 BNL_include_update.py`** — Automatically segments and masks individual particles
   from the reconstructed volume for per-particle analysis.
6. **`threshold masking code except 1 voxel auto h2h3 ratio_group.py`** — Classifies each voxel as H2 or H3
   phase by setting a particle-specific energy threshold, calibrated against the H2/H3 fraction obtained
   from in-situ XRD at the corresponding SOC.
7. **`3d analysis (origin).py`** — Runs downstream statistical analysis (domain size/count, core-shell
   surface–bulk comparison, heterogeneity, etc.) and exports results in Origin-compatible format.

## Requirements

- Python 3.x
- (List required packages here, e.g., numpy, scipy, scikit-image, tomopy, etc.)

## Usage

(Add example commands / input data format here.)

## License

See [LICENSE](./LICENSE). This code is provided for viewing purposes only — usage requires
prior written permission from the author.

## Contact

gabriel97@snu.ac.kr
