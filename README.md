# DG-PBs-KD

**Confidence-aware multi-teacher knowledge distillation for imbalanced protein-DNA binding residue prediction**

DG-PBs-KD adds multi-teacher knowledge distillation to a diffusion-augmented graph neural network for residue-level DNA-binding prediction. The student uses frozen ESM-2 representations and parallel GCN, GAT and GraphSAGE branches. CLAPE-DB, ESM-NBR and TransBind provide training targets for real residues.

## Method

- **Soft targets:** a weighted mean of teacher probabilities provides continuous supervision.
- **Agreement weights:** weighted teacher disagreement determines the relative contribution of each residue to distillation. Agreement does not guarantee prediction correctness.
- **Class normalization:** positive and negative distillation losses are normalized separately.
- **Synthetic-node masking:** diffusion-generated positive nodes receive hard-label supervision but no teacher targets.
- **Student-only prediction:** independent-test prediction requires ESM-2 and the student graph network, without teacher inference or diffusion sampling.

## Main files

| File | Purpose |
| --- | --- |
| `kd_data_loader_v2.py` | ESM-2 features, residue alignment and graph construction |
| `improved_gnn_model.py` | Student graph-network backbone |
| `kd_gnn_model_v2.py` | Confidence-aware, class-normalized distillation |
| `ddpm_diffusion_model.py` | Diffusion model for feature augmentation |
| `kd_augmentation_v2.py` | Synthetic-positive augmentation and node masks |
| `kd_cv_confidence_v2.py` | Protein-level cross-validation for weighted three-teacher KD |
| `analyze_v2_multiseed_stable_threshold.py` | Threshold summary from training OOF predictions |
| `kd_final_train_test_v2_triple.py` | Final Train-573 training and Test-129/Test-181 evaluation |
| `kd_final_train_test_646_lkd1p4.py` | Final Train-646 training and Test-46 evaluation |
| `kd_cv_teacher_ablation.py` | Cross-validation for teacher-composition experiments |
| `run_dual_teacher_full_pipeline.py` | Dual-teacher label preparation, coefficient selection and final evaluation |

These instructions refer to the KD scripts above. The legacy `main.py` and `config.py` are not the entry points for reproducing the reported KD configurations.

## Environment

The core student pipeline imports:

- Python
- PyTorch
- PyTorch Geometric
- fair-esm
- NumPy
- SciPy
- scikit-learn

Install PyTorch and PyTorch Geometric builds compatible with your hardware and CUDA environment, together with the remaining packages. Teacher inference has additional dependencies in the corresponding teacher projects. Exact dependency versions are not pinned in this README; a clean-environment reproduction has not been verified for this documentation release.

The loader uses `esm2_t33_650M_UR50D` and extracts 1,280-dimensional residue representations. It reuses a local checkpoint when available and otherwise uses the fair-esm download/cache mechanism. A CUDA GPU is recommended for full training.

## Data preparation

Place the benchmark sequence and label files in:

```text
Raw_data/
  DNA-573_Train.txt
  DNA-129_Test.txt
  DNA-181_Test.txt
  DNA-646_Train.txt
  DNA-46_Test.txt
```

A basic protein record contains an identifier, an amino-acid sequence and one binary label per residue:

```text
>example_protein
ACDEFG
001010
```

Prepare the weighted three-teacher labels at:

```text
teacher_soft_labels_v2/
  DNA-573_Train_teacher_soft_labels_v2.csv
  DNA-646_Train_teacher_soft_labels_v2.csv
```

Each CSV must contain these columns:

```text
protein_id,residue_index,amino_acid,label,teacher_soft_prob,teacher_confidence
```

Residue indices are **1-based**. Identifiers, residue positions, amino acids and labels must match the raw training records. Teacher probabilities and confidence weights must lie in [0, 1]. Independent-test datasets do not require teacher-label CSVs.

The directory names above describe required inputs; they do not imply that datasets, teacher outputs or pretrained weights are bundled with every code release. Obtain third-party resources under their original terms and preserve the exact benchmark label versions.

## Running the three-teacher pipeline

Run commands from the project root after preparing the inputs and environment.

### 1. Check the training workflow

```bash
python kd_cv_confidence_v2.py --dataset 573 --seed 42 --lambda-kd 1.2 --smoke
```

Smoke mode is a reduced workflow check and does not reproduce the paper's scores.

### 2. Train-573 cross-validation and threshold preparation

```bash
python kd_cv_confidence_v2.py --dataset 573 --seed 42 --lambda-kd 1.2
python kd_cv_confidence_v2.py --dataset 573 --seed 2024 --lambda-kd 1.2
python kd_cv_confidence_v2.py --dataset 573 --seed 3407 --lambda-kd 1.2
python analyze_v2_multiseed_stable_threshold.py
```

The analysis script writes `kd_results/V2_lambda1p20_stable_threshold/stable_threshold_summary.json`. The Train-573 final-training script requires this file and the corresponding cross-validation outputs to load the threshold and training duration.

### 3. Final training and independent evaluation

```bash
python kd_final_train_test_v2_triple.py --seed 42
python kd_final_train_test_646_lkd1p4.py --seed 42
```

The first command trains on Train-573 and evaluates Test-129 and Test-181. The second trains on Train-646 and evaluates Test-46 using its locked settings. Both commands train a model before evaluation; they are not checkpoint-only inference commands.

## Running the dual-teacher pipeline

The dual configuration uses equal ESM-NBR and TransBind weights. Its driver requires teacher prediction exports at:

```text
ThreeTeacher_DNA_Predictor_ReadyCode/ThreeTeacher_DNA_Predictor/outputs/
  DNA_573_Train/residue_predictions.csv
  DNA_646_Train/residue_predictions.csv
```

Prepare these exports using the teacher pipeline and check the expected fields in `prepare_dual_soft_labels` within `run_dual_teacher_full_pipeline.py`.

```bash
python run_dual_teacher_full_pipeline.py --only 573
python run_dual_teacher_full_pipeline.py --only 646
```

These commands perform a coefficient search followed by final training and evaluation. They are substantially more expensive than a single training run. Outputs are stored under `kd_results/`.

## Reported configurations and results

Teacher fusion weights below are ordered as CLAPE-DB / ESM-NBR / TransBind.

| Configuration | Training set | Fusion weights | KD coefficient | Decision threshold | Final epochs |
| --- | --- | --- | --- | --- | --- |
| Triple | Train-573 | 0.25 / 0.35 / 0.40 | 1.2 | 0.240643 | 14 |
| Triple | Train-646 | 0.65 / 0.10 / 0.25 | 1.4 | 0.246401 | 14 |
| Dual | Train-573 | 0 / 0.50 / 0.50 | 1.4 | 0.219288 | 17 |
| Dual | Train-646 | 0 / 0.50 / 0.50 | 0.8 | 0.172343 | 15 |

All reported final models use seed 42, augmentation coefficient 0.2 and distillation temperature 1.0. Thresholds in this table are rounded for display; scripts and saved summaries retain the values used for evaluation.

| Test set | Triple MCC | Dual MCC | Triple F1 | Dual F1 |
| --- | --- | --- | --- | --- |
| DNA-129 | 0.4727 | 0.4600 | 0.5026 | 0.4919 |
| DNA-181 | 0.3314 | 0.3443 | 0.3516 | 0.3647 |
| DNA-46 | 0.4731 | 0.4717 | 0.5184 | 0.5114 |

These are manuscript results, not results from a new execution performed while preparing this README.

## Evaluation notes

- Diffusion augmentation is fitted within each training fold during cross-validation. Validation proteins remain unaugmented.
- Weighted three-teacher targets use fusion weights fitted on the complete training set. OOF scores are development estimates, not fully nested validation estimates.
- Final thresholds are selected using training data and fixed before independent evaluation.
- Current DNA-46 labels contain 10,876 residues and 965 positives. The earlier DG-PBs baseline used 10,867 residues and 956 positives, so those baseline scores are not directly comparable.
- Dual and triple configurations have different selected training settings. Their comparison does not isolate teacher count alone.
- Final results use one training seed. Protein-level bootstrap intervals do not measure variability across training seeds.
- Teacher-training overlap and sequence homology have not been fully audited. Separate ablations are needed to establish the contributions of agreement weighting and class normalization.

## Citation and acknowledgments

This code accompanies the manuscript **DG-PBs-KD: Confidence-aware multi-teacher knowledge distillation for imbalanced protein-DNA binding residue prediction**. Formal publication metadata will be added when available.

Please also acknowledge the original ESM-2, CLAPE, ESM-NBR and TransBind resources, the benchmark data sources, and the software libraries used in your experiments. Third-party code, datasets and model weights retain their original licenses and terms.

## Contact

Weizhong Lin: linweizhong@jcu.edu.cn
