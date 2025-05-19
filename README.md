Okay, here's a comprehensive README.md file for your GitHub project based on the provided paper.

# CL-GNN: Binding Affinity Prediction of Protein-Ligand using Contrastive Learning and Graph Neural Networks

This repository contains the implementation for the paper "Binding Affinity Prediction of Protein-Ligand using CL-GNN Model" by Neeraj Bandhey and Afra Jakir Rehman. The project focuses on predicting protein-ligand binding affinity using a Contrastive Learning-Graph Neural Network (CL-GNN) framework, designed to leverage self-supervised learning on molecular graph representations.

## Table of Contents
1. [Abstract](#abstract)
2. [Introduction](#introduction)
3. [Key Features](#key-features)
4. [Methodology](#methodology)
    - [Data Augmentation](#data-augmentation)
    - [Contrastive Pre-training](#contrastive-pre-training)
    - [Fine-tuning for Binding Affinity Prediction](#fine-tuning-for-binding-affinity-prediction)
    - [Graph Neural Network Encoder](#graph-neural-network-encoder)
5. [Dataset](#dataset)
6. [Requirements](#requirements)
7. [Installation](#installation)
8. [Usage](#usage)
    - [Data Preparation](#data-preparation)
    - [Pre-training CL-GNN](#pre-training-cl-gnn)
    - [Fine-tuning and Evaluation](#fine-tuning-and-evaluation)
9. [Results](#results)
    - [Comparison with Other Models](#comparison-with-other-models)
    - [Performance on Train, Validation, and Test Sets](#performance-on-train-validation-and-test-sets)
    - [Progress After Presentation (Cross-Validation on Unseen Data)](#progress-after-presentation-cross-validation-on-unseen-data)
10. [Limitations](#limitations)
11. [Future Improvements](#future-improvements)
12. [Author Contributions](#author-contributions)
13. [Acknowledgments](#acknowledgments)
14. [References](#references)
15. [License](#license)

## Abstract
Classical experimental methods for determining protein-ligand binding affinity are typically labor-intensive and costly. This work introduces a Contrastive Learning - Graph Neural Network (CL-GNN) model to address these challenges. Due to computational constraints, the model is pretrained on a representative sample of 500 protein-ligand complexes from a larger dataset. The CL-GNN approach employs contrastive learning to learn protein and ligand representations from augmented molecular graphs. After fine-tuning, the model demonstrates competitive performance on benchmark sets, achieving a high concordance index (CI) and low root-mean-square error (RMSE). The framework also supports explainability through visualization, offering biological insights for drug optimization.

## Introduction
Accurate prediction of protein-ligand binding affinity is crucial in drug discovery and virtual screening. While conventional methods are accurate, they are computationally demanding. Machine learning (ML) offers a faster and more cost-effective alternative. This project explores a structure-based ML approach using self-supervised learning (SSL) to overcome the limitations of labeled data scarcity. Contrastive learning, a type of SSL, is employed to learn meaningful molecular representations from unlabeled data by contrasting positive and negative sample pairs. The CL-GNN framework combines GNNs with molecular graph augmentation techniques for this purpose.

## Key Features
- **Contrastive Learning:** Leverages self-supervised learning to learn robust representations from unlabeled protein-ligand data.
- **Graph Neural Networks (GNNs):** Utilizes GNNs (specifically AttentiveFP layers) to capture structural and chemical information from protein and ligand molecular graphs.
- **Molecular Graph Augmentation:** Employs techniques like Atom Masking, Bond Deletion, and Subgraph Removal to generate diverse views of molecular graphs for contrastive learning.
- **Two-Stage Learning:**
    1.  **Pre-training:** Learns general molecular representations in a self-supervised manner.
    2.  **Fine-tuning:** Adapts the pretrained model for the specific task of binding affinity prediction on labeled data.
- **Interpretability:** The model design allows for potential visualization of key residues and atoms contributing to binding affinity.

## Methodology
The CL-GNN framework consists of two main stages: pre-training and fine-tuning.

![CL-GNN Framework Overview](https://user-images.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/assets/FIGURE_1_PATH.png)
*(You should replace the above with an actual image from your project, or a link to Figure 1 in the paper if hosted elsewhere. For now, it's a placeholder)*

### Data Augmentation
For each protein-ligand pair, molecular graphs are augmented using three techniques, with two chosen randomly for each sample:
1.  **Atom Masking:** Randomly deletes a set of vertices (atoms) and their associated edges.
2.  **Bond Deletion:** Randomly removes edges (bonds) between atoms.
3.  **Subgraph Removal:** Removes subgraphs using random walk algorithms.
Approximately 20% of atoms, bonds, or substructures are randomly removed, creating structurally different but semantically similar graph pairs (positive pairs).

### Contrastive Pre-training
-   Protein and ligand graphs (original and augmented) are fed into separate GNN encoders (`fp()` for proteins, `fl()` for ligands).
-   The encoder outputs are concatenated and passed through a non-linear projection head.
-   The **NT-Xent loss** function is used to maximize the similarity between augmented views of the same complex (positive pairs) and minimize similarity with other complexes (negative pairs) in the batch.
    ```
    L_i,j = -log [ exp(sim(z_i, z_j)/τ) / Σ_{k=1, k≠i}^{2N} exp(sim(z_i, z_k)/τ) ]
    ```
    where `sim()` is cosine similarity, `z` are projected latent vectors, and `τ` is a temperature parameter.

### Fine-tuning for Binding Affinity Prediction
-   The pretrained GNN encoders are kept, and their weights are used as initialization.
-   A new, randomly initialized prediction head (e.g., MLP) is added on top of the concatenated GNN outputs.
-   The model is fine-tuned on a labeled dataset (e.g., CASF-2013) to predict binding affinity values (e.g., pKa or Ki/Kd).
-   Metrics used: Pearson's correlation coefficient (Rp), Root Mean Square Error (RMSE), and Concordance Index (CI).
    ```
    RMSE = sqrt( (1/N) * Σ (ŷ_n - y_n)^2 )
    ```

### Graph Neural Network Encoder
-   Proteins are converted into binding pockets (residues within 5.0 Å of the co-crystalized ligand), represented as an undirected graph `Gp = (Vp, Ep)`.
-   Ligands are represented as 2D undirected graphs `Gl = (Vl, El)`.
-   GNNs update node features iteratively:
    ```
    a_v^(k) = AGGREGATE^(k)({h_u^(k-1) : u ∈ N(v)})
    h_v^(k) = COMBINE^(k)(h_v^(k-1), a_v^(k))
    h_G = READOUT({h_v^(k) : v ∈ G})
    ```
-   The architecture uses three AttentiveFP layers, producing 64-dimensional representations.

## Dataset
-   **Pre-training:** A representative subset of 500 unlabeled protein-ligand complexes randomly sampled from the BioLiP database. Complexes were reprocessed using PDBFixer.
-   **Fine-tuning & Testing:**
    -   CASF-2013 benchmark set (derived from PDBbind v.2013).
    -   Fine-tuning test set: 300 complexes from CASF-2013 (out of a sample pool of 5000).
    -   Final testing: Complete CASF-2013 core set (285 complexes).

## Requirements
-   Python 3.x
-   PyTorch
-   PyTorch Geometric (PyG)
-   RDKit
-   NumPy
-   Pandas
-   Scikit-learn
-   Matplotlib (for plotting)
-   PDBFixer (for data preprocessing)

A `requirements.txt` file can be generated using:
```bash
pip freeze > requirements.txt


(It's recommended to create this in a clean virtual environment after installing all necessary packages.)

Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Install dependencies:

pip install -r requirements.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

(Ensure RDKit and PyTorch/PyG are installed correctly, potentially following their specific installation guides if issues arise.)

Usage
Data Preparation

Scripts for preprocessing BioLiP and CASF-2013 data should be run first. This typically involves:

Parsing PDB files for proteins and SDF/MOL2 files for ligands.

Generating graph representations (nodes, edges, features).

Using PDBFixer for cleaning protein structures.
(Provide specific commands or script names if available, e.g., python preprocess_biolip.py --input_dir path/to/biolip --output_dir data/preprocessed_biolip)

Pre-training CL-GNN

Run the pre-training script with appropriate configurations:

python pretrain_cl_gnn.py \
    --data_dir data/preprocessed_biolip_500 \
    --epochs 5000 \
    --batch_size 300 \
    --lr 1e-4 \
    --output_model_path models/cl_gnn_pretrained.pth
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

(Adjust parameters and paths as per your script.)

Fine-tuning and Evaluation

Run the fine-tuning script using the pretrained model, and then evaluate on the test set:

# Fine-tuning
python finetune_affinity.py \
    --pretrained_model_path models/cl_gnn_pretrained.pth \
    --train_data data/casf2013_finetune_train \
    --val_data data/casf2013_finetune_val \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-5 \
    --output_model_path models/cl_gnn_finetuned.pth

# Evaluation
python evaluate_affinity.py \
    --model_path models/cl_gnn_finetuned.pth \
    --test_data data/casf2013_core_set_285 \
    --output_results results/casf2013_predictions.csv
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

(Adjust parameters and paths as per your script.)

Results
Comparison with Other Models (CASF-2013 Test Set)
Model	CI	MSE	RMSE
SimBoost	0.740	1.901	1.378
DeepDTA	0.745	1.810	1.346
GraphDTA	0.734	1.993	1.418
PLA-MoRe	0.780	1.515	1.230
Proposed Model	0.800	2.160	1.417

The proposed model outperforms baselines in Concordance Index (CI).

MSE is slightly higher, RMSE is competitive.

Performance on Train, Validation, and Test Sets
Dataset	Split (Source)	MSE	RMSE	CI	Rp
Training	500 samples (BioLiP, for pre-training task)	1.559	1.249	0.832	0.865
Validation	300 samples (from 5000 CASF-2013 for fine-tune)	2.043	1.429	0.833	0.858
Test	285 samples (CASF-2013 core set)	2.164	1.471	0.800	0.891

(Plots from page 6 and 7 of the paper would be excellent here if you can replicate or include them as images.)

Training Performance: Strong fit on the small training set.

Good Generalization: Moderate decline from training to validation/test, indicating good generalization without significant overfitting.

Consistent CI: CI ≥ 0.800 across splits suggests consistent ranking ability.

Progress After Presentation (Cross-Validation on Unseen Data)

As suggested during a presentation, cross-validation was performed on a custom 10-fold split of unseen data (not part of the original training/CASF-2013 fine-tuning sets).

Results: MSE: 45.632, RMSE: 6.751, CI: 0.22

Analysis: These results are significantly poorer than on the benchmark sets. Potential reasons include:

The pre-training phase on only 500 diverse BioLiP samples may not have been sufficient to learn broadly generalizable representations for truly unseen, diverse chemical spaces.

The specific augmentations (atom masking, bond deletion, subgraph deletion) might inadvertently distort critical structural features essential for generalization to very different datasets, leading to misleading contrastive pairs for those new structures.

The "unseen dataset" used for this CV might have a distribution very different from BioLiP and CASF-2013.

This highlights the challenge of generalizing from limited pre-training data and the sensitivity of contrastive learning methods to augmentation strategies when applied to diverse, out-of-distribution datasets.

Limitations

Limited Pretraining Dataset: Only 500 samples from BioLiP were used, which might not capture the full chemical and structural diversity needed for optimal generalization.

Single Benchmark for Testing: Primary evaluation was on CASF-2013. Broader testing on other benchmarks (e.g., CASF-2016, other PDBbind core sets) would provide a more comprehensive performance profile.

Slightly Higher MSE on Test Set: Possibly due to structural or chemical diversity in the test set not fully encapsulated by the small training data.

Sensitivity to Augmentations: As shown by the "Progress After Presentation", the chosen augmentations might not be optimal for all types of molecular data, especially when generalizing to very different datasets.

Future Improvements

Balance Ranking and Regression Objectives: Combine ranking-based loss (for CI) with regression-based loss (for RMSE) during fine-tuning.

Calibration Techniques: Apply calibration layers or post-processing to improve absolute prediction accuracy.

Hyperparameter Tuning: Fine-tune learning rate, batch size, loss weights, and augmentation strategies to optimize both CI and RMSE.

Larger Pre-training Dataset: Utilize a much larger and more diverse dataset for pre-training.

Advanced Augmentation Strategies: Explore more sophisticated or adaptive data augmentation techniques.

Cross-Dataset Evaluation: Rigorously evaluate on multiple, diverse benchmark datasets.

Author Contributions

Afra Jakir Rehman: Choosing the topic, ideation for improvements, identifying shortcomings, future work planning, data sourcing guidance.

Neeraj Bandhey: Model implementation, training, fine-tuning, dataset editing, custom cross-validation setup, architectural understanding, report writing, generating output results.

Acknowledgments

This work was supported by Professor Dr. Manikandan Narayanan, Associate Professor in the Department of Computer Science & Engineering at Indian Institute of Technology Madras.

Special mention to Teaching Assistants Saish Jaiswal, Ritwiz Kamal, and Nency Bansal for their reviews and suggestions.

References

Li Q, Zhang X, Wu L, Bo X, He S, Wang S. PLA-MoRe: A Protein-Ligand Binding Affinity Prediction Model via Comprehensive Molecular Representations. J Chem Inf Model. 2022 Sep;62(18):4380-4390.

Zhang Y, Huang C, Wang Y, Li S, Sun S. CL-GNN: Contrastive Learning and Graph Neural Network for Protein-Ligand Binding Affinity Prediction. J Chem Inf Model. 2025 Apr;65(4):1724-1735. (Note: This is the citation style used in the paper for a future publication. This might be the actual reference for the work this project is based on or aims to be.)

Shen, J., et al. (Relevant paper for GNN protein pocket definition, if applicable).
(Add other key references from the paper if necessary)

License

This project is licensed under the MIT License - see the LICENSE.md file for details.
(It's good practice to include an actual LICENSE.md file in your repository. If you don't have one, you can choose a common open-source license like MIT.)

**Before committing this README.md:**
1.  **Replace Placeholders:**
    *   `YOUR_USERNAME/YOUR_REPO` with your actual GitHub username and repository name (for the image link).
    *   `FIGURE_1_PATH.png` with the actual path to your Figure 1 image if you include it.
    *   Update script names and command-line arguments in the [Usage](#usage) section to match your actual project structure.
2.  **Add a `LICENSE.md` file** if you choose a license like MIT.
3.  **Consider adding actual plots** from your results if possible (e.g., as images in a `docs` or `assets` folder and link them).
4.  If the reference "Zhang Y, et al. CL-GNN..." is indeed the foundational work this project implements or is inspired by, ensure its citation is accurate. The date "2025" suggests it's either a placeholder from the template used for the paper or a future expected publication. Clarify if possible.
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
