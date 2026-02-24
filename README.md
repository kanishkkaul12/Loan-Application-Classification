# DATA603 Project 1 — Loan Approval Classification

## Overview

This project benchmarks four supervised classification algorithms on a loan approval prediction task. Models are evaluated on a labeled dataset containing 28 applicant features, with the target variable being `LoanApproved` (binary: 0 or 1). Two experimental settings are compared:

- **Part 1 — Original Features**: Models trained on the full feature set
- **Part 2 — PCA-Reduced Features**: Models trained on PCA-compressed representations (5, 10, and 15 components)

***

## Dataset

| Split | File |
|-------|------|
| Training | `TrainingData.csv` |
| Testing | `TestingData.csv` |

Each dataset contains 28 columns including applicant demographics, financial history, loan details, and the binary label `LoanApproved`. The final column is the target label; all preceding columns are features.

***

## Methodology

### Classifiers Evaluated

- **Linear Discriminant Analysis (LDA)**
- **Decision Tree Classifier**
- **k-Nearest Neighbors (kNN)** — evaluated at k = 1, 3, 5, and 10
- **Support Vector Machine (SVM)** — linear kernel, C = 1.0

### Evaluation Metrics

All classifiers are evaluated using:

- **Accuracy** — proportion of correctly classified test samples
- **Type 1 Error (False Positive Rate)** — FP / (FP + TN); proportion of rejected applicants incorrectly classified as approved
- **Type 2 Error (False Negative Rate)** — FN / (FN + TP); proportion of approved applicants incorrectly classified as rejected

***

## Part 1 Results — Original Features

| Model | Accuracy | Type 1 Error | Type 2 Error |
|-------|----------|-------------|-------------|
| LDA | **0.905** | 0.160 | **0.030** |
| Decision Tree | 0.820 | 0.155 | 0.205 |
| kNN (k=1) | 0.783 | 0.250 | 0.185 |
| kNN (k=3) | 0.823 | 0.185 | 0.170 |
| kNN (k=5) | 0.833 | 0.200 | 0.135 |
| kNN (k=10) | 0.833 | 0.170 | 0.165 |
| SVM | 0.875 | 0.180 | 0.070 |

**Key observations:**
- LDA achieves the highest overall accuracy (90.5%) and the lowest Type 2 Error (3.0%), making it the strongest performer on the original feature set.
- SVM delivers the second-best accuracy (87.5%) with a very low Type 2 Error (7.0%).
- Decision Tree and kNN variants lag behind, with kNN at k=1 producing the worst accuracy (78.25%) and highest Type 1 Error (25%).

***

## Part 2 Results — PCA-Reduced Features

PCA was applied at three component levels (5, 10, 15) and only kNN (k=5) and SVM were evaluated in this setting.

| Model | PCA Components | Accuracy | Type 1 Error | Type 2 Error |
|-------|---------------|----------|-------------|-------------|
| kNN (k=5) | 5 | 0.843 | 0.190 | 0.125 |
| SVM | 5 | 0.865 | 0.190 | 0.080 |
| kNN (k=5) | 10 | 0.833 | 0.200 | 0.135 |
| SVM | 10 | 0.873 | 0.175 | 0.080 |
| kNN (k=5) | 15 | 0.833 | 0.200 | 0.135 |
| SVM | 15 | **0.885** | 0.165 | 0.065 |

**Key observations:**
- SVM consistently improves with more PCA components, reaching 88.5% accuracy and the lowest Type 2 Error (6.5%) at 15 components.
- kNN with PCA peaks at 5 components (84.25%) but plateaus or drops with more components, suggesting PCA provides little added benefit for kNN beyond 5 components.
- SVM with 15 PCA components slightly outperforms SVM on the original features (88.5% vs. 87.5%), suggesting that dimensionality reduction marginally improves SVM generalization here.

***

## How to Run

1. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```

2. **Place data files** in the same directory:
   - `TrainingData.csv`
   - `TestingData.csv`

3. **Run the notebook**
   ```bash
   jupyter notebook code.ipynb
   ```

The notebook executes in cell order and outputs accuracy, Type 1 Error, and Type 2 Error for each model, along with bar chart visualizations comparing all classifiers.

***

## File Structure

```
.
├── code.ipynb          # Main experiment notebook
├── TrainingData.csv    # Training split (28 features + label)
├── TestingData.csv     # Testing split (28 features + label)
└── README.md           # This file
```

***

## Dependencies

| Package | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Model training, PCA, and evaluation metrics |
| `matplotlib` | Result visualization |
