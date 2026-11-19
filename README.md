# Student Academic Performance Prediction Using Machine Learning

## Project description
This educational project generates a reproducible synthetic dataset of 1,200 fictional student records and classifies performance into **At Risk**, **Average**, or **High**. It compares five supervised classifiers. The data is synthetic and is not validated for real student decisions.

## Team members
- K. Abhilash Goud — 25RBAIM0028 (the submitted report names this author).

Please add any additional confirmed team members and roll numbers before submitting a shared team repository. Each student must fork/push to their own public repository and submit their own repository URL.

## Dataset
`student_performance_dataset.csv` contains 1,200 generated records and 11 columns (10 features and the target). It is synthetic, so no external dataset link applies. The notebook also regenerates the data using a fixed random seed.

## Algorithms
- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (K=7)
- Gaussian Naive Bayes

## Install and run
```bash
python -m pip install -r requirements.txt
python student_academic_performance_prediction.py
```
The script regenerates `student_performance_dataset.csv`, prints model reports and comparison scores, and displays visualizations. To run interactively, open `student_academic_performance_prediction.ipynb` in Google Colab and run all cells.

## Results
On the notebook's saved 80/20 stratified split (960 train / 240 test), the recorded results are:

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 50.42% | 49.23% | 51.00% | 49.64% |
| Random Forest | 49.58% | 49.82% | 49.88% | 49.80% |
| Decision Tree | 49.17% | 48.22% | 49.86% | 47.64% |
| Gaussian Naive Bayes | 49.17% | 49.11% | 50.04% | 46.60% |
| K-Nearest Neighbors | 44.17% | 44.17% | 44.48% | 44.11% |

The notebook includes class distribution, model accuracy comparison, and confusion matrix plots. Logistic Regression has the highest recorded accuracy (50.42%) on this split. These figures describe only the synthetic demonstration dataset and should not be generalized to actual students.

## Submission notes
The cleaned report is included as `Student_Academic_Performance_Prediction_Report.pdf`. The presentation is included as an editable PowerPoint file. The unrelated `CLASS PPT.pdf` in the source folder covers email spam detection, so it was not included. Ensure this repository is **public** and each member submits their own repository link.
