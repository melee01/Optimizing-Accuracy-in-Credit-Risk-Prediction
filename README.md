# Optimizing Accuracy in Credit Risk Prediction  

> Supervised + unsupervised ML to assess borrower risk on real-world credit data from Kaggle.  

---

## Overview  
Credit risk prediction plays a vital role in financial decision-making, especially in loan approvals and credit card issuance. This project explores **supervised and unsupervised machine learning techniques** to improve the accuracy of credit risk assessment.  

By comparing different approaches, the aim is to identify models that effectively distinguish between good and bad credit behavior, contributing to **better financial stability** and **robust risk management**.  

**Dataset:** [Credit Card Approval Prediction](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction) (uploaded in this repo under `dataset/`).  

---

## Documents  
- [Project Proposal](./documents/credit_risk_prediction_proposal.pdf) – background, motivation, and detailed methodology.  
- [Presentation Slides](./documents/project_presentation_slides.pdf) – summarized key results, visuals, and final insights.  

---

## Motivation  
The 2008 Financial Crisis highlighted the dangers of poor credit risk assessment. Early identification of high-risk applicants can prevent defaults and strengthen the financial system.  

This project aims to:  
- Build predictive models to classify borrower risk.  
- Compare supervised regression models with unsupervised clustering/anomaly detection.  
- Improve interpretability and accuracy through feature engineering, model optimization, and evaluation metrics.  

---

## Dataset  
- **Source**: Kaggle ([link](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction))  
- **Features**: Demographic and financial information such as education level, income, employment status, number of children, and delinquency records.  
- **Target**: Credit risk status (derived from delinquency history).  

**Preprocessing steps included:**  
- Handling missing values  
- Encoding categorical features (one-hot, label encoding)  
- Scaling numerical values  
- Outlier detection and removal  
- Resampling techniques for imbalanced data  

---

## Methodology  

### Supervised Learning  
We modeled credit scores based on delinquency history and engineered additional features such as credit utilization rate and employment-age ratio.  

**Models tested:**  
- Linear & Lasso Regression  
- Decision Tree & Random Forest  
- LightGBM  

**Evaluation Metrics:**  
- Root Mean Squared Error (RMSE)  
- Mean Absolute Error (MAE)  
- R² Score  
- Feature importance analysis  

---

### Unsupervised Learning  
We explored unsupervised approaches to detect anomalies and segment borrowers without labels:  

- K-Means Clustering → grouping individuals into different risk categories  
- Isolation Forest → anomaly detection for high-risk borrowers  

**Evaluation Metrics:**  
- *Clustering*: Silhouette Score, Davies-Bouldin Index, Elbow Method  
- *Anomaly Detection*: AUC-ROC, anomaly score distributions, PCA visualizations  

---

## Tools & Libraries  
- Python  
- Jupyter Notebook  
- Scikit-learn  
- LightGBM / XGBoost  
- TensorFlow/Keras (optional deep learning extensions)  
- Matplotlib, Seaborn (EDA & visualization)  

---

## Results & Insights  

### Model Performance Comparison  

| Model             | RMSE   | MAE   | R²    | Source   |
|-------------------|-------:|------:|------:|----------|
| Decision Tree     | 0.0309 | 0.0047 | 0.9524 | Train/CV |
| Lasso Regression  | 0.1391 | 0.0696 | 0.0188 | Test     |
| Linear Regression | 0.1391 | 0.0697 | 0.0189 | Test     |
| Random Forest     | 0.0914 | 0.0473 | 0.5762 | Test     |
| LightGBM          | 0.0956 | 0.0497 | 0.5372 | Test     |

Random Forest and LightGBM provided the best balance of **accuracy** and **generalization**.  

---

## How to Run  

1. Clone the repository:  
   ```bash
   git clone https://github.com/melee01/Optimizing-Accuracy-in-Credit-Risk-Prediction.git
   cd Optimizing-Accuracy-in-Credit-Risk-Prediction
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook credit_risk_prediction.ipynb
4. Run all cells to reproduce the results.

---

## Repository Structure
   ```bash
   .
├── dataset/
│   ├── application_record.csv
│   └── credit_record.csv
├── documents/
│   ├── credit_risk_prediction_proposal.pdf   # project proposal
│   └── project_presentation_slides.pdf       # presentation slides
├── images/
│   ├── nb_image_03.svg   # Feature importance
│   ├── nb_image_07.svg   # Clustering results
│   ├── nb_image_12.svg   # Isolation Forest anomalies
│   ├── nb_image_15.svg   # ROC curve
│   └── ... (other figures)
├── credit_risk_prediction.ipynb              # main analysis notebook
├── requirements.txt                          # dependencies
└── README.md
   ```

---

## References
- Dataset: Kaggle – Credit Card Approval Prediction
- Academic papers and Kaggle discussions on credit scoring
- Documentation: Scikit-learn, LightGBM, XGBoost

---

## Contributors
1. Zhong Zhu Chen
2. He James
3. Melisa Lee
4. Liu Mingcheng
5. Syarwina Ridwan
6. Nur Diyanah Binte Hasan Malik
