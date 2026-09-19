India Cricket Match Outcome Prediction
Project Overview

This project analyzes and predicts India's international cricket match outcomes using a CART / Decision Tree classification model.

The project was developed as part of a Business Analytics project using Python and Scikit-learn.

Objective

The objective is to understand whether historical match-related variables can help explain India's match outcomes and to study how decision-tree pruning can reduce overfitting.

Target Variable
0 → India Won
1 → India Lost
Dataset

The dataset contains information related to India's international cricket matches, including team composition, match conditions, opponent, audience information and match-level statistics.

Total matches: 2,930
India wins: 2,457
India losses: 473
Missing values: None
Tools & Technologies
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook
Methodology
Data loading and exploration
Missing-value analysis
Data preprocessing
Categorical variable encoding
Train-test split
Decision Tree classification
Model evaluation
Overfitting analysis
Decision Tree pruning
Final model evaluation
Model Results
Before Pruning
Metric	Training	Testing
Accuracy	100%	91.01%
After Pruning
Metric	Training	Testing
Accuracy	87%	84%

The initial model showed signs of overfitting because of its perfect training accuracy. Pruning reduced the train-test performance gap and controlled the complexity of the tree.

Key Insights
Audience size and player performance variables were among the more influential features in the model.
The initial decision tree showed clear signs of overfitting.
Pruning reduced the difference between training and testing performance.
The confusion matrix and class-specific precision and recall provide additional insight beyond overall accuracy.
Repository Structure
india-cricket-match-outcome-prediction/
│
├── data/
│   └── Sports_Data.csv
│
├── notebooks/
│   └── cricket_match_prediction.ipynb
│
├── reports/
│   └── Business_Analytics_Report.pdf
│
├── images/
│
├── README.md
└── .gitignore
Future Improvements
Handle class imbalance
Tune model hyperparameters
Compare Decision Tree with Random Forest and Gradient Boosting
Evaluate ROC-AUC and Precision-Recall
Develop an interactive dashboard
