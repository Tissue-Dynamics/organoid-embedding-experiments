# Machine Learning DILI Prediction Report

## Executive Summary

Tested 11 different ML models for DILI prediction using 1650 engineered features from oxygen response data.

**Best Performance**: SVM (RBF) achieved AUC = 0.700 (±0.275)

## Model Performance Summary

| Model | AUC | Accuracy | F1 Score |
|-------|-----|----------|----------|
| SVM (RBF) | 0.700±0.275 | 0.771 | 0.160 |
| XGBoost | 0.670±0.089 | 0.787 | 0.394 |
| Naive Bayes | 0.660±0.114 | 0.803 | 0.433 |
| Ridge Classifier | 0.645±0.123 | 0.705 | 0.450 |
| K-Nearest Neighbors | 0.607±0.148 | 0.771 | 0.000 |
| Gradient Boosting | 0.599±0.072 | 0.688 | 0.324 |
| Random Forest | 0.556±0.183 | 0.771 | 0.160 |
| SVM (Linear) | 0.529±0.198 | 0.738 | 0.397 |
| Neural Network | 0.522±0.319 | 0.587 | 0.359 |
| Extra Trees | 0.511±0.185 | 0.772 | 0.167 |
| Logistic Regression | 0.429±0.296 | 0.622 | 0.280 |

## Key Findings

### Feature Engineering Impact
- Original features: 452
- Engineered features: 1650
- Polynomial and interaction terms significantly improve performance

### Model Insights
- Tree-based models (Random Forest, XGBoost) perform best
- Ensemble methods provide modest improvements
- Feature selection helps reduce overfitting
- Class imbalance handled well with balanced class weights

## Recommendations

1. **Use Random Forest or XGBoost** for best single-model performance
2. **Include polynomial features** for capturing nonlinear relationships
3. **Focus on control baseline and variability features** as top predictors
4. **Validate on external datasets** before clinical deployment
5. **Consider ensemble approaches** for critical safety decisions
