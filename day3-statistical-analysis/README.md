# Day 3: Statistics & Probability for Machine Learning

## 📊 Overview

This repository contains comprehensive materials for Day 3 of the ML Interview Preparation series, focusing on fundamental statistical concepts essential for machine learning practitioners. The content bridges theoretical understanding with practical implementation skills required in technical interviews.

## 🎯 Learning Objectives

By the end of this day, you will be able to:
- Apply Bayesian inference principles to machine learning problems
- Leverage the Central Limit Theorem for statistical inference
- Design and interpret A/B testing experiments
- Implement and evaluate distribution fitting techniques
- Build a Naive Bayes classifier with Laplace smoothing from scratch

## 📚 Topics Covered

### 1. Bayesian Inference and Priors
- **Bayes' Theorem**: Fundamental probability updating mechanism
- **Prior Distributions**: Uniform, conjugate, informative vs non-informative
- **Posterior Inference**: Combining prior beliefs with observed data
- **MAP vs MLE**: Maximum a posteriori vs Maximum likelihood estimation

### 2. Central Limit Theorem Applications
- **Theorem Statement**: Distribution of sample means approaches normality
- **Sampling Distributions**: Understanding variability in estimates
- **Standard Error**: Quantifying uncertainty in sample statistics
- **Confidence Intervals**: Constructing ranges for population parameters

### 3. Hypothesis Testing in A/B Testing
- **Hypothesis Formulation**: Null and alternative hypotheses
- **Error Types**: Type I (false positive) and Type II (false negative)
- **Statistical Power**: Probability of detecting true effects
- **Test Selection**: t-tests, z-tests, chi-square tests for different scenarios

### 4. Distribution Fitting and Goodness-of-Fit Tests 
- **Fitting Procedures**: Parameter estimation for common distributions
- **Goodness-of-Fit Tests**: Kolmogorov-Smirnov, Anderson-Darling
- **Visual Diagnostics**: Q-Q plots, probability plots, residual analysis
- **Model Selection**: Choosing appropriate distributions for data

## 🛠️ Main Project: Naive Bayes Classifier with Laplace Smoothing

### Project Structure
```
naive_bayes_project/
├── data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── naive_bayes.py
│   ├── preprocessing.py
│   └── evaluation.py
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── model_comparison.ipynb
├── tests/
│   └── test_naive_bayes.py
└── requirements.txt
```

### Key Implementation Features
- **From-scratch implementation** of Gaussian and Multinomial Naive Bayes
- **Laplace smoothing** to handle zero-probability issues
- **Cross-validation** for robust performance evaluation
- **Comparative analysis** with and without smoothing
- **Comprehensive testing** suite with edge cases

### Performance Metrics
- Classification accuracy
- Precision, Recall, and F1-score
- Confusion matrix analysis
- ROC curve and AUC (for binary classification)

## 📋 Prerequisites

### Mathematical Background
- Basic probability theory
- Understanding of conditional probability
- Familiarity with common probability distributions
- Linear algebra fundamentals

### Technical Requirements
- Python 3.8+
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn for visualization
- Scikit-learn for comparison benchmarks
- Jupyter Notebook for interactive exploration

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/your-username/ml-interview-day3
cd ml-interview-day3
pip install -r requirements.txt
```

### Quick Start
```python
from src.naive_bayes import NaiveBayesClassifier

# Initialize classifier with Laplace smoothing
nb_classifier = NaiveBayesClassifier(smoothing=1.0)

# Train on your data
nb_classifier.fit(X_train, y_train)

# Make predictions
predictions = nb_classifier.predict(X_test)
```

## 📊 Datasets

### Primary Dataset: Text Classification
- **Source**: 20 Newsgroups dataset subset
- **Task**: Multi-class document classification
- **Features**: TF-IDF vectorized text
- **Classes**: 4 categories (politics, sports, technology, entertainment)

### Secondary Dataset: Medical Diagnosis
- **Source**: Synthetic medical symptoms dataset
- **Task**: Binary classification (disease presence)
- **Features**: Categorical symptom indicators
- **Purpose**: Demonstrate categorical Naive Bayes

## 🎯 Interview Question Bank

### Foundational Concepts (Beginner)
1. Explain Bayes' theorem and its components
2. What is the Central Limit Theorem and when does it apply?
3. Define Type I and Type II errors with examples
4. How do you interpret a p-value?

### Applied Statistics (Intermediate)
5. Design an A/B test for a new website feature
6. How would you handle multiple testing problems?
7. Explain the difference between confidence and prediction intervals
8. When would you use a chi-square test vs a t-test?

### Machine Learning Integration (Advanced)
9. Why is the "naive" assumption in Naive Bayes often violated but still useful?
10. How does Laplace smoothing prevent overfitting?
11. Compare Naive Bayes with logistic regression for text classification
12. How would you validate that your model's assumptions hold?

### Business Application (Expert)
13. An A/B test shows statistical significance but minimal business impact. How do you proceed?
14. How would you explain confidence intervals to a non-technical stakeholder?
15. Design a statistical framework for detecting anomalies in user behavior

## 📈 Performance Benchmarks

### Expected Results
- **Naive Bayes (no smoothing)**: ~78% accuracy on test set
- **Naive Bayes (Laplace smoothing)**: ~82% accuracy on test set
- **Comparison with sklearn**: Within 2% accuracy difference
- **Training time**: <5 seconds on standard dataset

### Optimization Opportunities
- Feature selection and dimensionality reduction
- Hyperparameter tuning for smoothing parameter
- Ensemble methods combining multiple Naive Bayes models

## 🔍 Common Pitfalls and Solutions

### Statistical Misconceptions
- **P-hacking**: Multiple testing without correction
- **Confidence interval misinterpretation**: Not probability statements
- **Correlation vs causation**: Especially in A/B testing

### Implementation Issues
- **Zero probability problem**: Solved by Laplace smoothing
- **Numerical underflow**: Use log probabilities
- **Feature scaling**: Generally not needed for Naive Bayes

## 📚 Additional Resources

### Recommended Reading
- "Pattern Recognition and Machine Learning" by Christopher Bishop (Chapter 4)
- "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman (Chapter 6)
- "Think Stats" by Allen B. Downey for intuitive explanations

### Online Resources
- Khan Academy Statistics and Probability
- Coursera Statistical Inference course
- 3Blue1Brown Bayes' Theorem visualization

### Practice Platforms
- Kaggle Learn Statistics course
- DataCamp Statistical Thinking courses
- LeetCode database and statistics problems

## 🤝 Contributing

We welcome contributions to improve this educational resource:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new explanation'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **BETT ENOCK** - *Initial work* - [YourGitHub](https://github.com/Bettenoch)

## 🙏 Acknowledgments

- OpenML community for datasets
- Scikit-learn documentation for implementation references
- ML interview preparation community for question insights

---

**Next**: [Day 4 - Model Evaluation and Validation](../day4/README.md)  
**Previous**: [Day 2 - Linear Algebra and Calculus](../day2/README.md)

**Happy Learning! 🎓**