# Grading-System-Model
# Automated Short Answer Grading (ASAG) with BERT and fully connected neural layers

## Overview
This project implements an AI-powered model for automated short-answer grading that ensures efficiency and fairness in assessment.

## Dataset Overview
Name of Dataset: ASAG Dataset
Data Source: github-https://github.com/DigiKlausur/ASAG-Dataset/tree/master
Original No of Records: 607
Purpose: Used for training and evaluating the grading model.
Key Columns in the Dataset
Student Answer: The response given by a student.
Reference Answer: The correct or ideal answer for evaluation.
Cosine Similarity: Measures similarity between student and reference answers.
Length Ratio: Ratio of the student's answer length to the reference answer.
Bigram Overlap: Overlapping bigrams between student and reference answers.
Aligned Score: Expert-graded scores used as ground truth.
Final Grade: The predicted score by the model.

## Features
**Data Preprocessing & Feature Engineering**
Cleaned and preprocessed text (noise removal, tokenization).
Extracted key features (cosine similarity, N-gram overlaps).

**Model Architecture**
Used Hybrid Deep Learning Model (BERT + Feature-based NN).
Implemented layers with activation functions.
Applied MSELoss and AdamW optimizer for stability.

**Training & Evaluation**
Used SMOTE to balance training data.
Evaluated using Accuracy, MAE as key performance metrics.
Applied hyperparameter tuning & augmentation to improve results.

## Installation & Setup
### Prerequisites
Ensure you have the following dependencies installed:
```sh
pip install torch transformers pandas numpy scikit-learn
```

### Dataset
The model uses the **ASAG dataset**, which includes student answers, reference answers, and grading features.

## Results & Analysis
- **Performance Metrics:**
  - Accuracy: Measures prediction accuracy
  - R-squared: Indicates model fit
- **Fairness Check:**
  - Correlation of grades with NLP features (e.g., cosine similarity, alignment scores)

## Future Improvements
- Integrate **Transformer-based grading models**(e.g., GPT)
- Improve fairness adjustments based on grading consistency analysis
- Add a web-based **grading interface** for usability

