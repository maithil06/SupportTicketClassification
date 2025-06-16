# End-to-End Support Ticket Classification System

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

A machine learning system to automatically classify IT support tickets using open-source Python libraries. This project offers a cost-effective, high-performance alternative to third-party software, covering the full ML lifecycle: data preprocessing, model benchmarking, hyperparameter tuning, and real-time prediction.

📓 **Kaggle Notebook**: [Explore the detailed analysis](https://www.kaggle.com/code/maithil06/support-ticket-classification)  
🌐 **GitHub Repository**: You're here!  

## Table of Contents
- [Features](#-features)
- [Models Explored](#-models-explored)
- [Tech Stack](#️-tech-stack)
- [Installation](#️-installation)
- [Usage](#-usage)
- [Performance](#-performance)
- [Future Improvements](#-future-improvements)
- [License](#-license)

## 🌟 Features
- **Automated Ticket Categorization**: Classifies raw ticket text into predefined categories.
- **Advanced NLP Preprocessing**: Text cleaning, stemming (Porter & Snowball), and TF-IDF vectorization with n-grams.
- **Comprehensive Model Benchmarking**: Evaluates a wide range of models to identify the best performer.
- **Hyperparameter Tuning**: Optimizes models using GridSearchCV and RandomizedSearchCV.
- **Ready-to-Use Prediction**: Scripts to save models and classify new tickets in real-time.

## 🤖 Models Explored
This project benchmarks multiple machine learning models:

| Category | Models |
|----------|--------|
| **Classical ML (TF-IDF)** | Multinomial Naive Bayes, Logistic Regression, Linear SVC, SGD Classifier |
| **Gradient Boosting** | XGBoost, LightGBM |
| **Deep Learning** | LSTM, GRU, 1D CNN (Inception-style) |
| **Transfer Learning** | Fine-tuned DistilBERT, RoBERTa (Hugging Face) |

## 🛠️ Tech Stack
- **Python**: 3.8+
- **Data & Analysis**: Pandas, NumPy
- **ML & Preprocessing**: Scikit-learn
- **Deep Learning**: TensorFlow 2.x, Keras
- **NLP**: NLTK, Hugging Face Transformers & Datasets
- **Gradient Boosting**: XGBoost, LightGBM
- **Serialization**: Joblib

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/maithil06/SupportTicketClassification.git
   cd SupportTicketClassification
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
