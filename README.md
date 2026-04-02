# 📩 SMS Spam Classifier - Machine Learning & NLP
Welcome to the repository for my **Month 1 Internship Project** at **Arch Technology**. This project involves building an end-to-end NLP pipeline to classify SMS messages as "Spam" or "Ham" (Legitimate).

## 📌 Project Overview
The goal was to create a robust classifier that can handle the "noisy" nature of SMS text (slang, abbreviations, and irregular punctuation). 

### 🧠 Algorithm: Stochastic Gradient Boosting (SGB)
To finalize the project, I implemented **Stochastic Gradient Boosting**. 
* **Reduced Overfitting:** Introduced randomness to generalize better on unseen text.
* **Robustness:** Handled outliers and noisy data more effectively than standard Gradient Boosting.

## 🛠️ Technical Workflow
1. **Data Cleaning & EDA:** Handled severe class imbalance and visualized word frequencies.
2. **NLP Pipeline:** - Tokenization
   - Removing Stopwords & Punctuation
   - Stemming (using PorterStemmer)
3. **Vectorization:** Converted text to numerical data using `TfidfVectorizer`.
4. **Model Tournament:** Compared 10+ models (Naive Bayes, SVM, Random Forest, etc.) before selecting SGB for its precision.

## 💻 Tech Stack
* **Python** (Pandas, NumPy, Scikit-learn)
* **NLTK** (Natural Language Toolkit)

## 📂 Repository Structure
```bash
├── app.py                 
├── model.pkl              
├── vectorizer.pkl         
├── spam_classifier.ipynb
├── nltk.txt
├── train_model.py
└── README.md              
