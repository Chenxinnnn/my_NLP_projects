# 📚 Text Representation, Classification, and Word Embedding Learning

This week focuses on building basic text classifiers and learning word embeddings from corpus data. We explore how linear models perform on text classification tasks and how to extract dense semantic representations of words.

## 📂 Project Structure

- `submission.py` — Main code for classification and embedding tasks
- `test_classification.py` — Unit tests for text classification models
- `test_embedding.py` — Unit tests for word embeddings
- `hw1_writeup.pdf` — Written solutions with proofs and theoretical analysis
- `hw1_report.tex` — LaTeX source file (optional)
- `README.md` — Project overview and documentation (this file)

## 🧠 Key Tasks

- **Naive Bayes Classification**  
  - Built a multinomial Naive Bayes model for binary text classification.
  - Proved that the Naive Bayes decision boundary is linear.
  - Discussed the assumptions of Naive Bayes and methods to improve data separability.

- **Natural Language Inference (NLI)**  
  - Implemented a logistic regression model for NLI between sentence pairs.
  - Designed and compared unigram-based and custom feature extractors.
  - Performed error analysis on misclassified examples.

- **Word Embedding Learning**  
  - Constructed a word co-occurrence matrix from the *Emma* corpus.
  - Applied truncated SVD to obtain dense word vectors.
  - Compared similarity metrics: dot product vs cosine similarity.
  - Analyzed semantic similarity between words like "man", "woman", "happy", and "sad".

## 📈 Methods and Techniques

- Multinomial Naive Bayes for text classification
- Logistic Regression for sentence pair inference
- Bag-of-Words (unigram) and custom feature engineering
- Truncated SVD for low-dimensional word embeddings
- Cosine similarity for semantic comparison

## 🛠️ Tools and Libraries

- Python
- NumPy
- NLTK
- Matplotlib (optional for visualization)
- (Restricted usage of external libraries as per assignment rules)

## 📌 Notes

- Written parts contain mathematical derivations and theoretical analysis.
- Coding parts focus on building models from scratch with minimal library dependencies.
- Evaluation based on classification accuracy and semantic consistency of word embeddings.
