# ✨ My NLP Projects (Fall 2024)

This repository showcases several **Natural Language Processing** (NLP) projects completed as part of coursework for *DS-GA-1011: Natural Language Processing with Representation Learning* at New York University.

Each folder contains a self-contained project focusing on different core NLP tasks, including classification, machine translation, and word embedding analysis.

🔹 **Note:** These projects are for educational purposes only and were completed individually or with minimal collaboration.  
🔹 **Goal:** To demonstrate understanding of NLP models, training, and evaluation through practical exercises.

---

## 📂 Project List
### 🔹 **lit_corpus_word_embeddings**

- **Task:** Build a text classifier using unigram and custom feature extractors on a small literary corpus (entailment classification).
- **Approach:** 
  - Implemented a **unigram feature extractor**.
  - Trained a perceptron model using **gradient-based updates**.
  - Performed basic error analysis to understand classification mistakes.
- **Results:**
  - **Training error rate:** < 0.3 ✅
  - **Test error rate:** < 0.4 ✅
- **Observations:**
  - Common failure cases involved misinterpreting synonyms, verb-object relations, and complex clauses.

### 🔹 **German_to_English_Machine_Translation**

- **Task:** Build a Transformer-based German-to-English translation model, including theoretical derivations, core module implementations, and BLEU score evaluation.
- **Approach:**
  - Derived **RNN backpropagation gradients** and analyzed **vanishing/exploding gradients**.
  - Explained **scaled dot-product attention** and **positional encoding** mechanisms.
  - Implemented:
    - `attention()` with masking and dropout
    - `MultiHeadedAttention()` module
  - Conducted beam search experiments with different beam sizes.
- **Results:**
  - Successfully passed all unit tests for attention modules. ✅
  - BLEU score analysis:
    - **Beam size 2–3** achieved highest BLEU (~38.9).
    - BLEU score slightly dropped with beam size 4–5.
- **Observations:**
  - Proper scaling in attention is crucial to prevent softmax saturation.
  - Moderate beam sizes balance exploration and translation accuracy.
  - Model struggles when beam size becomes too large due to noisy candidate translations.
 
### 🔹 **Fine-tuning_Language_Models**

- **Task:** Fine-tune a BERT-base model for sentiment analysis on the IMDB dataset, and explore robustness under out-of-distribution (OOD) transformations.
- **Approach:**
  - Fine-tuned BERT on IMDB original data, achieving high accuracy.
  - Designed a **custom typo insertion** transformation based on QWERTY keyboard neighbors.
  - Evaluated model robustness on typo-transformed test set.
  - Applied **data augmentation** by mixing 5,000 typo-transformed examples into the training data to improve OOD robustness.
- **Results:**
  - Accuracy on original test set: **0.929** ✅
  - Accuracy on typo-transformed test set: **0.87384**
  - After augmentation:
    - Original test set accuracy: **0.92664**
    - Transformed test set accuracy: **0.89332** ✅
- **Observations:**
  - Data augmentation with typos improved OOD performance by ~2%.
  - Original performance slightly dropped after augmentation but remained high.
  - Limitation: Only introducing typos targets a narrow OOD scenario; real-world OOD issues (e.g., synonyms, context shifts) require broader augmentation strategies.
 
### 🔹 **Arithmetic_Reasoning**

- **Task:** Fine-tune and prompt LLaMA-2 models to perform 7-digit integer addition tasks.
- **Approach:** 
  - Designed prompts to guide LLaMA-2 in correctly performing addition without internal calculation.
  - Implemented preprocessing and postprocessing strategies without leaking arithmetic logic.
  - Participated in leaderboard competition using customized prompt engineering.
- **Results:**
  - **Baseline average accuracy:** ~0.02
  - **After prompt tuning:** Improved to ~0.22
  - **Mean Absolute Error (MAE):** Reduced to below 5e6 ✅
- **Observations:**
  - Carefully designed prompts significantly improved model performance without modifying model weights.
  - Handling variability across different random seeds was key to stabilizing results across multiple evaluations.
