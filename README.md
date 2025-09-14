# NLP Projects (Natural Language Processing)

This repository contains three projects focused on **text representation techniques** and their application in Natural Language Processing (NLP).  
Each notebook explores a different approach, from classical statistical methods to pre-trained deep learning models.

---

## Contents

1. **TF-IDF (Term Frequency – Inverse Document Frequency)**  
   - File: `nlp_tfidf.ipynb`  
   - Description: Implementation of a classical text representation model.  
   - Goal: Transform documents into numerical vectors based on the relative importance of words within a corpus.  
   - Use cases: text classification, similarity analysis, information retrieval.  

2. **Word2Vec**  
   - File: `nlp_word2vec.ipynb`  
   - Description: Training of a neural network-based embedding model that captures semantic relationships between words.  
   - Goal: Represent words in a vector space where semantically similar words are located close to each other.  
   - Use cases: semantic analysis, word clustering, NLP task enhancement.  

3. **BERT (Bidirectional Encoder Representations from Transformers)**  
   - File: `nlp_bert.ipynb`  
   - Description: Usage of a deep language model based on **Transformers**, pre-trained on large amounts of text.  
   - Goal: Apply contextual embeddings that capture the meaning of words depending on their context.  
   - Use cases: text classification, sentiment analysis, question answering.  

---

## Technologies used

- Python  
- Scikit-learn  
- Gensim  
- Transformers (Hugging Face)  
- Jupyter Notebook

---

## Results and Conclusion

In this project, different approaches for **SMS spam detection** were compared:

- **TF-IDF + SVM** achieved the **best overall performance** with **98.3% accuracy** and an **F1-score of 0.93**, showing that classical models combined with statistical text representations remain very competitive.  
- **TF-IDF + Naive Bayes** also performed strongly (**97% accuracy**), confirming its effectiveness as a fast and reliable baseline for spam detection.  
- **Word2Vec + Logistic Regression / SVM** delivered decent results (**95–96% accuracy**), although slightly below TF-IDF. This is likely because averaging embeddings does not fully capture the semantic and syntactic nuances of the messages.  

---

## Repository goal

Compare different approaches to text representation in NLP, from traditional statistical methods to deep learning models, highlighting their strengths and limitations.
Additionally, the repository includes a case study on SMS spam detection, where these approaches (TF-IDF and Word2Vec) are applied and compared in practice. This demonstrates how representation choices impact real-world text classification tasks.
