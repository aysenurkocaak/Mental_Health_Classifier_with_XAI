# Mental Health Classifier with XAI 🌟🧠❤️

## Project Overview

This project is an artificial intelligence model developed to classify emotional health conditions based on text data. The model classifies user-provided sentences into categories such as **Anxiety**, **Bipolar Disorder**, **Depression**, **Normal**, **Personality Disorder**, **Stress**, and **Suicidal**. The project not only provides accurate classification results but also uses **LIME** and **PCA** techniques for explainable AI (XAI), making it easier to understand the model’s decision-making process.

🎯 The project is available as a live demo on Hugging Face : 🎯 

 👉 [** Live Demo : Mental Health Classifier (chatbotcuk) with XAI **](https://huggingface.co/spaces/aylaylomm/Mental_Health_Classifier_with_XAI)

---

## Project Contents 🌟

![](https://github.com/aysenurkocaak/photo/blob/main/Mental%20Health%20Classifier%20With%20XAI%20-%20a%20Hugging%20Face%20Space%20by%20aylaylomm%20-%20Opera%2026.04.2025%2020_29_40.png)
![](https://github.com/aysenurkocaak/photo/blob/main/Mental%20Health%20Classifier%20With%20XAI%20-%20a%20Hugging%20Face%20Space%20by%20aylaylomm%20-%20Opera%2026.04.2025%2020_29_49.png)
![](https://github.com/aysenurkocaak/photo/blob/main/Mental%20Health%20Classifier%20With%20XAI%20-%20a%20Hugging%20Face%20Space%20by%20aylaylomm%20-%20Opera%2026.04.2025%2020_30_13.png)

## Key Features 🌟

- **Text Classification**: Classifies sentences entered by users into 7 emotional health classes.
- **MPNet Embeddings**: Converts text into meaningful vector representations.
- **PCA for Dimensionality Reduction**: Reduces high-dimensional data to lower dimensions. Specifically, **PCA** reduces 768-dimensional data obtained from the **MPNet** model to 400 dimensions while preserving over 96.54% of the data.
- **XGBoost Classifier**: A powerful classifier for multi-class data.
- **LIME for Explainability**: Visually explains which words influenced the model's decisions.
- **Balancing (Random Under-sampling)**: Equalizes the sample size of over-represented classes to ensure that the model gives equal importance to all classes.
- **Cross-Validation**: The model is validated using 5-fold cross-validation to ensure reliability.

---

## Methods Used 🌟

1. **Data Preprocessing**:
   - The data is converted to lowercase.
   - Links, HTML tags, punctuation, and numbers are cleaned.
   - Missing values are removed.
   
2. **Feature Extraction**:
   - **MPNet** model is used to convert text into vector representations.
   - **PCA** is applied for dimensionality reduction (768 dimensions reduced to 400 dimensions, preserving 91% of the data).

3. **Class Imbalance Management**:
   - **Random Under-sampling** is used to equalize the sample size of imbalanced classes.

4. **Model Training**:
   - **XGBoost** algorithm is used for model training.
   - **SMOTE** is applied to increase the sample size for the under-represented class.

5. **Explainability**:
   - **LIME** is used to visually explain the model's decisions based on the words that most influenced the predictions.

6. **Application Development**:
   - The trained model is integrated into a **chatbot** interface, where users can input sentences and classify their emotional state.

---

## Technologies 🌟

- **Python**: The core programming language.
- **Hugging Face**: Used for MPNet and other transformer models.
- **PCA**: For dimensionality reduction.
- **XGBoost**: For model training.
- **LIME**: For model explainability.
- **Streamlit**: For creating the user interface.
- **Gradio**: For easy model deployment.
- **scikit-learn**: For data preprocessing and model evaluation.

---



