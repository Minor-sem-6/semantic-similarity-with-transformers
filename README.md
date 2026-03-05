# Transformer-Based Semantic Similarity Analysis

## Overview

This project explores how **Transformer-based models** contribute to **semantic similarity tasks in Natural Language Processing (NLP)**.
We evaluate different transformer architectures (e.g., SBERT, GPT, T5, BARD) to understand how well they capture semantic relationships between sentences.

The study involves:

* Generating **sentence embeddings**
* Computing **semantic similarity scores**
* Evaluating models **before and after fine-tuning**
* Comparing model performance and analyzing improvements.

---

## Objectives

* Analyze how transformer models encode semantic meaning.
* Compare semantic similarity performance across multiple transformer architectures.
* Evaluate model behavior **without fine-tuning vs. after fine-tuning**.
* Study the impact of different transformer layers on similarity performance.

---

## Models Evaluated

The following transformer models are analyzed in this project:

* SBERT
* GPT
* T5
* BARD

---

## Project Workflow

1. **Dataset Preparation**

   * Load and preprocess semantic similarity datasets.

2. **Embedding Generation**

   * Use transformer models to generate sentence embeddings.

3. **Similarity Computation**

   * Compute similarity using metrics such as:

     * Cosine Similarity
     * Euclidean Distance

4. **Baseline Evaluation**

   * Evaluate similarity performance **without fine-tuning**.

5. **Fine-tuning**

   * Fine-tune transformer models on the dataset.

6. **Post Fine-tuning Evaluation**

   * Compare improvements after training.

7. **Analysis**

   * Analyze performance differences across models and layers.

---
