# Boosting Semantic Similarity with Transformers: A Multi-Model Study

A comparative study of transformer-based language models for **Semantic Textual Similarity (STS)** and **Automated Short Answer Grading (ASAG)**.

This project evaluates how different transformer architectures capture semantic meaning and how fine-tuning affects their performance on semantic similarity tasks.

---

## Overview

Semantic similarity is a fundamental Natural Language Processing (NLP) task that measures how closely two pieces of text convey the same meaning.

Traditional grading systems often rely on keyword matching, making them ineffective when students express correct concepts using different wording. This project explores transformer-based models that generate contextual sentence embeddings to understand semantic meaning rather than lexical overlap.

The study compares multiple transformer architectures across several benchmark datasets and evaluates both pretrained and fine-tuned models.

---

## Objectives

- Study how transformer models encode semantic information.
- Compare encoder, decoder and encoder-decoder architectures.
- Evaluate semantic similarity across multiple datasets.
- Analyze the impact of fine-tuning on model performance.
- Investigate transformer models for Automated Short Answer Grading (ASAG).

---

## Transformer Models Evaluated

| Model | Architecture | Purpose |
|--------|-------------|---------|
| SBERT (all-mpnet-base-v2) | Encoder | Sentence Embeddings |
| Pythia-160M | Decoder | Contextual Embeddings |
| T5-Base | Encoder-Decoder | Text-to-Text Sentence Embeddings |

---

## Datasets

The models were evaluated on three publicly available ASAG datasets:

- Mohler Dataset
- SciEntsBank Dataset
- Beetle Dataset

---

## Project Workflow

### 1. Data Preparation

- Dataset collection
- Text cleaning
- Standardization
- Score normalization

### 2. Embedding Generation

Sentence embeddings are generated using:

- SBERT
- Pythia
- T5

### 3. Similarity Computation

Similarity between student answers and reference answers is computed using embedding representations.

### 4. Experiments

Three experimental settings were conducted:

### Experiment 1
Embedding-based semantic similarity baseline.

### Experiment 2
Regression-based approach without transformer fine-tuning.

### Experiment 3
Fine-tuning transformer models for semantic similarity.

---

## Evaluation Metrics

Performance is evaluated using:

- Pearson Correlation
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Quadratic Weighted Kappa (QWK)

---

## Project Structure

```text
project/
│
├── datasets/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── preprocessing/
│   ├── embeddings/
│   │   ├── sbert_embedder.py
│   │   ├── pythia_embedder.py
│   │   └── t5_embedder.py
│   │
│   ├── experiments/
│   │   ├── experiment1_similarity.py
│   │   ├── experiment2_zeroshot.py
│   │   └── experiment3_finetune.py
│   │
│   └── evaluation/
│       ├── evaluation_metrics.py
│       └── run_evaluation.py
│
├── results/
│   ├── experiment1/
│   ├── experiment2/
│   └── experiment3/
│
└── README.md
```

---

## Tech Stack

- Python
- Hugging Face Transformers
- Sentence Transformers
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

---

## Key Features

- Comparative evaluation of three transformer architectures.
- Multi-dataset benchmarking.
- Fine-tuning experiments.
- Automatic semantic similarity scoring.
- Comprehensive performance analysis using multiple evaluation metrics.
- Visualization of experimental results.

---

## Results

The comparative analysis demonstrates that:

- Transformer-based embeddings significantly outperform traditional lexical similarity methods.
- Fine-tuning consistently improves semantic similarity performance.
- Different transformer architectures behave differently across datasets.
- Encoder-based sentence embedding models provide strong baseline performance for ASAG tasks.

---

## Applications

- Automated Short Answer Grading (ASAG)
- Semantic Textual Similarity (STS)
- Educational Technology
- Intelligent Tutoring Systems
- Question Answering
- Information Retrieval
- Document Similarity
- Plagiarism Detection

---

## Future Work

Future improvements include:

- Exploring larger transformer models.
- Studying layer-wise embedding representations.
- Cross-domain evaluation.
- Domain-specific fine-tuning.
- Knowledge distillation for lightweight deployment.
- Integration with modern Large Language Models.

---

## Authors

- Pragya Varshney
- Aryan Jain
- Astha Bansal

Department of Computer Science & Engineering  
Jaypee Institute of Information Technology, Noida

---

## Citation

If you use this work, please cite the project report accordingly.
