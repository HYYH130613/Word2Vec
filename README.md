# Word2Vec
Overview
Word2Vec is a popular technique for learning dense vector representations (embeddings) of words from large text corpora. The skip-gram model predicts the context words surrounding a target word, thereby capturing semantic and syntactic relationships. This project provides a clean, minimal implementation that can be easily understood and extended.

Features
Customizable vector dimensions and context window size.

Two loss functions: standard softmax (expensive) and efficient negative sampling.

Subsampling of frequent words to speed up training and improve embedding quality.

Negative sampling using a precomputed unigram table (based on word frequencies raised to 0.75 power).

Save/load model parameters during training to resume or evaluate later.

Visualization of learned embeddings using SVD (or any other dimensionality reduction).

Requirements
Python 3.12+

NumPy

Matplotlib (for visualization)

A dataset module named dataset_word2vec (see Dataset section below)
