# Research-Internship-Assignment-2
Train a neural language model from scratch using PyTorch. The goal is to demonstrate  understanding of how sequence models learn to predict text and how model design and  training affect performance. 



# 📖 Pride and Prejudice Text Processing and Tokenization

This repository contains the results and methodologies for processing and tokenizing the full text of Jane Austen's **Pride and Prejudice**.

## 📚 Dataset Source

The raw text for this analysis was sourced from **Project Gutenberg**.

## ⚙️ Preprocessing Steps

Before tokenization, the raw text underwent the following sequence of preprocessing steps:

1.  **Content Extraction:** The text was strictly isolated to the main narrative content, specifically from the starting marker **"CHAPTER I"** up to the defined end marker of the book.
2.  **Lowercasing:** All characters in the extracted text were converted to **lowercase** to ensure case-insensitivity during tokenization.
3.  **Cleaning:** **Punctuation**, **numbers**, and **extra spaces** (including tabs and multiple consecutive spaces) were removed.

---

## ✂️ Tokenization Approaches

Two distinct tokenization strategies were implemented and tested to prepare the text for further natural language processing tasks:

### 1. Word-Level Tokenization

* **Method:** Simple **space-separated** tokenization.
* **Description:** This traditional approach treats any contiguous string of non-space characters as a single token, essentially resulting in a dictionary of all unique words found in the cleaned text.

### 2. Subword-Level Tokenization (BPE)

* **Method:** **Byte-Pair Encoding (BPE)**.
* **Library:** The subword units were learned and applied using the **Hugging Face `tokenizers` library**.
* **Description:** BPE is a data compression technique adapted for NLP. It iteratively merges the most frequent adjacent characters or character sequences to build a vocabulary of **subword units**.
* **Handling Rare/Unknown Tokens:** This approach is robust and inherently handles out-of-vocabulary (OOV) or rare words by breaking them down into known subword units. Any truly unknown or un-tokenizable sequence is represented by the **`<UNK>`** (Unknown) token.

# 📊 Evaluation Metric: Perplexity

**Perplexity (PPL)** is the standard and most widely used metric for evaluating the predictive performance of a language model (LM).

## 💡 Definition and Calculation

Perplexity quantifies how well a language model predicts a given test set.

* **Core Principle:** Lower perplexity indicates **better** predictive performance, as it implies the model assigns a higher probability to the observed sequence of words in the test set.
* **Relationship to Probability:** Minimizing perplexity is mathematically equivalent to **maximizing the probability** of the test set under the model.
* **Simple Calculation:** Perplexity is calculated by taking the **inverse probability** of the test set sequence and then normalizing this value by the **number of words ($N$)** in the test set.
    $$PPL(W) = P(w_1 w_2 \dots w_N)^{-\frac{1}{N}}$$

## 🌳 Interpreting Perplexity as a Branching Factor

Perplexity provides an intuitive interpretation regarding the model's uncertainty, often described as the **branching factor**.

* **Branching Factor:** This term represents the **average number of possible next words or tokens** (alternatives) that could plausibly follow a given context or sequence of words.
* **Weighted Average:** Perplexity can be thought of as the **weighted average branching factor** of the language modeled. For instance, a perplexity of 100 suggests that, on average, the model is as confused at every decision point as if it were choosing uniformly and randomly among 100 next words.
* **Goal:** A good language model should have a low perplexity, indicating a **small effective branching factor**—it is confident and accurate in predicting the limited set of words that are likely to follow.
