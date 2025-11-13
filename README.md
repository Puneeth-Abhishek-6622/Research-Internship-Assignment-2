# Research-Internship-Assignment-2: Neural Language Model Training

This repository documents an assignment to train a neural language model from scratch using PyTorch. The goal is to demonstrate an understanding of how sequence models learn to predict text and how model design and training affect performance.

The project involves:
* Cleaning and preprocessing a classic text corpus.
* Implementing two different tokenization strategies.
* Building, training, and evaluating multiple sequence models (LSTM, GRU, Transformer).
* Analyzing model performance using **Perplexity (PPL)**.

---

## 📖 Dataset & Preprocessing

### 📚 Dataset Source

The raw text for this analysis was sourced from **Project Gutenberg's** public domain library, using the full text of Jane Austen's **"Pride and Prejudice"**.

### ⚙️ Preprocessing Steps

Before tokenization, the raw text underwent the following sequence of preprocessing steps to ensure clean, standardized data:

1.  **Content Extraction:** The text was strictly isolated to the main narrative content. It was programmatically sliced from the starting marker **"CHAPTER I"** up to the defined end marker ("End of the Project Gutenberg"), removing all headers and footers.
2.  **Lowercasing:** All characters in the extracted text were converted to **lowercase** to ensure case-insensitivity (e.g., "The" and "the" are treated as the same word).
3.  **Cleaning:** **Punctuation**, **numbers**, and **extra spaces** (including tabs and multiple consecutive spaces) were removed using regular expressions. This simplifies the vocabulary, forcing the model to learn relationships purely from the words themselves.

---

## ✂️ Tokenization & Sequence Generation

Two distinct tokenization strategies were implemented and tested to prepare the text for the models.

### 1. Word-Level Tokenization (Used for Final Models)

* **Method:** Simple **space-separated** tokenization.
* **Description:** This traditional approach treats any contiguous string of non-space characters as a single token. A vocabulary was built from the set of all unique words present in the text.
* **Vocabulary Size:** **6,998 unique words**.
* **Dictionaries:** Two dictionaries, `word2idx` (word-to-index) and `idx2word` (index-to-word), were created to map words to unique integer IDs.

### 2. Subword-Level Tokenization (BPE)

* **Method:** **Byte-Pair Encoding (BPE)**.
* **Library:** The subword units were learned and applied using the **Hugging Face `tokenizers` library**.
* **Description:** BPE is a data compression technique adapted for NLP. It iteratively merges the most frequent adjacent characters to build a vocabulary of **subword units**.
* **Handling Rare/Unknown Tokens:** This approach robustly handles out-of-vocabulary (OOV) words by breaking them down into known subword units. Any truly unknown sequence is represented by the **`<UNK>`** (Unknown) token.

### Sequence Generation (Word-Level)

For the final models, the word-level token list was converted into fixed-length sequences.

* **Sequence Length (`seq_len`):** 30
* **Method:** A sliding window was moved across the token list. For every 30 consecutive words, a new data point was created where the first 30 words served as the `input` (features) and the 31st word served as the `target` (label).
* **Dataset Size:** This process resulted in **126,917** input-target pairs.
* **Data Split:** The dataset was split into a **90% training set** and a **10% validation set**.

---

## 📊 Evaluation Metric: Perplexity

**Perplexity (PPL)** is the standard and most widely used metric for evaluating the predictive performance of a language model (LM).

### 💡 Definition and Calculation

Perplexity quantifies how well a language model predicts a given test set.

* **Core Principle:** Lower perplexity indicates **better** predictive performance, as it implies the model assigns a higher probability to the observed sequence of words in the test set.
* **Relationship to Probability:** Minimizing perplexity is mathematically equivalent to **maximizing the probability** of the test set under the model.
* **Simple Calculation:** Perplexity is calculated by taking the **inverse probability** of the test set sequence and then normalizing this value by the **number of words ($N$)** in the test set.
    $$PPL(W) = P(w_1 w_2 \dots w_N)^{-\frac{1}{N}}$$

### 🌳 Interpreting Perplexity as a Branching Factor

Perplexity provides an intuitive interpretation regarding the model's uncertainty, often described as the **branching factor**.

* **Branching Factor:** This term represents the **average number of possible next words or tokens** (alternatives) that could plausibly follow a given context.
* **Weighted Average:** Perplexity can be thought of as the **weighted average branching factor** of the language. For instance, a perplexity of 100 suggests that, on average, the model is as confused at every decision point as if it were choosing uniformly and randomly among 100 next words.
* **Goal:** A good language model should have a low perplexity, indicating a **small effective branching factor**—it is confident and accurate in predicting the limited set of words that are likely to follow.

---

## 🧠 Model Architectures

All models were implemented from scratch in PyTorch to compare their effectiveness on this task.

* **LSTM**: A standard 2-layer LSTM model.
* **BiLSTM**: A 2-layer bidirectional LSTM.
* **BiLSTM + Attention**: A 2-layer BiLSTM with a Bahdanau-style attention mechanism.
* **GRU**: A standard 2-layer GRU model.
* **BiGRU**: A 2-layer bidirectional GRU.
* **BiGRU + Attention**: A 2-layer BiGRU with the same attention mechanism.
* **Transformer (Base)**: A Transformer model using 4 encoder layers.
* **Transformer (Variants)**: Other Transformer variants with different attention and output heads were also tested.

---

## ⚙️ Experimental Setup

The following parameters were used for training the primary RNN (LSTM/GRU) models:

* **Optimizer**: `torch.optim.AdamW`
* **Learning Rate**: `3e-4`
* **Loss Function**: `nn.CrossEntropyLoss`
* **Epochs**: 15
* **Early Stopping**: Patience set to `3`. If validation loss did not improve for 3 consecutive epochs, training was stopped.
* **Batch Size**: 128
* **Sequence Length**: 30
* **Embedding Dimension**: 256
* **Hidden Dimension**: 512
* **RNN Layers**: 2
* **Dropout**: 0.3

---

## 🏆 Results & Analysis

All models were trained on the word-level dataset until early stopping was triggered. The final validation loss and perplexity for the best epoch of each model are summarized below.

### Final Results Summary

| Model | Validation Loss | Validation PPL |
| :--- | :--- | :--- |
| **GRU** | **4.8097** | **127.98** |
| BiLSTM | 4.8314 | 134.37 |
| BiGRU | 4.8162 | 136.05 |
| BiGRU + Attn | 4.8256 | 136.98 |
| BiLSTM + Attn | 4.8234 | 137.52 |
| LSTM | 4.8948 | 142.27 |
| Transformer (Base) | 4.9786 | 163.65 |
| Transformer + Head | 5.2019 | 181.61 |
| Transformer + Attn | 5.9114 | 385.97 |

### 🔑 Key Observations

1.  **Best Model**: The **unidirectional GRU** was the clear winner, achieving the lowest validation perplexity of **127.98**. This suggests that a simpler architecture was most effective for this specific dataset size and task.

2.  **GRU vs. LSTM**: In both unidirectional and bidirectional forms, the GRU models consistently outperformed their LSTM counterparts (GRU PPL 127.98 vs. LSTM PPL 142.27).

3.  **Bidirectionality & Attention**: An interesting finding was that adding **bidirectionality or an attention mechanism consistently *worsened* the model's performance** (i.e., increased perplexity). This may indicate that for this narrative text, the most recent context (what just happened) is a much stronger predictor than the full-sequence context (what happens later).

4.  **Transformer Performance**: The Transformer models performed significantly worse than the RNNs. This is a common finding on smaller, single-book datasets. Transformers typically require massive amounts of data to learn effectively and outperform well-tuned RNNs.

5.  **Overfitting**: It was clearly observed across all models that training and validation loss would diverge after **epoch 7 or 8**. The validation loss would flatten or begin to rise, indicating that the models were starting to memorize the training data. The early stopping mechanism was crucial for capturing the "Best Fit" model before this overfitting occurred.

## 🚀 How to Run

1.  Clone the repository:
    ```bash
    git clone [https://github.com/Puneeth-Abhishek-6622/Research-Internship-Assignment-2.git](https://github.com/Puneeth-Abhishek-6622/Research-Internship-Assignment-2.git)
    cd Research-Internship-Assignment-2
    ```

2.  Install the required packages:
    ```bash
    pip install torch matplotlib tqdm tokenizers
    ```

3.  Ensure the dataset `Pride_and_Prejudice-Jane_Austen.txt` is available.

4.  Run the training notebook or script:
    * **Kaggle Notebook:** [LSTM, GRU, and Transformer Models](https://www.kaggle.com/code/melllogang/lstm-gru-and-transformer-models)
