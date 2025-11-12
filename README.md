# Research-Internship-Assignment-2
Train a neural language model from scratch using PyTorch. The goal is to demonstrate  understanding of how sequence models learn to predict text and how model design and  training affect performance. 


- Perplexity :in the context of language modeling, is a measure that quantifies how well a language model predicts a given test set, with lower perplexity indicating better predictive performance. In simpler terms, perplexity is calculated by taking the inverse probability of the test set and then normalizing it by the number of words. The lower the perplexity value, the better the language model is at predicting the test set. Minimizing perplexity is the same as maximizing probability
## Interpreting perplexity as a branching factor
- Perplexity can be interpreted as a measure of the branching factor in a language model. The branching factor represents the average number of possible next words or tokens given a particular context or sequence of words.
The Branching factor of a language is the number of possible next words that can follow any word. We can think of perplexity as the weighted average branching factor of a language.
