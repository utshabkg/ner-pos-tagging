# Explanation Document

## Preprocessing Code and Development Decisions

### Introduction

This document provides an overview of the preprocessing code and the decisions made during its development. The preprocessing steps are designed to prepare the dataset for use in an end-to-end machine learning system for Named Entity Recognition (NER) and Parts of Speech (POS) tagging.

### Preprocessing Steps

1. **Loading the Dataset**
   The dataset is loaded from a TSV file containing columns for Token, POS, and NER. Each row in the file represents a token along with its corresponding POS tag and NER label. There is also a sentence which contains all the words and punctuations.
2. **Tokenization and Cleaning**
   A basic tokenizer from the bnlp library is used to tokenize the text data. This step involves breaking down the Token column into individual tokens, handling noise and special characters. e.g. "(২৭" is splited into "(" as a perfect PUNCT (punctuation).
3. **Identifying and Handling Sentences**
   Sentences are identified by rows where all columns (Token, POS, NER) are NaN. These rows indicate the end of a sentence. The dataset is cleaned by removing rows with all NaN values, and a new column Sentence # is introduced to assign unique identifiers to each sentence.
4. **Vocabulary Building**
   Vocabulary lists for words, POS tags, and NER labels are created. This involves collecting unique tokens and tags and assigning each a unique integer index. Special attention is given to include a padding token (ENDPAD) to handle sequences of varying lengths.
5. **Combining and Padding Sequences**
   The dataset is transformed into a list of tuples where each tuple contains a word, POS tag, and NER label for each sentence. These tuples are then converted into sequences of integers using the built vocabularies. Padding is applied to ensure that all sequences have a uniform length of 25 tokens (From Box-plotting, we knew that, the maximum sentence length ignoring outliers is around 25), with padding tokens assigned a specific index.
6. **Data Splitting**
   The dataset is split into training, validation, and test sets. An 80-20 split is used for the initial division into training and test sets. The test set is further divided into validation and test subsets, each comprising 50% of the remaining data. Finally, it was 80-10-10. This ensures that the model can be trained, validated, and tested effectively.

### Development Decisions

**Tokenization and Cleaning**
Using the bnlp library's BasicTokenizer ensures that tokenization is handled appropriately for Bengali text, addressing language-specific nuances.

**Sentence Identification**
Incorporating sentence boundaries is crucial for maintaining the structure of the data and ensuring that the model can learn from entire sentences rather than isolated tokens.

**Vocabulary Handling**
Creating separate vocabularies for words, POS tags, and NER labels allows for effective transformation of textual data into numerical format, which is essential for machine learning models.

**Padding Sequences**
Padding sequences to a fixed length of 25 tokens accommodates variations in sentence length while maintaining consistency in input size for the model.

**Data Splitting**
The chosen split ratios provide a balanced approach for training, validation, and testing, allowing for comprehensive evaluation and tuning of the model.

### Challenges Faced

**Handling Missing Values:** Ensuring that rows with missing values are appropriately handled to avoid data corruption.
**Sequence Padding:** Determining the optimal sequence length and handling padding effectively to prevent data loss and ensure model efficiency.
**Vocabulary Size:** Managing the size of vocabularies to balance between coverage and computational efficiency.
