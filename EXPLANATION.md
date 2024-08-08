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

## Model Architecture

### Overview

The chosen model is a sequence-to-sequence architecture combining an embedding layer with an LSTM (Long Short-Term Memory) layer. The model is designed to simultaneously predict POS tags and NER labels for each token in a sentence.

### Architecture Details

1. **Input Layer**
   The model begins with an input layer that accepts sequences of integers, where each integer corresponds to a token in a sentence. The input shape is defined as (max_len,), where max_len is the maximum length of input sequences (25 tokens in this case).
2. **Embedding Layer**
   An embedding layer is used to transform token indices into dense vectors of a fixed size (25 dimensions). This layer learns the representations of tokens during training and captures semantic relationships between them.
3. **LSTM Layer**
   An LSTM layer with 100 units is employed to capture temporal dependencies in the token sequences. LSTM units are particularly suitable for handling sequential data and learning context from previous tokens. The ``return_sequences=True`` parameter ensures that the LSTM layer outputs a sequence for each token, which is necessary for sequence-to-sequence tasks. ``recurrent_dropout=0.1`` is applied to prevent overfitting by randomly dropping connections during training.
4. **Output Layers
   POS Output Layer:** A Dense layer with a softmax activation function is used to predict POS tags for each token. The number of units in this layer corresponds to the number of unique POS tags.
   **NER Output Layer:** Similarly, another Dense layer with a softmax activation function is used for predicting NER labels. The number of units matches the number of unique NER labels.
5. **Compilation**
   The model is compiled with the Adam optimizer, which is known for its efficiency and adaptive learning rates. The loss functions used are sparse_categorical_crossentropy for both POS and NER tasks, suitable for multi-class classification problems. Accuracy is used as the metric to evaluate the model's performance on both tasks.

### Development Decisions

**Choice of Model**
The sequence-to-sequence model with an LSTM is chosen due to its ability to handle sequential data effectively. LSTMs are well-suited for tasks involving dependencies across sequences, such as POS tagging and NER, where the context of surrounding words is crucial.

**Embedding Layer**
The embedding layer transforms token indices into dense vectors, capturing semantic meanings and relationships between tokens. This representation is essential for the LSTM to learn meaningful patterns from the data.

**LSTM Layer**
LSTMs are preferred over standard RNNs due to their ability to retain long-term dependencies and mitigate issues like vanishing gradients. This is critical for understanding the context in longer sentences.

**Output Layers**
Separate Dense layers for POS and NER tasks allow the model to perform dual predictions in a single pass. This setup simplifies the architecture and avoids the need for separate models for each task.

### Training and Evaluation

**Early Stopping**
Early stopping is used to monitor the validation accuracy for NER and prevent overfitting by stopping training if no improvement is observed for 10 consecutive epochs.

**Hyperparameter Tuning**
Hyperparameter tuning is conducted using Bayesian Optimization. The search focuses on optimizing:

* Embedding Dimension: Determines the size of the dense vectors for tokens.
* LSTM Units: Controls the capacity of the LSTM layer.
* Recurrent Dropout: Adjusts the dropout rate to prevent overfitting.
* Learning Rate: Manages the step size during optimization.

**Model Checkpoints**
Model checkpoints are utilized to save the best model based on validation accuracy for NER. This ensures that the model with the highest performance is retained and used for evaluation.

**Evaluation Metrics**
Metrics such as accuracy and classification reports for both POS and NER are used to evaluate model performance. Precision, recall, and F1-score are analyzed to assess the effectiveness of the model in predicting different tags and labels.

## Challenges and Considerations

**Handling Missing Values**
Ensuring that rows with missing values are appropriately handled to avoid data corruption.

**Sequence Padding**
Determining the optimal sequence length and handling padding effectively to prevent data loss and ensure model efficiency.

**Vocabulary Size**
Managing the size of vocabularies to balance between coverage and computational efficiency.

**Data Imbalance**
Tried to apply SMOTE and some other techniques to solve the class imbalance issue. Due to some issues and time-limit, it was not completed. Hoping to solve it soon.

**Hyperparameter Tuning**
Finding optimal hyperparameters required careful experimentation. The tuning results suggested an embedding dimension of 40, 200 LSTM units, 0.5 recurrent dropout, and a learning rate of 0.01, though these parameters were not applied in the final model.
