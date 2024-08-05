import numpy as np
import pickle
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences

def load_vocabulary_and_labels(vocab_path, pos_path, ner_path):
    with open(vocab_path, 'rb') as f:
        words_vocab = pickle.load(f)
    with open(pos_path, 'rb') as f:
        pos_vocab = pickle.load(f)
    with open(ner_path, 'rb') as f:
        ners_vocab = pickle.load(f)
    return words_vocab, pos_vocab, ners_vocab

def prepare_input(sentence, words_vocab, max_len=25):
    tokens = sentence.split()
    x = [words_vocab.get(token, words_vocab['ENDPAD']) for token in tokens]
    x_padded = pad_sequences(maxlen=max_len, sequences=[x], padding='post', value=words_vocab['ENDPAD'])
    return x_padded, tokens

def decode_predictions(predictions, pos_vocab, ners_vocab):
    pos_pred, ner_pred = predictions
    pos_pred = np.argmax(pos_pred, axis=-1)[0]
    ner_pred = np.argmax(ner_pred, axis=-1)[0]
    pos_vocab = dict(zip(pos_vocab.values(), pos_vocab.keys())) # reverse the dictionaries
    ners_vocab = dict(zip(ners_vocab.values(), ners_vocab.keys()))

    pos_labels = [pos_vocab[idx] for idx in pos_pred]
    ner_labels = [ners_vocab[idx] for idx in ner_pred]

    return pos_labels, ner_labels

def load_model_for_inference(model_path):
    return load_model(model_path)
