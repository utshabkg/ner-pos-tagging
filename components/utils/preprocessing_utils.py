import pandas as pd
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from bnlp import BasicTokenizer
import pickle
import os

def check_tokens(tokens):
    if isinstance(tokens, list):
        if len(tokens) == 2:
            return tokens[0]
        else:
            return ' '.join(tokens)  # Detokenize
    return tokens

def load_data(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None, names=['Token', 'POS', 'NER'], skip_blank_lines=False)
    data_cleaned = data.dropna(how='all').reset_index(drop=True)
    
    basic_tokenizer = BasicTokenizer()
    data['Token'] = data['Token'].apply(lambda x: basic_tokenizer.tokenize(x) if isinstance(x, str) else x)
    data['Token'] = data['Token'].apply(check_tokens)
    
    return data_cleaned

def add_sentence_numbers(data):
    data['Sentence #'] = None
    sentence_counter = 0

    for index, row in data.iterrows():
        if pd.isna(row['POS']) and pd.isna(row['NER']):
            sentence_counter += 1
        else:
            data.at[index, 'Sentence #'] = sentence_counter
    
    return data.dropna(subset=['POS', 'NER'])

def get_vocabulary(data):
    words = list(set(data['Token'].values))
    words.append('ENDPAD')
    poss = list(set(data['POS'].values))
    ners = list(set(data['NER'].values))
    
    words_vocab = {word: i for i, word in enumerate(words)}
    pos_vocab = {pos: i for i, pos in enumerate(poss)}
    ners_vocab = {ner: i for i, ner in enumerate(ners)}
    
    return words, poss, ners, words_vocab, pos_vocab, ners_vocab

def combine_word_ner_pos(data):
    tuples_fun = lambda s: [(word, pos, ner) for word, pos, ner in zip(s['Token'].values.tolist(), s['POS'].values.tolist(), s['NER'].values.tolist())]
    combination = data.groupby('Sentence #').apply(tuples_fun).tolist()
    return combination

def prepare_data(combination, words_vocab, pos_vocab, ners_vocab, max_len=25):
    x = [[words_vocab[tuple[0]] for tuple in c] for c in combination]
    x = pad_sequences(maxlen=max_len, sequences=x, padding='post', value=len(words_vocab) - 1)

    y_pos = [[pos_vocab[tuple[1]] for tuple in c] for c in combination]
    y_pos = pad_sequences(maxlen=max_len, sequences=y_pos, padding='post', value=len(pos_vocab) - 1)

    y_ner = [[ners_vocab[tuple[2]] for tuple in c] for c in combination]
    y_ner = pad_sequences(maxlen=max_len, sequences=y_ner, padding='post', value=len(ners_vocab) - 1)
    
    return x, y_pos, y_ner

def save_data(train_x, test_x, train_y_pos, test_y_pos, train_y_ner, test_y_ner, words_vocab, pos_vocab, ners_vocab, words, poss, ners, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    np.save(f'{output_dir}/train_x.npy', train_x)
    np.save(f'{output_dir}/test_x.npy', test_x)
    np.save(f'{output_dir}/train_y_pos.npy', train_y_pos)
    np.save(f'{output_dir}/test_y_pos.npy', test_y_pos)
    np.save(f'{output_dir}/train_y_ner.npy', train_y_ner)
    np.save(f'{output_dir}/test_y_ner.npy', test_y_ner)

    with open(f'{output_dir}/words_vocab.pkl', 'wb') as f:
        pickle.dump(words_vocab, f)

    with open(f'{output_dir}/pos_vocab.pkl', 'wb') as f:
        pickle.dump(pos_vocab, f)

    with open(f'{output_dir}/ners_vocab.pkl', 'wb') as f:
        pickle.dump(ners_vocab, f)

    with open(f'{output_dir}/words.pkl', 'wb') as f:
        pickle.dump(words, f)

    with open(f'{output_dir}/poss.pkl', 'wb') as f:
        pickle.dump(poss, f)

    with open(f'{output_dir}/ners.pkl', 'wb') as f:
        pickle.dump(ners, f)
