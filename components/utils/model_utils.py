import numpy as np
import pickle

def load_data(PROCESSED_DATA_DIR):
    train_x = np.load(f'{PROCESSED_DATA_DIR}/train_x.npy')
    test_x = np.load(f'{PROCESSED_DATA_DIR}/test_x.npy')
    train_y_pos = np.load(f'{PROCESSED_DATA_DIR}/train_y_pos.npy')
    test_y_pos = np.load(f'{PROCESSED_DATA_DIR}/test_y_pos.npy')
    train_y_ner = np.load(f'{PROCESSED_DATA_DIR}/train_y_ner.npy')
    test_y_ner = np.load(f'{PROCESSED_DATA_DIR}/test_y_ner.npy')

    with open(f'{PROCESSED_DATA_DIR}/words_vocab.pkl', 'rb') as f:
        words_vocab = pickle.load(f)

    with open(f'{PROCESSED_DATA_DIR}/pos_vocab.pkl', 'rb') as f:
        pos_vocab = pickle.load(f)

    with open(f'{PROCESSED_DATA_DIR}/ners_vocab.pkl', 'rb') as f:
        ners_vocab = pickle.load(f)

    with open(f'{PROCESSED_DATA_DIR}/words.pkl', 'rb') as f:
        words = pickle.load(f)

    with open(f'{PROCESSED_DATA_DIR}/poss.pkl', 'rb') as f:
        poss = pickle.load(f)

    with open(f'{PROCESSED_DATA_DIR}/ners.pkl', 'rb') as f:
        ners = pickle.load(f)

    return train_x, test_x, train_y_pos, test_y_pos, train_y_ner, test_y_ner, words_vocab, pos_vocab, ners_vocab, words, poss, ners

def split_validation_set(test_x, test_y_pos, test_y_ner, test_size=0.5, random_state=42):
    from sklearn.model_selection import train_test_split
    val_x, test_x, val_y_pos, test_y_pos, val_y_ner, test_y_ner = train_test_split(
        test_x, test_y_pos, test_y_ner, test_size=test_size, random_state=random_state
    )
    return val_x, test_x, val_y_pos, test_y_pos, val_y_ner, test_y_ner

def save_metrics(report_path, pos_accuracy, pos_classification_report, ner_accuracy, ner_classification_report):
    with open(report_path, 'w') as file:
        file.write("POS Tagging Metrics:\n")
        file.write(f"Accuracy: {pos_accuracy}\n\n")
        file.write("Classification Report:\n")
        file.write(pos_classification_report)

        file.write("\nNER Tagging Metrics:\n")
        file.write(f"Accuracy: {ner_accuracy}\n\n")
        file.write("Classification Report:\n")
        file.write(ner_classification_report)

def load_model(model_path):
    from keras.models import load_model
    return load_model(model_path)
