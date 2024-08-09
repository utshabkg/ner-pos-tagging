import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.models import load_model as keras_load_model


def load_data(processed_data_dir):
    train_x = np.load(f"{processed_data_dir}/train_x.npy")
    test_x = np.load(f"{processed_data_dir}/test_x.npy")
    train_y_pos = np.load(f"{processed_data_dir}/train_y_pos.npy")
    test_y_pos = np.load(f"{processed_data_dir}/test_y_pos.npy")
    train_y_ner = np.load(f"{processed_data_dir}/train_y_ner.npy")
    test_y_ner = np.load(f"{processed_data_dir}/test_y_ner.npy")

    with open(f"{processed_data_dir}/words_vocab.pkl", 'rb') as f:
        words_vocab = pickle.load(f)

    with open(f"{processed_data_dir}/pos_vocab.pkl", 'rb') as f:
        pos_vocab = pickle.load(f)

    with open(f"{processed_data_dir}/ners_vocab.pkl", 'rb') as f:
        ners_vocab = pickle.load(f)

    with open(f"{processed_data_dir}/words.pkl", 'rb') as f:
        words = pickle.load(f)

    with open(f"{processed_data_dir}/poss.pkl", 'rb') as f:
        poss = pickle.load(f)

    with open(f"{processed_data_dir}/ners.pkl", 'rb') as f:
        ners = pickle.load(f)

    return train_x, test_x, train_y_pos, test_y_pos, train_y_ner, \
        test_y_ner, words_vocab, pos_vocab, ners_vocab, words, poss, ners


def split_validation_set(test_x, test_y_pos, test_y_ner, seed=42):
    val_x, test_x, val_y_pos, test_y_pos, val_y_ner, test_y_ner = train_test_split(
        test_x, test_y_pos, test_y_ner, test_size=0.5, random_state=seed
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

    print(f"Metrics have been saved to '{report_path}'.")


def load_model(model_path):
    return keras_load_model(model_path)
