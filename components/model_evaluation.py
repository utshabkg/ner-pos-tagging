from utils.model_utils import load_data, split_validation_set, save_metrics, load_modell
import numpy as np
import random
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import classification_report, accuracy_score

class ModelEvaluator:
    def __init__(self, model_path, report_path, PROCESSED_DATA_DIR, seed=42):
        self.model_path = model_path
        self.report_path = report_path
        self.PROCESSED_DATA_DIR = PROCESSED_DATA_DIR
        self.seed = seed
        self.set_seed()

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def load_and_split_data(self):
        train_x, test_x, train_y_pos, test_y_pos, train_y_ner, test_y_ner, words_vocab, pos_vocab, ners_vocab, words, poss, ners = load_data(PROCESSED_DATA_DIR)
        val_x, test_x, val_y_pos, test_y_pos, val_y_ner, test_y_ner = split_validation_set(test_x, test_y_pos, test_y_ner)
        return train_x, val_x, test_x, train_y_pos, val_y_pos, test_y_pos, train_y_ner, val_y_ner, test_y_ner, pos_vocab, ners_vocab

    def evaluate_model(self, model, test_x, test_y_pos, test_y_ner, batch_size=128):
        return model.evaluate(test_x, {"pos_output": test_y_pos, "ner_output": test_y_ner}, batch_size=batch_size)

    def generate_metrics(self, model, test_x, test_y_pos, test_y_ner, pos_vocab, ners_vocab):
        predictions = model.predict(test_x)

        pred_pos = np.argmax(predictions[0], axis=-1)
        pred_ner = np.argmax(predictions[1], axis=-1)

        true_pos = np.array(test_y_pos)
        true_ner = np.array(test_y_ner)

        pred_pos_flat = pred_pos.flatten()
        true_pos_flat = true_pos.flatten()
        pred_ner_flat = pred_ner.flatten()
        true_ner_flat = true_ner.flatten()

        valid_pos_indices = true_pos_flat != len(pos_vocab)
        valid_ner_indices = true_ner_flat != len(ners_vocab)

        true_pos_filtered = true_pos_flat[valid_pos_indices]
        pred_pos_filtered = pred_pos_flat[valid_pos_indices]
        true_ner_filtered = true_ner_flat[valid_ner_indices]
        pred_ner_filtered = pred_ner_flat[valid_ner_indices]

        pos_accuracy = accuracy_score(true_pos_filtered, pred_pos_filtered)
        pos_classification_report = classification_report(true_pos_filtered, pred_pos_filtered, target_names=list(pos_vocab.keys()), zero_division=0)

        ner_accuracy = accuracy_score(true_ner_filtered, pred_ner_filtered)
        ner_classification_report = classification_report(true_ner_filtered, pred_ner_filtered, zero_division=0)

        return pos_accuracy, pos_classification_report, ner_accuracy, ner_classification_report

    def run(self):
        train_x, val_x, test_x, train_y_pos, val_y_pos, test_y_pos, train_y_ner, val_y_ner, test_y_ner, pos_vocab, ners_vocab = self.load_and_split_data()

        model = load_modell(self.model_path)

        print("Model loaded. Evaluating on test data...")

        test_loss, test_pos_loss, test_ner_loss, test_pos_accuracy, test_ner_accuracy = self.evaluate_model(model, test_x, test_y_pos, test_y_ner)
        print(f"Test Loss: {test_loss}")
        print(f"Test POS Loss: {test_pos_loss}")
        print(f"Test NER Loss: {test_ner_loss}")
        print(f"Test POS Accuracy: {test_pos_accuracy}")
        print(f"Test NER Accuracy: {test_ner_accuracy}")

        pos_accuracy, pos_classification_report, ner_accuracy, ner_classification_report = self.generate_metrics(model, test_x, test_y_pos, test_y_ner, pos_vocab, ners_vocab)

        print("POS Tagging Metrics:")
        print(f"Accuracy: {pos_accuracy}")
        print(f"\nClassification Report:\n{pos_classification_report}")

        print("NER Tagging Metrics:")
        print(f"Accuracy: {ner_accuracy}")
        print(f"\nClassification Report:\n{ner_classification_report}")

        save_metrics(self.report_path, pos_accuracy, pos_classification_report, ner_accuracy, ner_classification_report)

        print(f"Metrics have been saved to '{self.report_path}'.")

if __name__ == "__main__":
    model_path = '../notebooks/models_evaluation/models/base_model.h5'
    report_path = '../reports/final_score_custom.txt'
    PROCESSED_DATA_DIR = "../notebooks/analysis-preprocessing/processed_data"
    trainer = ModelEvaluator(model_path, report_path, PROCESSED_DATA_DIR)
    trainer.run()
