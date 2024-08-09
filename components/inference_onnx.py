import numpy as np
import onnxruntime as ort
from utils.inference_utils import (
    load_vocabulary_and_labels,
    prepare_input,
    decode_predictions
)


class ModelInferenceONNX:
    def __init__(self, model_path, vocab_path, pos_path, ner_path, max_len=25):
        self.session = ort.InferenceSession(model_path)
        self.words_vocab, self.pos_vocab, self.ners_vocab = load_vocabulary_and_labels(
            vocab_path, pos_path, ner_path)
        self.max_len = max_len

    def predict(self, sentence):
        x_padded, tokens = prepare_input(
            sentence, self.words_vocab, self.max_len)
        x_padded = x_padded.astype(np.float32)
        inputs = {self.session.get_inputs()[0].name: x_padded}

        predictions = self.session.run(None, inputs)
        pos_labels, ner_labels = decode_predictions(
            predictions, self.pos_vocab, self.ners_vocab)
        return pos_labels, ner_labels, tokens

    def run_inference(self, sentence):
        pos_labels, ner_labels, tokens = self.predict(sentence)
        print("Input Sentence:", sentence)
        print("Word \t POS \t NER")
        for i in range(len(tokens)):
            print(f"{tokens[i]} \t {pos_labels[i]} \t {ner_labels[i]}")
        print()


if __name__ == "__main__":
    model_path = '../notebooks/models_evaluation/models/base_model.onnx'
    vocab_path = '../notebooks/analysis-preprocessing/processed_data/words_vocab.pkl'
    pos_path = '../notebooks/analysis-preprocessing/processed_data/pos_vocab.pkl'
    ner_path = '../notebooks/analysis-preprocessing/processed_data/ners_vocab.pkl'

    while True:
        sentence = input("One Bangla Sentence please! ")

        inference = ModelInferenceONNX(
            model_path, vocab_path, pos_path, ner_path)
        inference.run_inference(sentence)
