from utils.model_utils import load_data, split_validation_set, save_metrics
import numpy as np
import random
import tensorflow as tf
from keras.layers import Dense, Embedding, LSTM, Input
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import backend as K


class ModelTrainer:
    def __init__(self, model_path, PROCESSED_DATA_DIR, max_len=25, seed=42):
        self.model_path = model_path
        self.PROCESSED_DATA_DIR = PROCESSED_DATA_DIR
        self.max_len = max_len
        self.seed = seed
        self.set_seed()
        self.train_x = None
        self.val_x = None
        self.test_x = None
        self.train_y_pos = None
        self.val_y_pos = None
        self.test_y_pos = None
        self.train_y_ner = None
        self.val_y_ner = None
        self.test_y_ner = None
        self.words_vocab = None
        self.pos_vocab = None
        self.ners_vocab = None

    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def load_and_split_data(self):
        self.train_x, self.test_x, self.train_y_pos, self.test_y_pos, self.train_y_ner, self.test_y_ner, \
            self.words_vocab, self.pos_vocab, self.ners_vocab, words, poss, ners = load_data(self.PROCESSED_DATA_DIR)
        self.val_x, self.test_x, self.val_y_pos, self.test_y_pos, self.val_y_ner, self.test_y_ner = \
            split_validation_set(self.test_x, self.test_y_pos, self.test_y_ner)

    def model_trainer(self):
        K.clear_session()
        nbr_words = len(self.words_vocab)
        nbr_pos = len(self.pos_vocab)
        nbr_ners = len(self.ners_vocab)

        input_layer = Input(shape=(self.max_len,))
        embedding_layer = Embedding(input_dim=nbr_words, output_dim=25, input_length=self.max_len)(input_layer)
        lstm_layer = LSTM(units=100, activation='tanh', return_sequences=True, recurrent_dropout=0.1)(embedding_layer)

        pos_output = Dense(nbr_pos, activation='softmax', name='pos_output')(lstm_layer)
        ner_output = Dense(nbr_ners, activation='softmax', name='ner_output')(lstm_layer)

        model = Model(inputs=input_layer, outputs=[pos_output, ner_output])

        model.compile(optimizer="adam",
                      loss={"pos_output": "sparse_categorical_crossentropy",
                            "ner_output": "sparse_categorical_crossentropy"},
                      metrics={"pos_output": "accuracy", "ner_output": "accuracy"})

        print(model.summary())

        early_stopping = EarlyStopping(monitor='val_ner_output_accuracy', min_delta=0, patience=10, verbose=1,
                                       mode='max')

        history = model.fit(self.train_x,
                            {"pos_output": self.train_y_pos, "ner_output": self.train_y_ner},
                            validation_data=(self.val_x, {"pos_output": self.val_y_pos, "ner_output": self.val_y_ner}),
                            batch_size=32,
                            epochs=100,
                            callbacks=[early_stopping])

        model.save(self.model_path)
        print(f"Model has been saved to '{self.model_path}'.")

    def run(self):
        self.load_and_split_data()
        self.model_trainer()


if __name__ == "__main__":
    model_path = '../notebooks/models_evaluation/models/custom_data_model.h5'
    PROCESSED_DATA_DIR = "../notebooks/analysis-preprocessing/processed_data"

    trainer = ModelTrainer(model_path, PROCESSED_DATA_DIR)
    trainer.run()
