from utils.preprocessing_utils import load_data, add_sentence_numbers, get_vocabulary, combine_word_ner_pos, \
    prepare_data, save_data
from sklearn.model_selection import train_test_split


class Preprocessor:
    def __init__(self, file_path, output_dir, max_len=25, test_size=0.2, random_state=42):
        self.file_path = file_path
        self.output_dir = output_dir
        self.max_len = max_len
        self.test_size = test_size
        self.random_state = random_state

    def run(self):
        data = load_data(self.file_path)
        data = add_sentence_numbers(data)
        words, poss, ners, words_vocab, pos_vocab, ners_vocab = get_vocabulary(data)
        combination = combine_word_ner_pos(data)
        x, y_pos, y_ner = prepare_data(combination, words_vocab, pos_vocab, ners_vocab, self.max_len)

        train_x, test_x, train_y_pos, test_y_pos, train_y_ner, test_y_ner = train_test_split(
            x, y_pos, y_ner, test_size=self.test_size, random_state=self.random_state)

        save_data(train_x, test_x, train_y_pos, test_y_pos, train_y_ner, test_y_ner,
                  words_vocab, pos_vocab, ners_vocab, words, poss, ners, self.output_dir)


if __name__ == "__main__":
    file_path = '../dataset/data.tsv'
    output_dir = '../notebooks/analysis-preprocessing/processed_data'
    preprocessor = Preprocessor(file_path, output_dir)
    preprocessor.run()
    print("Data Preprocessing is completed.")
