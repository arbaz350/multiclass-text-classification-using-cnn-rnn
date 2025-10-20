import json, pickle, os
import numpy as np
import tensorflow as tf
from app.utils import clean_str, pad_sentences, map_word_to_index_list

class Predictor:
    def __init__(self, trained_dir):
        if not trained_dir.endswith("/"):
            trained_dir += "/"
        self.params = json.load(open(trained_dir + "trained_parameters.json"))
        self.words_index = json.load(open(trained_dir + "words_index.json"))
        self.labels = json.load(open(trained_dir + "labels.json"))
        with open(trained_dir + "embeddings.pickle", "rb") as f:
            self.emb = np.array(pickle.load(f), dtype=np.float32)
        self.model = tf.keras.models.load_model(trained_dir + "best_model_tf2.keras", compile=False)

    def predict(self, texts):
        examples = [clean_str(t).split() for t in texts]
        x_ = pad_sentences(examples, forced_sequence_length=self.params["sequence_length"])
        x_ = map_word_to_index_list(x_, self.words_index)
        x_test = np.array(x_)
        logits = self.model.predict(x_test, batch_size=self.params["batch_size"])
        preds = np.argmax(logits, axis=1)
        pred_labels = [self.labels[p] for p in preds]
        return pred_labels
