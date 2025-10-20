import re
import numpy as np

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def pad_sentences(sentences, padding_word="<PAD/>", forced_sequence_length=None):
    sequence_length = forced_sequence_length or max(len(x) for x in sentences)
    padded = []
    for sentence in sentences:
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded.append(new_sentence)
    return padded

def map_word_to_index_list(examples, words_index):
    x = []
    for sentence in examples:
        x.append([words_index.get(w, 0) for w in sentence])
    return x
