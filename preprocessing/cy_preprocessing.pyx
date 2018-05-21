import random as rand
import sys
import numpy as np
cimport numpy as np
cimport cython

cdef __mod(encoder, get_feature, word_list, data_size, window_size=5):
    window_sentence = []
    train_data = []
    teach_data = []
    rand_num = rand.randint(0, len(word_list) - data_size + 1)

    for char in word_list[rand_num: rand_num + data_size]:
        sys.stdout.write("\r now:(%d/%d)" % (len(train_data), data_size))
        sys.stdout.flush()
        feature = get_feature.char2feature(char, encoder)
        feature = feature.reshape(4 * 4 * 8)
        window_sentence.append(feature)
        if len(window_sentence) == window_size + 1:
            train_data.append(window_sentence[:-1])
            teach_data.append(window_sentence[1:])
            window_sentence = window_sentence[1:]
    print("")
    train_data = np.array(train_data)
    teach_data = np.array(teach_data)

    print("train shape:", train_data.shape)
    print("teach shape:", teach_data.shape)
    return train_data, teach_data


def cy_mod_make_train_data(encoder, get_feature, word_list, data_size, window_size=5):
    return __mod(encoder, get_feature, word_list, data_size, window_size=5)
