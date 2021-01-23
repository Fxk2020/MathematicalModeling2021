import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# 1.载入数据预处理
imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
# print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
# print(train_data[0])
# print(train_labels[0])

# 一个映射单词到整数索引的词典
word_index = imdb.get_word_index()

# 保留第一个索引
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# 2.构建神经网络
vocab_size = 10000  # 输入形状是用于电影评论的词汇数目（10,000 词）

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

# 3.编译和训练模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
x_val = train_data[:5000]
partial_x_train = train_data[5000:]
y_val = train_labels[:5000]
partial_y_train = train_labels[5000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)


# 从history中取出数据
def draw_history():
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    plt.subplot(1, 2, 1)
    epochs = np.arange(40)
    plt.plot(epochs, acc, 'bo', label='accuracy')
    plt.plot(epochs, val_acc, 'b', label='val_accuracy')
    plt.legend(loc='lower right')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('train and val accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='loss')
    plt.plot(epochs, val_loss, 'b', label='val_loss')
    plt.legend(loc='lower right')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('train  and val loss ')

    plt.show()


history_dict = history.history
draw_history()

# 4.评估模型
# test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
#
# print('\nTest accuracy:', test_acc)
# print('\nTest loss:', test_loss)
