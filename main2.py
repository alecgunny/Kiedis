from __future__ import print_function
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from get_data import get_all_lyrics
from keras.models import Model
from keras.layers import Dense, Activation, Masking, Input, Dropout
from keras.layers import LSTM
from keras.utils.data_utils import get_file
import numpy as np
import random

path = 'rhcp_lyrics.txt'
if not os.path.exists(path):
	print("Compiling lyrics")
	lyrics = get_all_lyrics()
	with open('rhcp_lyrics.txt', 'w') as f:
		f.write(lyrics)

text = open(path).read().lower()
print('corpus length:', len(text))

chars = set(text)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
transform_chars = lambda x: char_indices[x]
transform_ints = lambda x: indices_char[x]
print("Transforming corpus...")
int_text = map(transform_chars, text)
vectors = np.eye(len(chars)).astype('float32')

maxlen = 40
minlen = 15
batch_size = 128
step_size = 3
batches_per_epoch = (len(text) // step_size) // batch_size + 1
def TrainGen():
    while True:
    #for _ in range(batches_per_epoch + 1):
        batch_x = np.zeros((batch_size, maxlen, len(chars)), dtype='float32')
        batch_y = np.zeros((batch_size, len(chars)), dtype='float32')
        for b in range(batch_size):
            length = np.random.randint(minlen, maxlen+1)
            idx = np.random.randint(len(text) - length)
            ints = int_text[idx: idx+length]
            batch_x[b, :length] = vectors[ints]
            batch_y[b] = vectors[int_text[idx+length]]
        yield batch_x, batch_y

# build the model: 2 stacked LSTM
print('Build model...')
input_shape = (None, len(chars))
lstm_kwargs = dict(init='glorot_uniform', forget_bias_init='one')

input = Input(input_shape, dtype='float32')
masking = Masking(0)(input)
lstm1 = LSTM(512, dropout_W=0.1, dropout_U=0.25, return_sequences=True, **lstm_kwargs)(masking)
lstm2 = LSTM(512, dropout_W=0.25, dropout_U=0.25, return_sequences=False, **lstm_kwargs)(lstm1)
dropout = Dropout(0.25)(lstm2)
dense = Dense(len(chars), init='glorot_uniform', activation='softmax')(dropout)
model = Model(input=input, output=dense)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit_generator(TrainGen(), samples_per_epoch=batch_size*batches_per_epoch, nb_epoch=1)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = list(int_text[start_index: start_index + maxlen])
        generated += ''.join(map(transform_ints, sentence))
        print('----- Generating with seed: "' + generated + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = vectors[sentence[-maxlen:]][np.newaxis]
            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            generated += next_char
            sentence.append(next_index)
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()