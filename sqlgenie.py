
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split # type: ignore
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

# --- 1. Dataset ---
# We'll need a dataset of SQL queries and their "modernized" or "enhanced" versions.
# For this example, we'll create a small dummy dataset.
# In a real-world scenario, you would need a large and high-quality dataset.

raw_data = {
    "input_sql": [
        "SELECT * FROM users WHERE age > 30",
        "SELECT name, email FROM customers WHERE country = 'USA'",
        "SELECT COUNT(*) FROM orders",
        "SELECT p.name, c.name FROM products p JOIN categories c ON p.category_id = c.id"
    ],
    "target_sql": [
        "SELECT * FROM users WHERE age > 30 AND is_active = 1",
        "SELECT name, email FROM customers WHERE country = 'USA' ORDER BY name",
        "SELECT COUNT(id) FROM orders",
        "SELECT p.name, c.name FROM products AS p INNER JOIN categories AS c ON p.category_id = c.id"
    ]
}

df = pd.DataFrame(raw_data)

# --- 2. Preprocessing ---

# Tokenize the input and target SQL queries
input_tokenizer = Tokenizer(filters='', lower=False, split=' ')
input_tokenizer.fit_on_texts(df['input_sql'])

target_tokenizer = Tokenizer(filters='', lower=False, split=' ')
target_tokenizer.fit_on_texts(df['target_sql'])

# Convert text to sequences of integers
input_sequences = input_tokenizer.texts_to_sequences(df['input_sql'])
target_sequences = target_tokenizer.texts_to_sequences(df['target_sql'])

# Pad the sequences to ensure they are all the same length
max_input_len = max(len(s) for s in input_sequences)
max_target_len = max(len(s) for s in target_sequences)

encoder_input_data = pad_sequences(input_sequences, maxlen=max_input_len, padding='post')
decoder_input_data = pad_sequences(target_sequences, maxlen=max_target_len, padding='post')

# For the decoder output, we need to one-hot encode it
decoder_target_data = np.zeros(
    (len(target_sequences), max_target_len, len(target_tokenizer.word_index) + 1),
    dtype='float32'
)

for i, seq in enumerate(target_sequences):
    for t, word_index in enumerate(seq):
        if word_index > 0:
            decoder_target_data[i, t, word_index] = 1.0


# --- 3. Model ---
# We'll use a sequence-to-sequence (Seq2Seq) model with LSTMs.

embedding_dim = 256
latent_dim = 256

# Encoder
encoder_inputs = Input(shape=(max_input_len,))
encoder_embedding = Embedding(len(input_tokenizer.word_index) + 1, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(max_target_len,))
decoder_embedding_layer = Embedding(len(target_tokenizer.word_index) + 1, embedding_dim)
decoder_embedding = decoder_embedding_layer(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(len(target_tokenizer.word_index) + 1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# --- 4. Training ---
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=2,
          epochs=50,
          validation_split=0.2)


# --- 5. Inference ---
# We'll create a separate inference model to generate predictions.

# Encoder model
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder model
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_inputs_single = Input(shape=(1,))
decoder_embedding_single = decoder_embedding_layer(decoder_inputs_single)
decoder_outputs_single, h, c = decoder_lstm(decoder_embedding_single, initial_state=decoder_states_inputs)
decoder_states_single = [h, c]
decoder_outputs_single = decoder_dense(decoder_outputs_single)

decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs_single] + decoder_states_single
)


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))
    # Populate the first character of target sequence with the start character.
    # NOTE: We don't have a start character in this simple example.

    # Sampling loop for a batch of sequences
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = None
        for word, index in target_tokenizer.word_index.items():
            if index == sampled_token_index:
                sampled_word = word
                break

        if sampled_word is not None:
            decoded_sentence += sampled_word + ' '

        # Exit condition: either hit max length or find stop character.
        if (sampled_word is None or
           len(decoded_sentence.split()) > max_target_len):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence.strip()


# --- 6. Example Usage ---
for i in range(len(df)):
    input_seq = encoder_input_data[i: i + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input SQL:', df['input_sql'][i])
    print('Decoded SQL:', decoded_sentence)
