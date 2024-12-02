# Importing Modules
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import RMSprop

# Downloading and Preparing Data
filepath = tf.keras.utils.get_file(
    "shakespeare.txt", 
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt"
)

text = open(filepath, "rb").read().decode(encoding="utf-8").lower()  # Lower case for consistency
chars = sorted(set(text))  # Unique characters in the text

# Character-to-index and index-to-character mappings
char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}

# Create input-output sequences
max_len = 40  # Sequence length
step = 3      # Step size for creating sequences

sentences = []
next_chars = []

for i in range(0, len(text) - max_len, step):
    sentences.append(text[i: i + max_len])
    next_chars.append(text[i + max_len])

# One-hot encoding of input and output sequences
x = np.zeros((len(sentences), max_len, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_index[char]] = 1
    y[i, char_to_index[next_chars[i]]] = 1

"""
# Uncomment this section to train the model and save it
# Build the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(max_len, len(chars))))
model.add(Dense(len(chars), activation="softmax"))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)

# Train the model
model.fit(x, y, batch_size=256, epochs=4)

# Save the trained model
model.save("shakesspeare.keras")
"""

# Load the trained model
model = tf.keras.models.load_model(r"\Shakespeare AI\shakesspeare.keras")

# Temperature sampling function for diversity in predictions
def sample_with_temperature(predictions, temperature=1.0):
    predictions = np.log(predictions + 1e-10) / temperature
    probabilities = np.exp(predictions) / np.sum(np.exp(predictions))
    return np.random.choice(len(probabilities), p=probabilities)

# Generate text function
def generate_text(seed_text, output_length, temperature=1.0):
    generated_text = seed_text
    if len(seed_text) < max_len:
        seed_text = seed_text.rjust(max_len)
    elif len(seed_text) > max_len:
        seed_text = seed_text[-max_len:]
    
    for _ in range(output_length):
        # Prepare input sequence
        input_seq = np.zeros((1, max_len, len(chars)))
        for t, char in enumerate(seed_text):
            input_seq[0, t, char_to_index.get(char, 0)] = 1.0
        
        # Predict the next character probabilities
        predictions = model.predict(input_seq, verbose=0)[0]
        next_index = sample_with_temperature(predictions, temperature)
        next_char = index_to_char[next_index]
        
        # Append to generated text and update seed
        generated_text += next_char
        seed_text = seed_text[1:] + next_char
    
    return generated_text

# Example Usage
seed = "Am i chines?\n"
output_length = 200  # Number of characters to generate
temp = [0.3, 0.5, 0.7 , 0.9 , 1 ]
for t in temp:
    print(f"_____________________{t}_____________________")
    print(generate_text(seed, output_length, temperature= t))