# Shakespeare Text Generator

This project demonstrates a text generation model based on **LSTM (Long Short-Term Memory)** using TensorFlow and Keras. The dataset used for training is a corpus of Shakespeare's works, and the goal is to generate text that mimics Shakespeare's style.

---

## Features

1. **Character-Level Language Model**: Trained to predict the next character in a sequence.
2. **Custom Text Generation**: Generate Shakespeare-like text with adjustable creativity using a temperature-based sampling mechanism.
3. **Pre-trained Model Loading**: Use a saved model to generate text without retraining.

---

## Requirements

- Python 3.6 or later
- Libraries:
  - `numpy`
  - `tensorflow` (Tested with TensorFlow 2.x)

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. **Train the Model**
Uncomment the training section in the script to train the LSTM model on the Shakespeare dataset:
```python
# Train the model
model.fit(x, y, batch_size=256, epochs=4)

# Save the trained model
model.save("shakespeare.keras")
```

### 2. **Generate Text**
Load the pre-trained model and generate text:
```python
seed_text = "To be or not to be,"
output_length = 200
temperature = 0.7  # Adjust creativity (higher = more creative, lower = more conservative)
print(generate_text(seed_text, output_length, temperature))
```

---

## Temperature Sampling

- **Temperature**: Controls the randomness in the text generation.
  - Lower values (e.g., `0.3`) make the model more conservative.
  - Higher values (e.g., `1.0`) make the model more creative and diverse.

Example:
```python
temp_values = [0.3, 0.5, 0.7, 0.9, 1.0]
for t in temp_values:
    print(f"--- Temperature: {t} ---")
    print(generate_text("Shall I compare thee", 200, t))
```

---

## Example Output

Seed: `"Am I chines?
"`

Generated text (temperature = 0.7):
```
Am I chines?
thy hast thy more the would thee so fare,
thee shall here in heaven, thee to the stars,
and to the fear to thee might stand in thee.

```

---

## Folder Structure

```
root/
├── README.md               # Project documentation
├── shakespeare_generator.py # Main script
└── requirements.txt        # Required Python libraries
```

---

## Notes

- **Pre-trained Model**: If training is skipped, ensure the `shakespeare.keras` file is present in the working directory.
- **Custom Dataset**: Replace the Shakespeare dataset with your own text to train a custom language model.

Happy text generating!
