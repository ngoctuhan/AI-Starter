# NLP Models Implementation Examples

This repository contains implementations of two fundamental Natural Language Processing (NLP) models using PyTorch:
1. Continuous Bag of Words (CBOW)
2. Sequence-to-Sequence (Seq2Seq)

## 1. Continuous Bag of Words (CBOW)

### Overview
The CBOW model is part of the Word2Vec family of models that learns word embeddings by predicting a target word given its surrounding context words. This implementation includes:

- Custom dataset handling for CBOW training data
- PyTorch implementation of the CBOW architecture
- Training loop with example data
- Word similarity calculation functionality

### Key Components

```python
class CBOWDataset:  # Handles data preprocessing and loading
class CBOW:         # Main model architecture
```

### Features
- Configurable window size for context words
- Vocabulary creation and management
- Word-to-index and index-to-word mapping
- Support for saving and loading vocabulary
- Word similarity calculation using cosine similarity

### Training Data
The implementation includes a sample dataset of sentences for training. The model learns to predict words based on their context in these sentences.

### Usage Example
```python
# Create dataset
dataset = CBOWDataset(example_data, window_size=2)

# Initialize model
model = CBOW(vocab_size, embedding_dim)

# Train model
# ... training loop ...

# Find similar words
similar_words = get_similar_words('sun', model, dataset)
```

## 2. Sequence-to-Sequence (Seq2Seq)

### Overview
The Seq2Seq model is designed for sequence transformation tasks like machine translation. This implementation features:

- Encoder-Decoder architecture using LSTM
- Support for machine translation
- Teacher forcing during training
- Customizable model parameters

### Key Components

```python
class Encoder:   # Processes input sequence
class Decoder:   # Generates output sequence
class Seq2Seq:   # Combines Encoder and Decoder
```

### Features
- LSTM-based encoder and decoder
- Embedding layers for input and output sequences
- Configurable model architecture (hidden size, number of layers, etc.)
- Support for teacher forcing during training
- Translation functionality

### Training Data
The implementation includes example English-Vietnamese translation pairs for training. The model learns to translate sentences between these languages.

### Usage Example
```python
# Initialize model
encoder = Encoder(vocab_size, embedding_size, hidden_size, num_layers)
decoder = Decoder(vocab_size, embedding_size, hidden_size, num_layers)
model = Seq2Seq(encoder, decoder, device)

# Train model
# ... training loop ...

# Translate text
translated = translate(model, "The weather is nice today.", word2index, device)
```

## Requirements

- Python 3.x
- PyTorch
- NumPy

## Installation

```bash
pip install torch numpy
```

## Model Parameters

### CBOW Parameters
- Embedding dimension: 100
- Window size: 2
- Learning rate: 0.001
- Number of epochs: 100

### Seq2Seq Parameters
- Embedding size: 256
- Hidden size: 512
- Number of layers: 2
- Learning rate: 0.001
- Batch size: 8
- Number of epochs: 100

## Directory Structure

```
.
├── CBOW.ipynb        # CBOW implementation and example
├── Seq2Seq.ipynb     # Seq2Seq implementation and example
└── README.md         # This file
```

## Limitations and Future Improvements

1. CBOW:
   - Limited vocabulary size
   - No handling of rare words
   - Basic similarity calculation

2. Seq2Seq:
   - No attention mechanism
   - Limited to relatively short sequences
   - Basic teacher forcing strategy

## Contributing

Feel free to submit issues and enhancement requests!

## References

1. Mikolov, T., et al. (2013). Distributed Representations of Words and Phrases and their Compositionality
2. Sutskever, I., et al. (2014). Sequence to Sequence Learning with Neural Networks