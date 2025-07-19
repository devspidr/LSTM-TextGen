LSTM Text Generation

A PyTorch-based Long Short-Term Memory (LSTM) network for generating natural language text.
The model is trained on a sample text corpus (alice.txt), learns sequential patterns, and generates new, stylistically similar text.


Table of Contents

    Overview

    Features

    Project Structure

    How It Works

    Requirements

    Usage

    Training Details

    Results

    Future Work

    License


Overview

This project demonstrates how to build a word-level LSTM text generator in PyTorch.
Key concepts covered:

    Word tokenization and vocabulary building

    Embedding words into dense vectors

    Sequence modeling with stacked LSTM layers

    Truncated Backpropagation Through Time (TBPTT)

    Gradient clipping to avoid exploding gradients

The model reads an input text corpus, learns language patterns, and then generates new text by predicting the next word in a sequence.
Features

    Word-level text generation

    Custom dictionary creation (word2idx and idx2word)

    Adjustable hyperparameters: embedding size, LSTM layers, hidden size, and timesteps

    Truncated backpropagation for efficient training

    Gradient clipping to stabilize learning

    Outputs generated text into results.txt


Project Structure
.
├── alice.txt              # Training corpus
├── results.txt            # Generated text output
├── Text_Embedding.ipynb   # Jupyter notebook for training & testing
├── lstm_text_gen.py       # Main training script (if saved as .py)
└── README.md              # Project documentation


How It Works
1. Data Preparation

    TextProcess class reads alice.txt, tokenizes it into words, and adds a special <eos> token.

    Builds a vocabulary mapping (word2idx and idx2word).

    Converts text into a 1D tensor of word indices.

    Reshapes into (batch_size × num_batches) for training.

2. Model Architecture

    Embedding Layer: Converts word indices into dense vectors (embed_size = 300).

    LSTM Layers: Two stacked layers (hidden_size = 1024, num_layers = 2) for sequence modeling.

    Fully Connected Layer: Maps LSTM outputs to the vocabulary size for next-word prediction.

3. Training

    Loss: CrossEntropyLoss

    Optimizer: Adam (learning_rate = 0.001)

    Sequence length (timesteps = 50) using Truncated BPTT

    Gradient clipping (clip_grad_norm) prevents exploding gradients.

    Model trains for 40 epochs on batches of size 32.

4. Text Generation

    Starts with a random word.

    Predicts the next word repeatedly for 500 steps.

    Samples words using torch.multinomial(probabilities).

    Saves output in results.txt.

Requirements

    Python 3.8+

    PyTorch

    NumPy

Install dependencies:
pip install torch numpy


Usage

1.Place your training corpus in the project directory as alice.txt.

2.Run training:
    python lstm_text_gen.py

3.After training, check the generated text:
    results.txt


Training Details

    Embedding Size: 300

    Hidden Size: 1024

    Number of LSTM Layers: 2

    Epochs: 40

    Batch Size: 32

    Timesteps: 50

    Learning Rate: 0.001

Optimization Notes:

    Uses TBPTT for memory efficiency.

    Gradient clipping at 0.5 prevents instability.


Results

Sample generated text:
alice was very much surprised <eos> the queen said the king <eos> and the rabbit 
hopped quickly over the hill <eos>


    Model captures the style and vocabulary of the original text.

    Generated sentences are often coherent, though not perfect due to the dataset size.

Future Work

    Train on larger corpora for improved fluency.

    Add beam search decoding for better text generation.

    Experiment with pre-trained embeddings (GloVe, Word2Vec).

    Implement character-level LSTM for finer control.

  Author:
      SOUNDAR BALAJI J
