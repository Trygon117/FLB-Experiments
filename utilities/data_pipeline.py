import torch
import urllib.request
import json
import os

def load_shakespeare():
    """
    Loads the Shakespeare dataset from Project Gutenberg.
    """
    # Download the text file from Project Gutenberg
    url = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urllib.request.urlopen(req) as response:
        raw_text = response.read().decode("utf-8")

    # Strip the Project Gutenberg header and footer
    start_idx = raw_text.find("I. A SCANDAL IN BOHEMIA")
    end_idx = raw_text.find("End of the Project Gutenberg")
    text = raw_text[start_idx:end_idx].strip()

    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)

    return text, vocab, vocab_size

def encode(string, vocab):
    char_to_id = {ch: i for i, ch in enumerate(vocab)}
    return [char_to_id[ch] for ch in string]

def decode(indices, vocab):
    id_to_char = {i: ch for i, ch in enumerate(vocab)}
    return "".join([id_to_char[idx] for idx in indices])

def get_batch(data, batch_size=16, seq_len=256):
    # Grab longer text chunks so the window has room to slide
    max_start = len(data) - seq_len
    start_indices = torch.randint(0, max_start, (batch_size,))
    
    sequences = torch.stack([data[i:i + seq_len] for i in start_indices])
    return sequences