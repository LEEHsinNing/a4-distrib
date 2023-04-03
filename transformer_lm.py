# models.py

import numpy as np
import torch
import torch.nn as nn
from transformer import PositionalEncoding
import math

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, d_internal, num_classes, num_layers):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, 1, d_internal)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.output = nn.Linear(d_model, num_classes)
        
        self.logsoftmax = nn.LogSoftmax(dim = -1)

    def forward(self, indices):
        """
        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        src = self.emb(indices)
        src = self.pos(src)

        mask = torch.triu(torch.ones(indices.shape[0], indices.shape[0]), diagonal=1).to(indices.device)
        mask = (mask == 0).unsqueeze(1)

        for layer in self.layers:
            src = layer(src, mask=mask)

        return self.logsoftmax(self.output(src)) # 32x10x512
    

class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, vocab_size, d_model, d_internal, num_classes, num_layers):
        self.model = Transformer(self, vocab_size, d_model, d_internal, num_classes, num_layers)

    def get_next_char_log_probs(self, context):
        self.model.eval()

    def get_log_prob_sequence(self, next_chars, context):
        self.model.eval()


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    train_exs = np.asarray(list(train_text))
    train_exs = train_exs.reshape(20, -1)

    dev_exs = np.asarray(list(dev_text))
    dev_exs = dev_exs.reshape(20, -1)
    

    raise Exception("Implement me")
