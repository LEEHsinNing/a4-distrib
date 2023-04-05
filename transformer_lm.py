# models.py

import numpy as np
import torch
import torch.nn as nn
import random
from transformer import PositionalEncoding
from torch import optim

class Transformer(nn.Module):
    def __init__(self, vocab_index, d_model, d_internal, num_classes, num_layers):
        super().__init__()
        self.indexer = vocab_index
        self.emb = nn.Embedding(len(vocab_index), d_model)
        self.pos = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, 1, d_internal)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.decoder = nn.Linear(d_model, num_classes)
        
        self.logsoftmax = nn.LogSoftmax(dim = -1)

    def forward(self, indices):
        """
        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x27 matrix)
        """
        src = self.emb(indices)
        src = self.pos(src)

        mask = (torch.triu(torch.ones(len(indices), len(indices))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        src = self.transformer_encoder(src, mask = mask)

        return self.logsoftmax(self.decoder(src))
    

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
    def __init__(self, vocab_index, d_model, d_internal, num_classes, num_layers):
        self.model = Transformer(vocab_index, d_model, d_internal, num_classes, num_layers)
        self.indexer = vocab_index

    def get_next_char_log_probs(self, context):
        self.model.eval()
        space = torch.LongTensor([self.indexer.index_of(' ')])
        indices = self.char_to_idx(context)
        indices  = torch.cat((space, indices), 0).long()

        output = self.model(indices) # 20x27

        with torch.no_grad():
            out = output[-1].numpy()
        return out # 27

    def get_log_prob_sequence(self, next_chars, context):
        self.model.eval()
        
        window_size = 20
        sequence = context + next_chars

        space = torch.LongTensor([self.indexer.index_of(' ')])
        indices = self.char_to_idx(sequence[:-1])
        sequence_idx  = torch.cat((space, indices), 0).long()

        output_idx = self.char_to_idx(sequence)
        
        seq_len = len(sequence)
        context_len = len(context)
        processed_len = 0

        output_tensor = torch.Tensor()
        while processed_len < seq_len:
            if seq_len - processed_len < window_size:
                indices = sequence_idx[processed_len:]
            else:
                indices = sequence_idx[processed_len:processed_len+window_size]
            probs = self.model(indices)
            if processed_len < context_len:
                output_tensor = torch.cat((output_tensor, probs[context_len:]), 0)
            else:
                output_tensor = torch.cat((output_tensor, probs), 0)
            processed_len += len(indices)
        
        log_probs = 0.0
        i = 0
        for c in output_idx[context_len:]:
            log_probs += output_tensor[i][c].item()
            i += 1
        
        return log_probs

    def char_to_idx(self, exs):
        output = np.array([self.indexer.index_of(ci) for ci in exs])
        return torch.from_numpy(output)

def char_to_idx(vocab_index, exs):
    output = np.array([vocab_index.index_of(ci) for ci in exs])
    return torch.from_numpy(output)

def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """

    num_positions = 20
    d_model = 100
    d_internal = d_model*8
    num_classes = len(vocab_index)
    num_layers = 2

    train_exs = np.asarray(list(train_text)).reshape(-1, num_positions)

    NeuralLM = NeuralLanguageModel(vocab_index, d_model, d_internal, num_classes, num_layers)

    model =  NeuralLM.model
    model.zero_grad()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    space = torch.LongTensor([vocab_index.index_of(' ')])

    num_epochs = 7
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        random.seed(t)
        ex_idxs = [i for i in range(0, len(train_exs))]
        random.shuffle(ex_idxs)
        loss_fcn = nn.NLLLoss()
        for ex_idx in ex_idxs:
            model.zero_grad()

            train_idx = char_to_idx(vocab_index, train_exs[ex_idx])
            # concat start of sequence at front
            model_input = torch.cat((space, train_idx), 0)
            probs = model.forward(model_input[:-1])

            # gold sequence to index
            output_idx = np.array(train_idx)
            output_tensor = torch.LongTensor(output_idx)
            output_tensor = torch.flatten(output_tensor)

            loss = loss_fcn(probs, output_tensor)  # TODO: Run forward and compute loss
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
        print("Epoch:", t+1, ", Total loss:", loss_this_epoch)
    model.eval()
    return NeuralLM
