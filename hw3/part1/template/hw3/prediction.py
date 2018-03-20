import torch
from torch.autograd import Variable
from hw3.hw3_part1 import TextNet
import pdb

model = TextNet(dictionary_size=33278, embedding_dim=400, hidden_dim=1150)
model.load_state_dict(torch.load("hw3/hw3_part1.pt", map_location=lambda storage, loc: storage))
model.eval()

def prediction(inp):
    """
    Input is a text sequences. Produce scores for the next word in the sequence.
    Scores should be raw logits not post-softmax activations.
    Load your model before generating predictions.
    :param inp: array of words (batch size, sequence length) [0-labels]
    :return: array of scores for the next word in each sequence (batch size, labels)
    """
    inp = Variable(torch.LongTensor(inp))
    out = model(inp, forward=0, stochastic=False)
    scores = out[:, -1, :].data.numpy()

    return scores
