import torch
from torch.autograd import Variable
from hw3.hw3_part1 import TextNet
import os
import pdb


def generation(inp, forward):
    """
    Generate a sequence of words given a starting sequence.
    Load your model before generating words.
    :param inp: Initial sequence of words (batch size, length)
    :param forward: number of additional words to generate
    :return: generated words (batch size, forward)
    """

    # load model
    model = TextNet(dictionary_size=33278, embedding_dim=400, hidden_dim=1150)
    model_file = os.path.abspath(os.path.join(__file__, "../hw3_part1_1.pt"))
    model.load_state_dict(
        torch.load(
            model_file,
            map_location=lambda storage,
            loc: storage))
    model.eval()

    # generate sequence
    inp = Variable(torch.LongTensor(inp.tolist()))
    out = model.forward(inp.t(), forward=forward, stochastic=True)
    out = torch.max(out, dim=2)[1]

    return out.data.numpy().T
