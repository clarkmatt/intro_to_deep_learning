import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torch.utils.data import DataLoader
from ctcdecode import CTCBeamDecoder
from warpctc_pytorch import CTCLoss
import numpy as np
import argparse
import pdb

gpu_dev =0

class WSJDataLoader(DataLoader):
    def __init__(self, dataset, phonemes, batch_size):
        self.dataset = dataset
        self.phonemes = phonemes
        self.batch_size = batch_size
        
    def __iter__(self):
        # Randomize dataset
        num_utterances = self.dataset.shape[0]
        rand_perm = np.random.permutation(num_utterances)
        data = self.dataset[rand_perm]
        phonemes = self.phonemes[rand_perm]

        pos = 0
        done = False
        while not done:

            #Get batch, this is the last iteration if position is larger than number of utterances
            seq_batch = data[pos:pos+self.batch_size]
            phoneme_batch = phonemes[pos:pos+self.batch_size]
            pos += self.batch_size
            if pos >= num_utterances: done = True
            
            ## Sort utterances by descending number of frames
            #Get sort order
            seq_lengths = np.asarray([utterance.shape[0] for utterance in seq_batch])
            sort_perm = np.argsort(seq_lengths)[::-1]

            #Sort sequence lengths and actual utterances, also store max sequence length for padding generation
            sorted_seq_lengths = seq_lengths[sort_perm]
            sorted_seq_batch = seq_batch[sort_perm]
            sorted_phoneme_batch = phoneme_batch[sort_perm]
            max_seq_len = int(sorted_seq_lengths[0])

            #Put sorted sequences into padded Tensor
            seq_tensor = torch.zeros(max_seq_len, sorted_seq_batch.shape[0], 40).float()
            for idx, (seq, seq_len) in enumerate(zip(sorted_seq_batch, sorted_seq_lengths)):
                seq_tensor[:seq_len, idx] = torch.FloatTensor(seq)

            #Get num labels/phonemes in each sequence
            num_phonemes = np.asarray([labels.shape[0] for labels in sorted_phoneme_batch])

            seq_tensor = Variable(seq_tensor)
            if torch.cuda.is_available():
                seq_tensor = seq_tensor.cuda()
            packed_data = pack_padded_sequence(seq_tensor, sorted_seq_lengths)

            # do the phonemes need to be a Tensor or Variable?
            yield packed_data, torch.IntTensor(np.concatenate(sorted_phoneme_batch)), num_phonemes


class WSJNet(nn.Module):
    def __init__(self, hidden_dim):
        super(WSJNet, self).__init__()
        self.rnns = nn.ModuleList([
            nn.LSTM(input_size=40, hidden_size=hidden_dim),
            nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim),
            nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim)])
        self.projection = nn.Linear(in_features=hidden_dim, out_features=47)

    def forward(self, input_, forward=0, stochastic=False):
        x = input_
        states = []
        for rnn in self.rnns:
            x, state = rnn(x)
            states.append(state)

        output, lengths  = pad_packed_sequence(x)
        x = self.projection(output)
        #if stochastic:
        #    gumbel = Variable(sample_gumbel(shape=x.size(), out=h.data.new()))
        #    x += gumbel
        logits = x
        return logits, lengths 

if __name__=="__main__":
    #Parse input args
    parser = argparse.ArgumentParser(description='Get Network Hyperparameters.')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=256)
    #parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=30.0)
    #parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-6)
    args = parser.parse_args()
    print(args, flush=True)

    ## Load data
    #root = '/home/matt/.kaggle/competitions/11785-hw3p2/'
    root = ''
    train_data = np.load("train.npy")
    train_phonemes = np.load(root+"train_phonemes.npy")+1
    val_data = np.load("dev.npy")
    val_phonemes = np.load(root+"dev_phonemes.npy")+1

    ## Initialize DataLoaders
    batch_size = 16
    train_loader = WSJDataLoader(train_data, train_phonemes, batch_size=batch_size)
    val_loader = WSJDataLoader(val_data, val_phonemes, batch_size=batch_size)

    ## Initialize network
    model = WSJNet(args.hidden_dim)
    if torch.cuda.is_available():
        model.cuda(gpu_dev)

    #optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = CTCLoss()

    running_loss = 0.0
    n_epochs = 30
    for e in range(n_epochs):
        for idx, (data, phonemes, num_phonemes) in enumerate(train_loader):
            #print("idx: ", idx, "data_shape: ", data.data.shape, "phonemes_shape: ", phonemes.shape, "num_phonemes: ", num_phonemes)
            model.train()
            optimizer.zero_grad()
           
            #Put on gpu if available
            phonemes = Variable(torch.IntTensor(phonemes))
            num_phonemes = Variable(torch.IntTensor(num_phonemes))
            if torch.cuda.is_available():
                phonemes = phonemes.cuda(gpu_dev)
                num_phonemes = num_phonemes.cuda(gpu_dev)

            out, seq_lengths = model(data)

            seq_lengths = Variable(torch.IntTensor(seq_lengths))
            if torch.cuda.is_available():
                seq_lengths = seq_lengths.cuda(gpu_dev)

            #Get our loss and calculate the gradients
            loss = criterion(out, phonemes.cpu(), seq_lengths.cpu(), num_phonemes.cpu())
            running_loss += loss.data[0]
            if idx % 100 == 0:
                print("iteration: {}, loss: {}".format(idx, loss.data[0]))
            loss.backward()
            #torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optimizer.step()

        print("epoch: {}, training_loss: {}".format(e, running_loss/idx))

#IMPLEMENTATION NOTES:
#Add +1 to phonemes
#Make sure output dim is 47
