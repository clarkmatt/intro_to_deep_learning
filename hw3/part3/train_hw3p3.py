import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from warpctc_pytorch import CTCLoss
from ctcdecode import CTCBeamDecoder
import Levenshtein as L
import numpy as np
import argparse
import pdb

gpu_dev = 0

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

class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, input_, dropout):
        if not self.training:
            return input_
        mask = torch.Tensor(1, input_.size(1), input_.size(2)).fill_(1-dropout).bernoulli()
        mask = Variable(mask/(1-dropout), requires_grad=False)
        if torch.cuda.is_available():
            mask = mask.cuda()
        mask = mask.expand_as(input_)
        return mask * input_

class WSJNet(nn.Module):
    def __init__(self, hidden_dim):
        super(WSJNet, self).__init__()
        self.locked_dropout = LockedDropout()
        #self.dropout = nn.Dropout(p=0.2)
        self.rnns = nn.ModuleList([
            nn.LSTM(input_size=40, hidden_size=hidden_dim, bidirectional=True),
            nn.LSTM(input_size=2*hidden_dim, hidden_size=hidden_dim, bidirectional=True),
            nn.LSTM(input_size=2*hidden_dim, hidden_size=hidden_dim, bidirectional=True)])
        #self.lstm = nn.LSTM(input_size=40, hidden_size=hidden_dim, num_layers=3, bidirectional=True)
        self.projection = nn.Linear(in_features=int(hidden_dim*2), out_features=139)

    def forward(self, input_, forward=0, stochastic=False):
        x = input_
        states = []
        for idx, rnn in enumerate(self.rnns):
            x, state = rnn(x)
            states.append(state)


            #locked dropout between rnn layers
            if idx < len(self.rnns):
                x, lengths  = pad_packed_sequence(x)
                x = self.locked_dropout(x, 0.2)
                x = pack_padded_sequence(x, lengths)
        #x, state = self.lstm(x)

        output, lengths  = pad_packed_sequence(x)
        logits = self.projection(output)
        return logits, lengths 

def test_dataset(dataloader, criterion):
    total_loss = 0.0
    correct = 0
    for idx, (data, phonemes, num_phonemes) in enumerate(dataloader):
        model.eval()
       
        #Put on gpu if available
        phonemes = Variable(torch.IntTensor(phonemes))
        num_phonemes = Variable(torch.IntTensor(num_phonemes))
        if torch.cuda.is_available():
            phonemes = phonemes.cuda(gpu_dev)
            num_phonemes = num_phonemes.cuda(gpu_dev)

        logits, seq_lengths = model(data)

        seq_lengths = Variable(torch.IntTensor(seq_lengths))
        if torch.cuda.is_available():
            seq_lengths = seq_lengths.cuda(gpu_dev)

        #Get our loss and calculate the gradients
        loss = criterion(logits, phonemes.cpu(), seq_lengths.cpu(), num_phonemes.cpu())
        total_loss += loss.data[0]/dataloader.batch_size

    average_loss = total_loss/idx # average loss
    error = 1-correct # total error

    return average_loss, 1-error

if __name__=="__main__":
    #Parse input args
    parser = argparse.ArgumentParser(description='Get Network Hyperparameters.')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=512)
    #parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=30.0)
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-6)
    args = parser.parse_args()
    print(args, flush=True)

    ## Load data
    #root = '/Users/matt/.kaggle/competitions/11785-hw3p3/'
    root = ''
    train_data = np.load(root+"train.npy")
    train_phonemes = np.load(root+"train_subphonemes.npy")+1
    val_data = np.load(root+"dev.npy")
    val_phonemes = np.load(root+"dev_subphonemes.npy")+1

    ## Initialize DataLoaders
    batch_size = 16
    train_loader = WSJDataLoader(train_data, train_phonemes, batch_size=batch_size)
    val_loader = WSJDataLoader(val_data, val_phonemes, batch_size=batch_size)

    ## Initialize decoder to map model output to predicted string
    #from phoneme_list import PHONEME_MAP
    #label_map = [' '] + PHONEME_MAP
    #decoder = CTCBeamDecoder(labels=label_map, blank_id=0)

    ## Initialize network
    model_file = "part3_bidirectional.pt"
    model = WSJNet(args.hidden_dim)
    if torch.cuda.is_available():
        model.cuda(gpu_dev)

    #optimizer = torch.optim.SGD(model.parameters(), lr=30.0, weight_decay=1e-6)
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    criterion = CTCLoss()

    best_val_loss = float('inf')
    running_loss = 0.0
    n_epochs = 50
    no_improvement_epochs = 0
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

            logits, seq_lengths = model(data)

            seq_lengths = Variable(torch.IntTensor(seq_lengths))

            #Get our loss and calculate the gradients
            loss = criterion(logits, phonemes.cpu(), seq_lengths.cpu(), num_phonemes.cpu())
            running_loss += loss.data[0]/batch_size

            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optimizer.step()

        val_loss, _ = test_dataset(val_loader, criterion)
        print("epoch: {}, learning_rate: {}, training_loss: {}, val_loss: {}".format(e, learning_rate, running_loss/idx, val_loss))
        running_loss = 0.0
        if val_loss < best_val_loss:
            try:
                os.remove(model_file)
            except BaseException:
                pass
            print("saving new best model: %s" % model_file, flush=True)
            torch.save(model.state_dict(), model_file)
            best_val_loss = val_loss
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs > 3:
                learning_rate /= 10
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
                no_improvement_epochs = 0
