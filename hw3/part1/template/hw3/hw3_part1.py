#Intro to Deep Learning
#Homework 3 - Part 3
#Author: Matt Clark
#Last Update: 3/8/2018

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import namedtuple
import argparse
import pdb


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x

class TextNet(nn.Module):
    def __init__(self, dictionary_size, embedding_dim, hidden_dim):
        super(TextNet, self).__init__()
        self.lockdrop = LockedDropout()
        self.embedding = nn.Embedding(num_embeddings=dictionary_size, embedding_dim=embedding_dim)
        self.rnns = nn.ModuleList([
            nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim),
            nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim),
            nn.LSTM(input_size=hidden_dim, hidden_size=embedding_dim)])
        self.projection = nn.Linear(in_features=embedding_dim, out_features=dictionary_size)

    def forward(self, input_, forward=0, stochastic=False):
        h = input_
        h = self.embedding(h)
        #h = self.lockdrop(h, 0.5)
        states = []
        for rnn in self.rnns:
            h, state = rnn(h)
            states.append(state)
        h = self.projection(h)
        if stochastic:
            gumbel = Variable(sample_gumbel(shape=h.size(), out=h.data.new()))
            h += gumbel
        logits = h
        if forward > 0:
            outputs = []
            h = torch.max(logits[:, -1:, :], dim=2)[1]
            for i in range(forward):
                h = self.embedding(h)
                for j, rnn in enumerate(self.rnns):
                    h, state = rnn(x, states[j])
                    states[j] = state
                h = self.projection(h)
                if stochastic:
                    gumbel = Variable(sample_gumbel(shape=h.size(), out=h.data.new()))
                    h += gumbel
                outputs.append(h)
                h = torch.max(h, dim=2)[1]
            logits = torch.cat([logits] + outputs, dim=1)
        return logits 

def sample_gumbel(shape, eps=1e-10, out=None):
    """
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))

class CrossEntropyLoss3D(nn.CrossEntropyLoss):
    def forward(self, input, target):
        return super(CrossEntropyLoss3D, self).forward(input.view(-1, input.size()[2]), target.view(-1))

def init_randn(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0,1)

def init_xavier(m):
    if type(m) == nn.Conv1d:
        fan_in = m.weight.size()[1]
        fan_out = m.weight.size()[0]
        std = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.normal_(0,std)

class WT2DataLoader(DataLoader):
    def __init__(self, dataset, batch_size, base_sequence_length):
        self.dataset = dataset
        self.batch_size = batch_size
        self.base_sequence_length = base_sequence_length
        
    def __iter__(self):
        # Randomly concatenate articles into one long string
        num_articles = self.dataset.shape[0]
        data = np.concatenate(self.dataset[np.random.permutation(num_articles)])

        # Truncate the data that doesn't fit
        words_per_batch = data.shape[0] // self.batch_size
        data = data[:(words_per_batch*self.batch_size)]

        # Reshape data into batches
        data = torch.LongTensor(data.tolist())
        data = data.view(self.batch_size, words_per_batch).t().contiguous()
        #if torch.cuda.is_available():
        #    data.cuda()

        pos = 0
        while True:
            #get next sequence length, stop if we're at the end of the batch
            seq_len = self.get_sequence_length()
            if (pos+seq_len+1) > (data.shape[0]):
                break

            inp = data[pos:pos+seq_len]
            target = data[pos+1:pos+seq_len+1]
            pos += seq_len
            yield inp, target
            
    def get_sequence_length(self):
        #get sequence of random length from each batch
        if np.random.binomial(1, 0.95):
            seq = self.base_sequence_length
            sequence_length = np.clip(int(np.random.normal(seq, 5)), seq-10, seq+10)
        else:
            seq = self.base_sequence_length//2
            sequence_length = np.clip(int(np.random.normal(seq, 5)), seq-10, seq+10)

        return sequence_length


def test_dataset(model, criterion, loader):
    total_loss = 0.0
    correct = 0.0
    num_samples = 0.0
    model.eval()
    for batch_idx, (inputs_, targets) in enumerate(loader):
        optimizer.zero_grad() #Torch accumulates gradients, let's zero out these for each batch

        #Load our data into Variables, put on GPU if available
        X = Variable(inputs_)
        Y = Variable(targets)
        if torch.cuda.is_available():
            X = X.cuda()
            Y = Y.cuda()

        out = model(X, forward=0, stochastic=True)

        #Get our predictions
        pred = out.data.max(2, keepdim=True)[1]
        predicted = pred.eq(Y.data.view_as(pred))
        correct += int(predicted.sum())
        num_samples += int(predicted.shape[0]*predicted.shape[1])


        #Get our loss and calculate the gradients
        loss = criterion(out, Y)

        #Track loss
        total_loss += loss.data[0]
        num_batches = batch_idx

    return 1-(correct/num_samples), total_loss/num_batches

def generate_test_results_csv(model, loader, n_members):
    predictions = torch.zeros(n_members)
    n_processed = 0
    model.eval()

    with open("my_submission.csv", 'w') as csv_file:
        csv_file.write("id,label\n")
        for data in loader:
            X = Variable(data.view(-1, int((2*k+1)*frame_size)))
            out = model(X)
            pred = out.data.max(1, keepdim=True)[1]
            n_end = n_processed + out.data.shape[0]
            predictions[n_processed:n_end] = pred
            n_processed = n_end

        # Write csv file
        [csv_file.write("%s,%s\n") % (i, label) for i, label in enumerate(predictions)]

    return predictions

# setup metric class
Metric = namedtuple('Metric', ['loss', 'train_error', 'val_loss', 'val_error'])

class Trainer():
    """ A simple training cradle
    """
    
    def __init__(self, model, optimizer, train_loader, val_loader, load_path=None):
        self.model = model
        if load_path is not None:
            self.model = torch.load(load_path)
        self.criterion = CrossEntropyLoss3D() #reduce=False to get array of sample losses
        self.optimizer = optimizer
        print(self.model, flush=True)
        
    def stop_cond(self):
        # TODO: Implement early stopping
        def deriv(ns):
            return [ns[i+1] - ns[i] for i in range(len(ns)-1)]
        val_errors = [m.val_error for m in self.metrics]
        back = val_errors[-10:]
        return sum(deriv(back)) > 0
            
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def run(self, epochs):
        print("begin training...", flush=True)
        self.metrics = []
        best_val_loss = float('inf')
        for e in range(n_epochs):
            #if self.stop_cond():
            #    return

            epoch_loss = 0
            running_loss = 0
            correct = 0
            model_file = "./hw3_part1_adam.pt"
            for batch_idx, (inputs_, targets) in enumerate(train_loader):
                #print((inputs_[1:]==targets[:-1]).all())
                #assert((inputs_[1:]==targets[:-1]).all())
                
                self.model.train() #We're training now
                self.optimizer.zero_grad() #Torch accumulates gradients, let's zero out these for each batch

                #Load our data into Variables, put on GPU if available
                X = Variable(inputs_)
                Y = Variable(targets)
                if torch.cuda.is_available():
                    X = X.cuda()
                    Y = Y.cuda()
                
                out = self.model(X, forward=0, stochastic=False)

                #Get our predictions 
                #predicted = pred.eq(Y.data.view_as(pred))
                #correct += int(predicted.sum())

                #Get our loss and calculate the gradients
                loss = self.criterion(out, Y)
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 0.25)
                self.optimizer.step()

                #Track loss
                num_batches = batch_idx
                epoch_loss += loss.data[0]
                pred = out.data.max(2, keepdim=True)[1]
                if (batch_idx % 100) == 0:
                    val_error, val_loss = test_dataset(self.model, self.criterion, val_loader)
                    print("batch_idx: {}, train_loss: {}, val_loss: {}, val_error: {}".format(batch_idx, loss.data[0], val_loss, val_error),  flush=True)
                    print(loss.data[0], flush=True)
                    print([vocab[word_id] for word_id in pred[:,0,0]])

                    #Save model if better
                    if val_loss < best_val_loss:
                        try:
                            os.remove(model_file)
                        except:
                            pass
                        print("saving new best model: %s" % model_file, flush=True)
                        self.save_model(model_file)
                        best_val_loss = val_loss

            #Report metrics for training and validation every epoch, save model if it's better than our last one.
            train_loss = epoch_loss/num_batches
            val_error, val_loss = test_dataset(self.model, self.criterion, val_loader)
            print("epoch: {}, num_batches: {}, train_loss: {}, val_loss: {}, val_error: {}".format(e, num_batches, train_loss, val_loss, val_error),  flush=True)
           
            #Save model if better
            if val_loss < best_val_loss:
                try:
                    os.remove(model_file)
                except:
                    pass
                print("saving new best model: %s" % model_file, flush=True)
                self.save_model(model_file)
                best_val_loss = val_loss

if __name__ == "__main__":

    #Parse input args
    parser = argparse.ArgumentParser(description='Get Network Hyperparameters.')
    parser.add_argument('--embedding_dim', dest='embedding_dim', type=int, default=400)
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=1150)
    parser.add_argument('--base_sequence_length', dest='base_sequence_length', type=int, default=70)
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=30.0)
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-6)
    args = parser.parse_args()
    print(args, flush=True)

    #Load the dataset and initialize DataLoaders
    train_data = np.load("dataset/wiki.train.npy")
    val_data = np.load("dataset/wiki.valid.npy")
    vocab = np.load("dataset/vocab.npy")

    dictionary_size = len(vocab)
    batch_size = 80

    #Dataloaders
#    if torch.cuda.is_available():
#        train_loader = WT2DataLoader(train_data, batch_size=batch_size, pin_memory=True)
#        val_loader = WT2DataLoader(val_data, batch_size=batch_size, pin_memory=True)
    train_loader = WT2DataLoader(train_data, batch_size, args.base_sequence_length)
    val_loader = WT2DataLoader(val_data, batch_size, args.base_sequence_length)

    #Initialize with unigram distribution trick
    flat_train_data = np.concatenate(train_data)
    unique, counts = np.unique(flat_train_data, return_counts=True)
    unigram_dist = counts/len(flat_train_data)

    def init_unigram(m):
        #Calculate unigram distribution
        
        #Smooth

        #Set bias of projection layer to smoothed unigram distribution
        smoothing = 0.1
        vocabsize = len(vocab)
        if type(m) == nn.Linear:
            m.bias.data = torch.Tensor(np.log((unigram_dist*(1.-smoothing))+(smoothing/vocabsize)).tolist())

    #Initialize the network
    net = TextNet(dictionary_size, args.embedding_dim, args.hidden_dim)
    net.apply(init_unigram)
    if torch.cuda.is_available():
        net.cuda()

    #Train the model
    n_epochs = 50
    #optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(net.parameters())
    #optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    my_trainer = Trainer(net, optimizer, train_loader=train_loader, val_loader=val_loader)
    my_trainer.run(n_epochs)
    np.save('my_metrics', np.asarray(my_trainer.metrics))

    #training_plot(my_trainer.metrics)
     

    ### TEST ###
    my_model = TextNet()
    generate_test_results_csv(my_model, test_loader, test_size)
