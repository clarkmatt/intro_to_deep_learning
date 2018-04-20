#train_las.py
#Author: Matt Clark
#Start Date: 4/10/18
#Last Edit: 4/10/18
#
#This program trains a speech utterances to characters transcription system 
#based on the Listen, Attend, and Spell model (https://arxiv.org/pdf/1508.01211.pdf)

# Imports
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import _use_shared_memory
import torch.nn.functional as F
import os
import numpy as np
import argparse
import pdb

#NOTE: Use vocab drawn from dataset?
vocab = np.asarray(["<sos>"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.'+-_") + ["<unk>"] + ["<eos>"]) # Possible characters in a sequence
gpu_dev = 0 # Which GPU device to run on


class WSJDataset(Dataset):
    def __init__(self, set_, root=os.getcwd()):
        assert set_ in ['train', 'val', 'test']
        self.root = root
        self.set_ = set_

        #Load indicated dataset
        if self.set_ is "train":
            self.data = np.load(self.root+"/train.npy")
            self.labels = np.load(self.root+"/train_transcripts.npy")
        elif self.set_ is "val":
            self.data = np.load(self.root+"/dev.npy")
            self.labels = np.load(self.root+"/dev_transcripts.npy")
        elif self.set_ is "test":
            self.data = np.load(self.root+"/test.npy")

        # Convert transcript to integer labels
        int_labels = []
        unk_int_label = np.where("<unk>" == vocab)[0][0]
        start_int_label = np.where("<sos>" == vocab)[0][0]
        end_int_label = np.where("<eos>" == vocab)[0][0]
        for i, transcript in enumerate(self.labels):
            int_label = [np.where(char==vocab)[0][0] if (char in vocab) else unk_int_label for char in transcript] + [end_int_label]
            #NOTE: Add start token?
            #int_label = [start_int_label] + [np.where(char==vocab)[0][0] if (char in vocab) else unk_int_label for char in transcript] + [end_int_label]
            int_labels.append(np.asarray(int_label))
        self.labels = np.asarray(int_labels, dtype=object)

        return

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def wsj_collate(batch):
    
    batch = np.asarray(batch)

    ## Create padded utterance and transcript tensors sorted by descending number of frames in utterances
    #Get utterance and transcript lengths
    utterance_lengths = np.asarray([len(utterance) for utterance in batch[:,0]])
    transcript_lengths = np.asarray([len(transcript) for transcript in batch[:,1]])
    max_utterance_length = int(np.max(utterance_lengths))
    max_transcript_length = int(np.max(transcript_lengths))

    #Get sort order and sort batch by descending utterance length
    sort_perm = np.argsort(utterance_lengths)[::-1]
    batch = batch[sort_perm,:]
    utterance_lengths = utterance_lengths[sort_perm]
    transcript_lengths = transcript_lengths[sort_perm]


    utterance_tensor = torch.zeros(max_utterance_length, batch.shape[0], 40).float()
    transcript_tensor = torch.zeros(max_transcript_length, batch.shape[0]).long() 
    for idx, batch_item in enumerate(batch):
        utterance_tensor[:batch_item[0].shape[0], idx] = torch.FloatTensor(batch_item[0])
        transcript_tensor[:batch_item[1].shape[0], idx] = torch.FloatTensor(batch_item[1])


    return utterance_tensor, utterance_lengths, transcript_tensor, transcript_lengths

class SequencePooling(nn.Module):
    def __init__(self):
        super(SequencePooling, self).__init__()
        return
        
    def forward(self, x):
        x, lens = pad_packed_sequence(x)
        x = x.transpose(1,0)

        # Remove last dimension of utterance length for odd numbered lengths, 
        # maybe we pad when constructing tensor in collate instead?
        if (x.size(1) % 2):
            x = x[:,:-1,:]

        x = x.contiguous().view(x.size(0), int(x.size(1)/2), x.size(2)*2)
        x = x.transpose(1,0)
        return pack_padded_sequence(x, [int(i/2) for i in lens])

class Listener(nn.Module):
    '''
    Pyramidal Bidirectional LSTM which obtains a lower dimensional representation of input utterances
    '''
    def __init__(self, hidden_dim=256):
        super(Listener, self).__init__()
        self.SeqPool0 = SequencePooling()
        self.BLSTM0 = nn.LSTM(80, hidden_dim, bidirectional=True) 
        self.SeqPool1 = SequencePooling()
        self.BLSTM1 = nn.LSTM(4*hidden_dim, hidden_dim, bidirectional=True) 
        self.SeqPool2 = SequencePooling()
        self.BLSTM2 = nn.LSTM(4*hidden_dim, hidden_dim, bidirectional=True)

    def forward(self, x):
        x = self.SeqPool0(x)
        x, state = self.BLSTM0(x)
        x = self.SeqPool1(x)
        x, state = self.BLSTM1(x)
        x = self.SeqPool2(x)
        x, state = self.BLSTM2(x)
        
        # Unpack and return
        x, lens = pad_packed_sequence(x)

        return x, lens

class Attention(nn.Module):
    def __init__(self, listener_output_dim, decoder_state_dim, context_dim=128):
        super(Attention, self).__init__()
        self.phi = nn.Linear(decoder_state_dim, context_dim)
        self.key_mlp = nn.Linear(listener_output_dim, context_dim)
        self.value_mlp = nn.Linear(listener_output_dim, context_dim)
        self.softmax = nn.Softmax(dim=-1)
        return


    def forward(self, decoder_state, listener_features, utterance_mask):
        '''
        INPUTS
            decoder_state: concatenated context and embedded input character
            listener_features: high level features of utterance
            utterance_lengths: number of utterance relevant features in listener_features
        '''
        # Get queries, keys, and values
        #query_feat = self.phi(torch.cat(decoder_state, 1)) #NOTE: does decoder state need to include cell and hidden state
        query = self.phi(decoder_state)
        query = query.unsqueeze(dim=1)
        key = self.key_mlp(listener_features)
        value = self.value_mlp(listener_features)

        # Calculate the scalar energy for each time step in each sample
        energy = torch.bmm(query, key.transpose(2,1).transpose(0,2))

        # Convert to a probability distribution
        attention = self.softmax(energy)

        # Create mask corresponding to utterance relevant features to apply to softmax
        #attention = torch.zeros(attention.size()).masked_scatter_(utterance_mask.data, attention)
        #attention = F.normalize(attention, p=1, dim=-1)
        attention = F.normalize(attention*utterance_mask, p=1, dim=-1)

        # Get context vector
        context = torch.bmm(attention, value.transpose(1,0))

        return attention, context

class Speller(nn.Module):
    def __init__(self, decoder_input_dim, decoder_hidden_dim, attention_input_dim, context_dim, embedding_dim):
        super(Speller, self).__init__()

        #Learn the initial context state
        self.initial_hx0 = nn.Parameter(torch.zeros(1, embedding_dim+context_dim).float(), requires_grad=True)
        self.initial_cx0 = nn.Parameter(torch.zeros(1, embedding_dim+context_dim).float(), requires_grad=True)
        self.initial_hx1 = nn.Parameter(torch.zeros(1, embedding_dim+context_dim).float(), requires_grad=True)
        self.initial_cx1 = nn.Parameter(torch.zeros(1, embedding_dim+context_dim).float(), requires_grad=True)
        self.initial_hx2 = nn.Parameter(torch.zeros(1, embedding_dim+context_dim).float(), requires_grad=True)
        self.initial_cx2 = nn.Parameter(torch.zeros(1, embedding_dim+context_dim).float(), requires_grad=True)

        self.embedding = nn.Embedding(len(vocab), embedding_dim)
        self.lstmcell0 = nn.LSTMCell(embedding_dim+context_dim, decoder_hidden_dim)
        self.lstmcell1 = nn.LSTMCell(decoder_hidden_dim, decoder_hidden_dim)
        self.lstmcell2 = nn.LSTMCell(decoder_hidden_dim, decoder_hidden_dim)
        self.attention = Attention(decoder_input_dim, decoder_hidden_dim, context_dim)
        self.vocab_distribution = nn.Linear(decoder_hidden_dim+context_dim, len(vocab))
        #self.softmax = nn.LogSoftmax(dim=-1)
        return

    def forward(self, listener_features, utterance_lengths, transcript=None, teacher_force=1.0):
        #Set up Teacher Force
        if transcript is None:
            teacher_force = 0.0
        teacher_force = np.random.random_sample() < teacher_force
            
        # Initializations
        hidden_state = None #LSTM starts with no hidden state
        batch_size = listener_features.size(1)
        max_utterance_length = listener_features.size(0)
        hx0 = self.initial_hx0.expand(batch_size, -1)
        cx0 = self.initial_cx0.expand(batch_size, -1)
        hx1 = self.initial_hx1.expand(batch_size, -1)
        cx1 = self.initial_cx1.expand(batch_size, -1)
        hx2 = self.initial_hx2.expand(batch_size, -1)
        cx2 = self.initial_cx2.expand(batch_size, -1)

        # Create mask to extract only utterance relevant listener features
        utterance_mask = torch.stack([torch.cat([torch.ones(l), torch.zeros(max_utterance_length-l)]) for l in utterance_lengths])
        utterance_mask = Variable(utterance_mask.unsqueeze(1))
        if torch.cuda.is_available():
            utterance_mask = utterance_mask.cuda()
        #NOTE: Alternative Masking Method
        #utterance_mask = torch.LongTensor(range(max_utterance_length)).unsqueeze(1) < torch.LongTensor(utterance_lengths).unsqueeze(0)
        #utterance_mask = utterance_mask.transpose(1,0).unsqueeze(1)
        attention, context = self.attention(hx2, listener_features, utterance_mask) 

        input_char = Variable(torch.zeros(batch_size, 1).long()) # First element in vocab is <sos>
        if torch.cuda.is_available():
            input_char = input_char.cuda()
        embedded_char = self.embedding(input_char)
        rnn_input = torch.cat([embedded_char, context], dim=-1)

        # LSTM loop
        attention_list = []
        char_logits_list = []
        input_chars = []
        for char_idx in range(transcript.size(0)):

            hx0, cx0 = self.lstmcell0(rnn_input.squeeze(1), (hx0, cx0))
            hx1, cx1 = self.lstmcell1(hx0, (hx1, cx1))
            hx2, cx2 = self.lstmcell2(hx1, (hx2, cx2))
            attention, context = self.attention(hx2, listener_features, utterance_mask)
            output = torch.cat([hx2.unsqueeze(dim=1), context], dim=-1)
            #char_logits = self.softmax(self.vocab_distribution(output))
            char_logits = self.vocab_distribution(output)
 
            # Save attention and prediction
            attention_list.append(attention)
            char_logits_list.append(char_logits)
            input_chars.append(input_char)

            # If we are using teacher force then the next char is given from 
            # the transcript, otherwise use our prediction
            if teacher_force:
                input_char = transcript[char_idx,:].unsqueeze(dim=1) 
            else:
                char_logits_value, input_char = torch.max(char_logits, dim=-1)

            # Create input for next LSTMCell
            embedded_char = self.embedding(input_char)
            rnn_input = torch.cat([embedded_char, context], dim=-1)

            ### Inference
            
        return char_logits_list, attention_list, input_chars

    # Override train method to pass train call too attention?
    #def train():
    #   super method
    #   attention.train()

class LAS(nn.Module):
    def __init__(self, listener_hidden_dim, decoder_hidden_dim, attention_input_dim, context_dim, embedding_dim):
        super(LAS, self).__init__()
        self.listener = Listener(hidden_dim=listener_hidden_dim)
        self.speller = Speller(listener_hidden_dim*2, decoder_hidden_dim, attention_input_dim, context_dim, embedding_dim) #decoder_input_dim, decoder_hidden_dim, attention_input_dim, context_dim, embedding_dim

    def forward(self, packed_utterance, transcript=None, teacher_force=1.0):
        listener_features, utterance_lengths = self.listener(packed_utterance)
        char_logits, attentions, input_chars = self.speller(listener_features, utterance_lengths, transcript=transcript, teacher_force=teacher_force)
        return char_logits, attentions, input_chars


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
        return super(CrossEntropyLoss3D, self).forward(
            input.view(-1, input.size()[2]), target.view(-1))

def test_val(model, criterion, dataloader):
    model.eval() #only doing inference

    for idx, (utterance_tensor, utterance_lengths, transcript_tensor, transcript_lengths) in enumerate(dataloader):

        transcript_tensor = Variable(transcript_tensor)
        utterance_tensor = Variable(utterance_tensor)
        if torch.cuda.is_available():
            transcript_tensor = transcript_tensor.cuda()
            utterance_tensor = utterance_tensor.cuda()
        packed_utterance = pack_padded_sequence(utterance_tensor, utterance_lengths)

        char_logits, attentions, input_chars = las_model(packed_utterance)
        char_logits = torch.cat(char_logits, 1)
        input_chars = torch.cat(input_chars, 1)

        transcript_mask = torch.LongTensor(range(transcript_tensor.size(0))).unsqueeze(1) < torch.LongTensor(transcript_lengths).unsqueeze(0)
        transcript_mask = transcript_mask.transpose(1,0).contiguous()
        transcript_mask = transcript_mask.view(-1)
        if torch.cuda.is_available():
            transcript_mask = transcript_mask.cuda()
        loss = criterion(char_logits, transcript_tensor.transpose(1,0).contiguous())
        loss[~transcript_mask] = 0
        loss = torch.sum(loss) / transcript_mask.sum()
        epoch_loss += loss.data[0]
        
    print("Epoch: {}, average_loss: {}".format(epoch_loss/idx))
    return epoch_loss/idx

def train_las():
    ### LOAD DATA ###
    #root = "./"
    root = "/home/matt/.kaggle/competitions/11785-hw4/"
    train_set = WSJDataset(set_='train', root=root)
    val_set = WSJDataset(set_='val', root=root)
    #test_set = WSJDataset(set_='test', root=root)

    train_size = len(train_set)
    val_size = len(val_set)
    #test_size = len(test_set)


    batch_size = 32
    if torch.cuda.is_available():
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=wsj_collate, pin_memory=True, num_workers=8)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=wsj_collate, pin_memory=True, num_workers=8)
        #test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=wsj_collate)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, collate_fn=wsj_collate)
        #test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)


    listener_hidden_dim = 256
    decoder_hidden_dim = 256
    context_dim = 128
    las_model = LAS(listener_hidden_dim, decoder_hidden_dim, 256, 128, 128)
    #listener = Listener(hidden_dim=listener_hidden_dim)
    #speller = Speller(listener_hidden_dim*2, decoder_hidden_dim, 256, 128, 128) #decoder_input_dim, decoder_hidden_dim, attention_input_dim, context_dim, embedding_dim
    if torch.cuda.is_available():
        #listener = listener.cuda()
        #speller = speller.cuda()
        las_model.cuda()

    #optimizer = torch.optim.Adam([{'params':listener.parameters()}, {'params':speller.parameters()}], lr=1e-3, weight_decay=1e-5)
    optimizer = torch.optim.Adam(las_model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = CrossEntropyLoss3D(reduce=False)

    best_val_loss = float('inf')
    best_train_loss = float('inf')
    running_loss = 0.0
    max_epochs = 100
    for epoch in range(max_epochs):
        for idx, (utterance_tensor, utterance_lengths, transcript_tensor, transcript_lengths) in enumerate(train_loader):
            #listener.train()
            #speller.train()
            las_model.train()

            optimizer.zero_grad()

            transcript_tensor = Variable(transcript_tensor)
            utterance_tensor = Variable(utterance_tensor)
            if torch.cuda.is_available():
                transcript_tensor = transcript_tensor.cuda()
                utterance_tensor = utterance_tensor.cuda()
            packed_utterance = pack_padded_sequence(utterance_tensor, utterance_lengths)

            char_logits, attentions, input_chars = las_model(packed_utterance, transcript_tensor)
            #listener_features, utterance_lengths = listener(packed_utterance)
            #char_logits, attentions, input_chars= speller(listener_features, utterance_lengths, transcript_tensor)
            char_logits = torch.cat(char_logits, 1)
            input_chars = torch.cat(input_chars, 1)

            transcript_mask = torch.LongTensor(range(transcript_tensor.size(0))).unsqueeze(1) < torch.LongTensor(transcript_lengths).unsqueeze(0)
            transcript_mask = transcript_mask.transpose(1,0).contiguous()
            transcript_mask = transcript_mask.view(-1)
            if torch.cuda.is_available():
                transcript_mask = transcript_mask.cuda()
            #loss = criterion(char_logits[transcript_mask.unsqueeze(-1).expand_as(char_logits)], transcript_tensor.transpose(1,0).contiguous()[transcript_mask])
            loss = criterion(char_logits, transcript_tensor.transpose(1,0).contiguous())
            #loss = loss.masked_scatter_(Variable(~transcript_mask), Variable(torch.zeros(loss.size())))
            loss[~transcript_mask] = 0
            loss = torch.sum(loss) / transcript_mask.sum()
            running_loss += loss.data[0]
            
            if idx%200==0:
                print("Iteration: {}, loss: {}".format(idx, loss.data[0]))
                val, index = torch.max(char_logits[0,:,:], dim=-1)
                #print("".join(vocab[input_chars.data[0,:]]))
                print("".join(vocab[transcript_tensor.data[:,0]]))
                print("".join(vocab[index.data]))
            
            loss.backward()
            optimizer.step()

        avg_epoch_loss = running_loss/idx
        print("Epoch: {}, average_loss: {}".format(epoch, avg_epoch_loss))
        if avg_epoch_loss < best_train_false:
            best_train_false = avg_epoch_loss
            torch.save(las_model, "./las.pt")

        #reset running_loss
        running_loss = 0.0
            
    return

if __name__=="__main__":
    train_las()


