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
cpu = False
gpu_dev =0

class WSJDataLoader(DataLoader):
    def __init__(self, dataset, phonemes, batch_size, shuffle=True):
        self.dataset = dataset
        self.phonemes = phonemes
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self):
        # Randomize dataset
        num_utterances = self.dataset.shape[0]
        if self.shuffle:
            rand_perm = np.random.permutation(num_utterances)
        else:
            rand_perm = np.asarray(range(num_utterances))
        data = self.dataset[rand_perm]
        if self.phonemes:
            phonemes = self.phonemes[rand_perm]

        pos = 0
        done = False
        while not done:

            #Get batch, this is the last iteration if position is larger than number of utterances
            seq_batch = data[pos:pos+self.batch_size]
            if self.phonemes:
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
            if self.phonemes:
                sorted_phoneme_batch = phoneme_batch[sort_perm]
            max_seq_len = int(sorted_seq_lengths[0])

            #Put sorted sequences into padded Tensor
            seq_tensor = torch.zeros(max_seq_len, sorted_seq_batch.shape[0], 40).float()
            for idx, (seq, seq_len) in enumerate(zip(sorted_seq_batch, sorted_seq_lengths)):
                seq_tensor[:seq_len, idx] = torch.FloatTensor(seq)

            #Get num labels/phonemes in each sequence
            if self.phonemes:
                num_phonemes = np.asarray([labels.shape[0] for labels in sorted_phoneme_batch])

            seq_tensor = Variable(seq_tensor)
            if torch.cuda.is_available() and not cpu:
                seq_tensor = seq_tensor.cuda()
            packed_data = pack_padded_sequence(seq_tensor, sorted_seq_lengths)

            # do the phonemes need to be a Tensor or Variable?
            if self.phonemes:
                yield packed_data, torch.IntTensor(np.concatenate(sorted_phoneme_batch)), num_phonemes
            else:
                yield packed_data, None, None



if __name__=="__main__":
    #Parse input args
    parser = argparse.ArgumentParser(description='Get Network Hyperparameters.')
    parser.add_argument('--hidden_dim', dest='hidden_dim', type=int, default=350)
    #parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=30.0)
    #parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=1e-6)
    args = parser.parse_args()
    print(args, flush=True)

    ## Load data
    #root = '/home/matt/.kaggle/competitions/11785-hw3p2/'
    root = ''
    test_data = np.load(root+"test.npy")

    ## Initialize DataLoaders
    batch_size = 1
    test_loader = WSJDataLoader(test_data, None, batch_size=batch_size, shuffle=False)


    ## Initialize network and load pretrained model
    from train_hw3p3 import WSJNet
    model_file = "part3_bidirectional.pt"
    model = WSJNet(args.hidden_dim)
    if torch.cuda.is_available() and not cpu:
        model.load_state_dict(torch.load(model_file))
        model.cuda(gpu_dev)
    else:
        model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
    model.eval()

    test_logits = np.empty(test_data.shape[0], dtype=object)
    for idx, (data, phonemes, num_phonemes) in enumerate(test_loader):
       
        logits, seq_lengths = model(data)
        test_logits[idx] = logits[:,0,1:].data.cpu().numpy()

    np.save("my_logits.npy", test_logits) 
