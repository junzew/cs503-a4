from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import maybe_cuda, setup_logger, unsort
import numpy as np
import math
from times_profiler import profiler

from transformer.Encoder import *
import transformer.Constants
from transformer.Layers import *


logger = setup_logger(__name__, 'train.log')
profilerLogger = setup_logger("profilerLogger", 'profiler.log', True)


def zero_state(module, batch_size):
    # * 2 is for the two directions
    return Variable(maybe_cuda(torch.zeros(module.num_layers * 2, batch_size, module.hidden))), \
           Variable(maybe_cuda(torch.zeros(module.num_layers * 2, batch_size, module.hidden)))

def generate_mask(ordered_doc_sizes, max_doc_size):
    pad_mask = maybe_cuda(torch.LongTensor([[1]*s + [0]*(max_doc_size-s) for s in ordered_doc_sizes]))
    return pad_mask

class PositionalEncoding(nn.Module):
    def __init__(self, dropout, dim, max_len=1000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]


class SentenceEncodingRNN(nn.Module):
    def __init__(self, input_size, hidden, num_layers):
        super(SentenceEncodingRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden = hidden
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden,
                            num_layers=self.num_layers,
                            dropout=0,
                            bidirectional=True)

    def forward(self, x):
        batch_size = x.batch_sizes[0]
        s = zero_state(self, batch_size)
        packed_output, _ = self.lstm(x, s)
        padded_output, lengths = pad_packed_sequence(packed_output) # (max sentence len, batch, 256) 

        maxes = Variable(maybe_cuda(torch.zeros(batch_size, padded_output.size(2))))
        for i in range(batch_size):
            maxes[i, :] = torch.max(padded_output[:lengths[i], i, :], 0)[0]

        return maxes

class Model(nn.Module):
    ''' A encoder model with self attention mechanism. '''
    def __init__(self, sentence_encoder, hidden, n_layers, n_head, d_k, d_v, d_model, d_inner, d_mlp, dropout=0.1):
        super(Model, self).__init__()

        self.PositionEncoder = PositionalEncoding(dropout, hidden*2)
        self.Transformer = Encoder(n_layers, n_head, d_k, d_v, d_model, d_inner)
        self.Dropoutlayer = nn.Dropout(p=dropout)
        self.Decoderlayer = self.build_decoder(hidden*2, d_mlp, dropout)
        self.sentence_encoder = sentence_encoder
        self.criterion = nn.CrossEntropyLoss()

    def pad(self, s, max_length):
        s_length = s.size()[0]
        v = Variable(maybe_cuda(s.unsqueeze(0).unsqueeze(0)))
        padded = F.pad(v, (0, 0, 0, max_length - s_length))  # (1, 1, max_length, 300)
        shape = padded.size()
        return padded.view(shape[2], 1, shape[3])  # (max_length, 1, 300)

    def pad_document(self, d, max_document_length):
        d_length = d.size()[0]
        v = d.unsqueeze(0).unsqueeze(0)
        padded = F.pad(v, (0, 0,0, max_document_length - d_length ))  # (1, 1, max_length, 300)
        shape = padded.size()
        return padded.view(shape[2], 1, shape[3])  # (max_length, 1, 300)

    def build_decoder(self,input_size,mlp_size,dropout,num_layers=1):
        decoder = []
        for i in range(num_layers):   
            decoder.append(nn.Linear(input_size, mlp_size))
            decoder.append(nn.ReLU())
            decoder.append(nn.Dropout(p=dropout))
        decoder.append(nn.Linear(mlp_size, 2))
        return nn.Sequential(*decoder)

    #def forward(self, enc_output, edu_mask, return_attns=False):
    def forward(self, batch, return_attns=False):
        batch_size = len(batch)

        sentences_per_doc = []
        all_batch_sentences = []
        for document in batch:
            all_batch_sentences.extend(document)
            sentences_per_doc.append(len(document))

        lengths = [s.size()[0] for s in all_batch_sentences]
        sort_order = np.argsort(lengths)[::-1]
        sorted_sentences = [all_batch_sentences[i] for i in sort_order]
        sorted_lengths = [s.size()[0] for s in sorted_sentences]

        max_length = max(lengths)
        logger.debug('Num sentences: %s, max sentence length: %s', 
                     sum(sentences_per_doc), max_length)

        padded_sentences = [self.pad(s, max_length) for s in sorted_sentences]
        big_tensor = torch.cat(padded_sentences, 1)  # (max_length, batch size, 300)
        processed_tensor = pack_padded_sequence(big_tensor, sorted_lengths)
        encoded_sentences = self.sentence_encoder(processed_tensor)
        unsort_order = Variable(maybe_cuda(torch.LongTensor(unsort(sort_order))))
        unsorted_encodings = encoded_sentences.index_select(0, unsort_order)

        index = 0
        encoded_documents = []
        for sentences_count in sentences_per_doc:
            end_index = index + sentences_count
            encoded_documents.append(unsorted_encodings[index : end_index, :])
            index = end_index

        doc_sizes = [doc.size()[0] for doc in encoded_documents]
        max_doc_size = np.max(doc_sizes)
        ordered_document_idx = np.argsort(doc_sizes)[::-1]
        ordered_doc_sizes = sorted(doc_sizes)[::-1]
        ordered_documents = [encoded_documents[idx] for idx in ordered_document_idx]
        padded_docs = [self.pad_document(d, max_doc_size) for d in ordered_documents]
        padded_docs = torch.stack(padded_docs).squeeze(2) # turn the tensor list into tensor

        ##############################################################
        pos_emb = self.PositionEncoder.pe[:, :padded_docs.size()[1]].expand(padded_docs.size())
        inputs = padded_docs+pos_emb

        sent_mask = generate_mask(ordered_doc_sizes, max_doc_size)
        non_pad_mask = sent_mask.unsqueeze(-1)
        slf_attn_mask = (1-sent_mask).unsqueeze(1).expand(-1,sent_mask.size()[1],-1).type(torch.bool)

        outputs = self.Transformer(inputs,non_pad_mask, slf_attn_mask)
        outputs = self.Dropoutlayer(outputs)
        outputs = self.Decoderlayer(outputs) # batch * length * 1

        doc_outputs = []
        for i, doc_len in enumerate(ordered_doc_sizes):
            doc_outputs.append(outputs[i, 0:doc_len - 1, :])  # -1 to remove last prediction

        unsorted_doc_outputs = [doc_outputs[i] for i in unsort(ordered_document_idx)]
        x = torch.cat(unsorted_doc_outputs, 0)
        
        return x

def create():
    sentence_encoder = SentenceEncodingRNN(input_size=300, hidden=256, num_layers=2)
    return Model(sentence_encoder, hidden=256, n_layers=2, n_head=4, d_k=64, d_v=64, d_model=512, d_inner=1024, d_mlp=100, dropout=0.1)
