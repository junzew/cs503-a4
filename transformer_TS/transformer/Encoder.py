''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Layers import *


# Most of the code was borrowed from "Yu-Hsiang Huang"
def get_non_pad_mask(seq):
	assert seq.dim() == 2
	return seq.ne(0).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
	''' Sinusoid position encoding table '''

	def cal_angle(position, hid_idx):
		return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

	def get_posi_angle_vec(position):
		return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

	sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

	sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
	sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

	if padding_idx is not None:
		# zero vector for padding dimension
		sinusoid_table[padding_idx] = 0.

	return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
	''' For masking out the padding part of key sequence. '''

	# Expand to fit the shape of key query attention matrix.
	len_q = seq_q.size(1)
	padding_mask = seq_k.eq(0)
	padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

	return padding_mask

def get_subsequent_mask(seq):
	''' For masking out the subsequent info. '''

	sz_b, len_s = seq.size()
	subsequent_mask = torch.triu(
		torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
	subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

	return subsequent_mask



class Encoder(nn.Module):
	''' A encoder model with self attention mechanism. '''

	def __init__(
			self,n_layers, n_head,d_k, d_v,
			d_model, d_inner, dropout=0.1):

		super().__init__()

		self.layer_stack = nn.ModuleList([
			EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
			for _ in range(n_layers)])

	def forward(self, enc_output, non_pad_mask, slf_attn_mask, return_attns=False):

		enc_slf_attn_list = []
		# -- Prepare masks
		# src_pos = src_pos.permute(1,0) #b * max_length
		# slf_attn_mask = get_attn_key_pad_mask(seq_k=src_pos, seq_q=src_pos)
		# non_pad_mask = get_non_pad_mask(src_pos)
		# -- Forward
		# print(src_seq.size())
		# print(self.position_enc(src_pos).size())
		# enc_output = src_seq + position_emb
		if (enc_output != enc_output).any():
			print('nan at line 91 in EncoderForSumm.py')

		for enc_layer in self.layer_stack:
			enc_output, enc_slf_attn = enc_layer(
				enc_output,
				non_pad_mask=non_pad_mask,
				slf_attn_mask=slf_attn_mask)

			if (enc_output != enc_output).any():
				print('nan at line 101 in EncoderForSumm.py')
			if return_attns:
				enc_slf_attn_list += [enc_slf_attn]

		if return_attns:
			return enc_output, enc_slf_attn_list
		return enc_output
