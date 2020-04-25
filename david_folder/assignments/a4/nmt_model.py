#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 4
nmt_model.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""
from collections import namedtuple
import sys
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model_embeddings import ModelEmbeddings
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        """ Init NMT Model.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(NMT, self).__init__()
        self.model_embeddings = ModelEmbeddings(embed_size, vocab)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        # default values
        self.encoder = None 
        self.decoder = None
        self.h_projection = None
        self.c_projection = None
        self.att_projection = None
        self.combined_output_projection = None
        self.target_vocab_projection = None
        self.dropout = None


        ### YOUR CODE HERE (~8 Lines)
        ### TODO - Initialize the following variables:
        
        ###     self.encoder (Bidirectional LSTM with bias); 
        # our input size is based around the number of features * embeddings usually, but slightly different here
        # for LSTM we are inputting a sentence of size m
        # each word in the sentence is converted to an embedding of size e
        # this embedding of size e is then fed in sequentially, so input size is just size of embedding. 
        
        
        # I think we just want one layer for the encoder, which is default
        # also need to add dropout 
        self.encoder = nn.LSTM(input_size = embed_size, 
                               hidden_size = self.hidden_size, 
                               num_layers = 1, 
                               bias = True, 
                               bidirectional = True)
        
        ###     self.decoder (LSTM Cell with bias)
        
        # decoder is initialized with a linear projection of hidden state & cell state from encoder...
        # from what I am reading the output of encoder should be a single vector rather than a series out outputs
        # specifically, our input to decoder is just the final hidden state of encoder
        # we are also going to be feeding in the associated word embedding....which is of size e. 
        # so we concatenate the final hidden state of encoder + word embedding e....
        
        # I am making a guess dimensions of hidden layer size for decoder is still hidden size input to class....
        # helpful reading: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
        self.decoder = nn.LSTMCell(input_size = self.hidden_size + embed_size, 
                                   hidden_size = self.hidden_size, 
                                   bias = True)
        
        
        ### What are these linear layers actually doing?
        # These are building a fully connected layer
        # finding a way to convert encoder's final hidden state to proper dimensions for decoder hidden state?
        # Same with the cell states.
        
        ###     self.h_projection (Linear Layer with no bias), called W_{h} in the PDF.
        # Applies a linear transformation to the incoming data: y = xA^T + b
        # Based on equation 3 in assignment I think input is 2 * hidden_size, output is hidden_size
        self.h_projection = nn.Linear(in_features = 2 * self.hidden_size, out_features = self.hidden_size, bias=False)
        
        
        ###     self.c_projection (Linear Layer with no bias), called W_{c} in the PDF.
        ## same as above: equation 4 shows that this reducing size 
        self.c_projection = nn.Linear(in_features = 2 * self.hidden_size, out_features = self.hidden_size, bias=False)
        
        #### Attention is happening after decoder input
        # input to attention is hidden_encoding, which is 2 * hidden size. output is m * 1....
        
        ###     self.att_projection (Linear Layer with no bias), called W_{attProj} in the PDF.
        # equations 7 - 9 show whats happening here......
        # equation 7 shows theinput to attention projection is h_enc which is of size 2 * hidden_size 
        # equation 8 shows the output of vector h x 2h for W_attproj
        self.att_projection = nn.Linear(in_features = 2 * self.hidden_size, out_features = self.hidden_size, bias=False)
        
        
        ###     self.combined_output_projection (Linear Layer with no bias), called W_{u} in the PDF.
        # equation 10 - 11 show us the size here: 
        # we concatenate attention output + decoder hidden state; output is size hidden state
        self.combined_output_projection = nn.Linear(in_features = 3 * self.hidden_size, out_features = self.hidden_size, bias=False)
        
        ###     self.target_vocab_projection (Linear Layer with no bias), called W_{vocab} in the PDF.
        # W_vocab is Vt x h, so hidden input, Vt length utput
        # the vocab.tgt is going to spit out entry for target language..source could be used also\
        # should be equivalent to model_embeddings weight shape....one of the aspects of that matrix.
        self.target_vocab_projection = nn.Linear(in_features =  self.hidden_size, out_features = len(self.vocab.tgt), bias=False)
        
        ###     self.dropout (Dropout Layer)
        # equation 12: dropout weight vector is h *1, so input is 1, and output is hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        
        ###
        ### Use the following docs to properly initialize these variables:
        ###     LSTM:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
        ###     LSTM Cell:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell
        ###     Linear Layer:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Linear
        ###     Dropout Layer:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.Dropout
        self.dropout = nn.Dropout(dropout_rate)

        ### END YOUR CODE


    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param source (List[List[str]]): list of source sentence tokens
        @param target (List[List[str]]): list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)   # Tensor: (src_len, b)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)   # Tensor: (tgt_len, b)

        ###     Run the network forward:
        ###     1. Apply the encoder to `source_padded` by calling `self.encode()`
        ###     2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
        ###     3. Apply the decoder to compute combined-output by calling `self.decode()`
        ###     4. Compute log probability distribution over the target vocabulary using the
        ###        combined_outputs returned by the `self.decode()` function.

        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()
        
        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores


    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ Apply the encoder to source sentences to obtain encoder hidden states.
            Additionally, take the final states of the encoder and project them to obtain initial states for decoder.

        @param source_padded (Tensor): Tensor of padded source sentences with shape (src_len, b), where
                                        b = batch_size, src_len = maximum source sentence length. Note that 
                                       these have already been sorted in order of longest to shortest sentence.
        @param source_lengths (List[int]): List of actual lengths for each of the source sentences in the batch
        @returns enc_hiddens (Tensor): Tensor of hidden units with shape (b, src_len, h*2), where
                                        b = batch size, src_len = maximum source sentence length, h = hidden size.
        @returns dec_init_state (tuple(Tensor, Tensor)): Tuple of tensors representing the decoder's initial
                                                hidden state and cell.
        """
        enc_hiddens, dec_init_state = None, None

        ### YOUR CODE HERE (~ 8 Lines)
        ### TODO:
        ###     1. Construct Tensor `X` of source sentences with shape (src_len, b, e) using the source model embeddings.
        ###         src_len = maximum source sentence length, b = batch size, e = embedding size. Note
        ###         that there is no initial hidden state or cell for the decoder.
        
        # model embeddings were initialized as 'self.model_embeddings', just need to read in & specify source
        # looking at other solution it looks like I need to pass in the source_padded....I am not super clear on this
        # Here is my interpretation: 
        #     - We create an instance of ModelEmbeddings during __init__ based on our vocab & expected embedding size
        #     - However, we have not yet passed any of the info in yet
        #     - We limit to our instance.source, which is an instance of nn.embedding: https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding
        #     - We then need to pass in our tensor of padded source sentences, which are (src_len, b)
        #     - The embedding size (e) is already built into our nn.embedding instance, so output will account for this
        #     - Final shape will be (src_len, b, e)
        X = self.model_embeddings.source(source_padded)
        
        
        ###     2. Compute `enc_hiddens`, `last_hidden`, `last_cell` by applying the encoder to `X`.
        ###         - Before you can apply the encoder, you need to apply the `pack_padded_sequence` function to X.
        ###         - After you apply the encoder, you need to apply the `pad_packed_sequence` function to enc_hiddens.
        ###         - Note that the shape of the tensor returned by the encoder is (src_len b, h*2) and we want to
        ###           return a tensor of shape (b, src_len, h*2) as `enc_hiddens`.
        
        # start by padding X: https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1utils_1_1rnn_1a6c14a90e57eb631f51f06e52a600f7f7.html
        # expects an input matrix along with lengths of each sentence
        X = pack_padded_sequence(X, source_lengths)
        
        # now we can apply the encoder: self.encoder
        # this is our instance of the LSTM for encoder, defined during __init__
        # returns: output, (hidden, cell) 
        #  (last_hidden, last_cell); unpacking for now since step 3 seems to need this
        enc_hiddens, (last_hidden, last_cell) = self.encoder(X)
        
        
        # finally, we need to apply to pad_packed_sequence function to enc_hideens
        # https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1utils_1_1rnn_1aea13e54d273ec16fa8288e2262c28f30.html
        # https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        # great overview above - in short its going to help us make more computationally efficient moves
        # The returned Tensor’s data will be of size T x B x *, where T is the length of the longest sequence and B is the batchsize. 
        # we don't need to keep the tensor of lengths, only need to actual tensor data
        enc_hiddens, _ = pad_packed_sequence(enc_hiddens)
        
        ###         - Note that the shape of the tensor returned by the encoder is (src_len b, h*2) and we want to
        ###           return a tensor of shape (b, src_len, h*2) as `enc_hiddens`.
        ### permute makes it easy to just rearrange the order of data:
        ## we have: (src_len, b, h*2) and we want: (b, src_len, h*2)....
        ## x.permute(1, 0, 2)
        enc_hiddens = enc_hiddens.permute(1,0,2)
        
        
        ###     3. Compute `dec_init_state` = (init_decoder_hidden, init_decoder_cell):
        ###         - `init_decoder_hidden`:
        ###             `last_hidden` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
        ###             Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
        ###             Apply the h_projection layer to this in order to compute init_decoder_hidden.
        ###             This is h_0^{dec} in the PDF. Here b = batch size, h = hidden size
        ###         - `init_decoder_cell`:
        ###             `last_cell` is a tensor shape (2, b, h). The first dimension corresponds to forwards and backwards.
        ###             Concatenate the forwards and backwards tensors to obtain a tensor shape (b, 2*h).
        ###             Apply the c_projection layer to this in order to compute init_decoder_cell.
        ###             This is c_0^{dec} in the PDF. Here b = batch size, h = hidden size
        ###
        
        # we want to go from (2,b,h) -> (b, 2*h); can do this with concatenation
        # I believe we want dim = 1 to stack cols 
        cat_hidden = torch.cat((last_hidden[0], last_hidden[1]), 1)
        
        # now we pass this value into our h_projection layer
        init_decoder_hidden = self.h_projection(cat_hidden)
        
        # compute last cell, again: we want to go from (2,b,h) -> (b, 2*h)
        cat_cell = torch.cat((last_cell[0], last_cell[1]), 1)
        
        # Apply the c_projection layer to this in order to compute init_decoder_cell.
        init_decoder_cell = self.c_projection(cat_cell)
        
        # make a tuple
        dec_init_state = (init_decoder_hidden, init_decoder_cell)
        
        ### See the following docs, as you may need to use some of the following functions in your implementation:
        ###     Pack the padded sequence X before passing to the encoder:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_padded_sequence
        ###     Pad the packed sequence, enc_hiddens, returned by the encoder:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pad_packed_sequence
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     Tensor Permute:
        ###         https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute


        ### END YOUR CODE

        return enc_hiddens, dec_init_state


    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
                dec_init_state: Tuple[torch.Tensor, torch.Tensor], target_padded: torch.Tensor) -> torch.Tensor:
        """Compute combined output vectors for a batch.

        @param enc_hiddens (Tensor): Hidden states (b, src_len, h*2), where
                                     b = batch size, src_len = maximum source sentence length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks (b, src_len), where
                                     b = batch size, src_len = maximum source sentence length.
        @param dec_init_state (tuple(Tensor, Tensor)): Initial state and cell for decoder
        @param target_padded (Tensor): Gold-standard padded target sentences (tgt_len, b), where
                                       tgt_len= maximum target sentence length, b = batch size. 

        @returns combined_outputs (Tensor): combined output tensor  (tgt_len, b,  h), where
                                        tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
        """
        
        #### David's interpretation of this step: 
        # We first remove the end token (for training? 
        # initialize the decoder state as the init state, which is the final hidden / cell state of the Encoder
        # the o_prev is just going to represent combined-output vector, which is the concatenation of the attention + decoder hidden state
        # Note: in the next step we will use decoder hidden state to calculate the attention -- happens in `.step()`
        # for a target sentence in our batch we will iterate through each target word, going through and: 
            # - passing in the word embeddings for target
            # - getting updated decoder hidden state & attention vectors 
            # - outputting the o_prev, which will eventually be passed through to build out the probability distribution over target words
        
        
        # Chop of the <END> token for max length sentences.
        target_padded = target_padded[:-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero --> we start with 0s
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        ### YOUR CODE HERE (~9 Lines)
        ### TODO:
        ###     1. Apply the attention projection layer to `enc_hiddens` to obtain `enc_hiddens_proj`,
        ###         which should be shape (b, src_len, h),
        ###         where b = batch size, src_len = maximum source length, h = hidden size.
        ###         This is applying W_{attProj} to h^enc, as described in the PDF.
        
        enc_hiddens_proj = self.att_projection(enc_hiddens)
        
        
        ###     2. Construct tensor `Y` of target sentences with shape (tgt_len, b, e) using the target model embeddings.
        ###         where tgt_len = maximum target sentence length, b = batch size, e = embedding size.
        
        # we need to convert the target language sentencesinto embeddings. 
        Y = self.model_embeddings.target(target_padded)
        
        
        ###     3. Use the torch.split function to iterate over the time dimension of Y.
        
        ###         Within the loop, this will give you Y_t of shape (1, b, e) where b = batch size, e = embedding size.
        ###             - Squeeze Y_t into a tensor of dimension (b, e). 
        ###             - Construct Ybar_t by concatenating Y_t with o_prev.
        ###             - Use the step function to compute the the Decoder's next (cell, state) values
        ###               as well as the new combined output o_t.
        ###             - Append o_t to combined_outputs
        ###             - Update o_prev to the new o_t.
        
        # going to separate ot tgt_len, b, e -> size of (1,b,e) for each word in the sentence (example in my testing notebook)
        for Y_t in torch.split(Y, 1):
            
            # squeeze Y_t into a tensor of dimension (b,e) --> this just removes the 1 in shape
            # For example, if input is of shape: (A \times 1 \times B \times C \times 1 \times D)(A×1×B×C×1×D) then the out tensor will be of shape: (A \times B \times C \times D)(A×B×C×D) .
            Y_t = torch.squeeze(Y_t)
       
            # Construct Ybar_t by concatenating Y_t with o_prev.
            # assuming we just dimension 1 again - will need to confirm 
            Ybar_t = torch.cat((Y_t, o_prev), 1)
            
            # Use the step function to compute the the Decoder's next (cell, state) values
            # as well as the new combined output o_t.
            # e_t looks like it won't actually be used? just added for debugging 
            # we wil get out updated dec_state (used in next iteration) as well as updated o_t, which is stored
            dec_state, o_t, e_t = self.step(Ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_masks)
            
            # Append o_t to combined_outputs
            combined_outputs.append(o_t)
            
            # Update o_prev to the new o_t.
            o_prev = o_t
        
        ###     4. Use torch.stack to convert combined_outputs from a list length tgt_len of
        ###         tensors shape (b, h), to a single tensor shape (tgt_len, b, h)
        ###         where tgt_len = maximum target sentence length, b = batch size, h = hidden size.
        
        # onto the torch.stack process 
        # this can take a list as input and output a stacked tensor...unclear on how to handle dimensions here so using default.
        combined_outputs = torch.stack(combined_outputs)
        
        ### Note:
        ###    - When using the squeeze() function make sure to specify the dimension you want to squeeze
        ###      over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###   
        ### Use the following docs to implement this functionality:
        ###     Zeros Tensor:
        ###         https://pytorch.org/docs/stable/torch.html#torch.zeros
        ###     Tensor Splitting (iteration):
        ###         https://pytorch.org/docs/stable/torch.html#torch.split
        ###     Tensor Dimension Squeezing:
        ###         https://pytorch.org/docs/stable/torch.html#torch.squeeze
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     Tensor Stacking:
        ###         https://pytorch.org/docs/stable/torch.html#torch.stack


        ### END YOUR CODE

        return combined_outputs


    def step(self, Ybar_t: torch.Tensor,
            dec_state: Tuple[torch.Tensor, torch.Tensor],
            enc_hiddens: torch.Tensor,
            enc_hiddens_proj: torch.Tensor,
            enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.

        @param Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        @param dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks shape (b, src_len),
                                    where b = batch size, src_len is maximum source length. 

        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                      We are simply returning this value so that we can sanity check
                                      your implementation.
        """

        combined_output = None

        ### YOUR CODE HERE (~3 Lines)
        ### TODO:
        ###     1. Apply the decoder to `Ybar_t` and `dec_state`to obtain the new dec_state.
        ###     2. Split dec_state into its two parts (dec_hidden, dec_cell)
        ###     3. Compute the attention scores e_t, a Tensor shape (b, src_len). 
        ###        Note: b = batch_size, src_len = maximum source length, h = hidden size.
        ###
        ###       Hints:
        ###         - dec_hidden is shape (b, h) and corresponds to h^dec_t in the PDF (batched)
        ###         - enc_hiddens_proj is shape (b, src_len, h) and corresponds to W_{attProj} h^enc (batched).
        ###         - Use batched matrix multiplication (torch.bmm) to compute e_t.
        ###         - To get the tensors into the right shapes for bmm, you will need to do some squeezing and unsqueezing.
        ###         - When using the squeeze() function make sure to specify the dimension you want to squeeze
        ###             over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        ###
        ### Use the following docs to implement this functionality:
        ###     Batch Multiplication:
        ###        https://pytorch.org/docs/stable/torch.html#torch.bmm
        ###     Tensor Unsqueeze:
        ###         https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
        ###     Tensor Squeeze:
        ###         https://pytorch.org/docs/stable/torch.html#torch.squeeze


        ### END YOUR CODE

        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.byte(), -float('inf'))

        ### YOUR CODE HERE (~6 Lines)
        ### TODO:
        ###     1. Apply softmax to e_t to yield alpha_t
        ###     2. Use batched matrix multiplication between alpha_t and enc_hiddens to obtain the
        ###         attention output vector, a_t.
        #$$     Hints:
        ###           - alpha_t is shape (b, src_len)
        ###           - enc_hiddens is shape (b, src_len, 2h)
        ###           - a_t should be shape (b, 2h)
        ###           - You will need to do some squeezing and unsqueezing.
        ###     Note: b = batch size, src_len = maximum source length, h = hidden size.
        ###
        ###     3. Concatenate dec_hidden with a_t to compute tensor U_t
        ###     4. Apply the combined output projection layer to U_t to compute tensor V_t
        ###     5. Compute tensor O_t by first applying the Tanh function and then the dropout layer.
        ###
        ### Use the following docs to implement this functionality:
        ###     Softmax:
        ###         https://pytorch.org/docs/stable/nn.html#torch.nn.functional.softmax
        ###     Batch Multiplication:
        ###        https://pytorch.org/docs/stable/torch.html#torch.bmm
        ###     Tensor View:
        ###         https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
        ###     Tensor Concatenation:
        ###         https://pytorch.org/docs/stable/torch.html#torch.cat
        ###     Tanh:
        ###         https://pytorch.org/docs/stable/torch.html#torch.tanh


        ### END YOUR CODE

        combined_output = O_t
        return dec_state, combined_output, e_t

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size. 
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.
        
        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)


    def beam_search(self, src_sent: List[str], beam_size: int=5, max_decoding_time_step: int=70) -> List[Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(1),
                                                                           src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.model_embeddings.target(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _  = self.step(x, h_tm1,
                                                      exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.model_embeddings.source.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.model_embeddings.embed_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)