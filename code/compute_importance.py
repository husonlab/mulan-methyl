'''
this program is for finding important features according to attention mechanism for user
'''
import pandas as pd
from transformers import BertModel, BertTokenizerFast, PreTrainedTokenizerFast, AutoTokenizer,BertTokenizer, DistilBertTokenizer, BertForMaskedLM, RobertaTokenizer, XLNetTokenizer, AlbertTokenizer, ElectraTokenizer
from transformers import ElectraForSequenceClassification, XLNetForSequenceClassification, DistilBertForSequenceClassification, BertForSequenceClassification, RobertaForSequenceClassification, AlbertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from torch.utils.data import DataLoader

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import numpy as np
from sklearn.preprocessing import Normalizer, MinMaxScaler, normalize


class ImportanceScore(object):

    def format_attention(self, attention):
        squeezed = []
        for layer_attention in attention:
            # batch_size x num_heads x seq_len x seq_len
            if len(layer_attention.shape) != 4:
                raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                                 "output_attentions=True when initializing your model.")
            squeezed.append(layer_attention.squeeze(0))
        # num_layers x num_heads x seq_len x seq_len
        return torch.stack(squeezed)

    def get_attention_dna(self, model, tokenizer, sentence_a, start, end, MAX_SEQ_LEN):
        '''
        :param model: finetuned language model
        :param tokenizer: corresponding tokenizer
        :param sentence_a: batch of sentences
        :param start: start pos of layer
        :param end: end pos of layer
        :param MAX_SEQ_LEN: length of tokenized sentences
        :return: attention score list for batach_size sentences.
        each record length aroud 85, the attention score each token give, when predict [CLS]
        and set of token list, each element should have same length with each element in attn_score.
        '''
        #inputs = tokenizer(sentence_a, return_tensors='pt')
        inputs = tokenizer(sentence_a, add_special_tokens = True, max_length=MAX_SEQ_LEN, truncation=True, padding='max_length', return_tensors='pt')
        input_ids = inputs['input_ids'] # (batch_size, tokenized_length)
        attention = model(input_ids)[-1] # length 12
        #input_id_list = input_ids[0].tolist() # Batch index 0
        input_id_list = input_ids.tolist()
        #tokens = tokenizer.convert_ids_to_tokens(input_id_list)
        batch_tokens = [tokenizer.convert_ids_to_tokens(i) for i in input_id_list]
        sep_pos_set = [tokens.index('[SEP]') for tokens in batch_tokens] # with batch size length
        batch_tokens_sub=[]
        for i in range(len(batch_tokens)):
            sep_idx = sep_pos_set[i]
            batch_tokens_sub.append(batch_tokens[i][1:sep_idx])
        attn = self.format_attention(attention) # layer, batch_size, attention_head_num, seq_length, seq_length
        attn_score = []
        for j in range(attn.shape[1]):
            tmp_attn_score = []
            sep_pos = sep_pos_set[j]
            for i in range(1, sep_pos): # exclude [CLS] [SEP]
                tmp_attn = attn[:,j,:,:,:]
                tmp_attn_score.append(float(tmp_attn[start:end+1,:,0,i].sum())) # a layer, all heads, [CLS], a token
            attn_score.append(tmp_attn_score)
        return attn_score, batch_tokens_sub



    def get_real_score(self, attention_scores):
        kmer = 6
        counts = np.zeros([len(attention_scores)+kmer-1])
        real_scores = np.zeros([len(attention_scores)+kmer-1])

        for i, score in enumerate(attention_scores):
            for j in range(kmer):
                counts[i+j] += 1.0
                real_scores[i+j] += score

        return real_scores

    def myfunc(self, data_sub, species, methy_type):
        print(f'start processing species {species} for 5hmC sites')
        # select 10% samples for each species and sites combination
        #data_sub = data_sub.sample(frac=0.1, replace=False, random_state=44)
        data_sub.reset_index(inplace=True, drop=True)
        # choice of pretrained language model
        pretrainedLM_list = ['bert', 'distilbert', 'electra']
        #pretrainedLM_list = ['distilbert']
        model_config = {
            'distilbert': (DistilBertTokenizer, DistilBertForSequenceClassification),
            'bert': (BertTokenizer, BertForSequenceClassification),
            'electra': (ElectraTokenizer, ElectraForSequenceClassification)
        }
        for pretrainedLM in pretrainedLM_list:
            print(f'current model is {pretrainedLM}')

            tokenPath = f'/home/ubuntu/project/dna_methy/pretrainedModel/{pretrainedLM}/tokenizer/{pretrainedLM}_seq_tax_trained'
            modelPath = f'/home/ubuntu/project/dna_methy/pretrainedModel/{pretrainedLM}/finetune/{methy_type}/model'

            MAX_SEQ_LEN = 100

            # load tokenizer and model
            tokenizer = model_config[pretrainedLM][0].from_pretrained(tokenPath)
            model = model_config[pretrainedLM][1].from_pretrained(modelPath, output_hidden_states=True, output_attentions=True, num_labels=2)

            # get attention value with batch size input
            if pretrainedLM == 'distilbert':
                start_layer = 5
                end_layer = 5
            else:
                start_layer = 11
                end_layer = 11

            my_ds = DataLoader(data_sub['text'], batch_size=128)
            for (idx, batch) in enumerate(my_ds):
                print(f'processing {idx} batch')
                my_b = batch
                attention_bs, tokens_bs = self.get_attention_dna(model, tokenizer, my_b, start=start_layer, end=end_layer, MAX_SEQ_LEN=MAX_SEQ_LEN)

                for i in range(len(my_b)):
                    tokens = tokens_bs[i]
                    attention_list = attention_bs[i]
                    dna_start = tokens.index('sequence')+2
                    if pretrainedLM == 'electra':
                        dna_end = tokens.index('for')-1
                    else:
                        dna_end = tokens.index('For')-1
                    dna_token = tokens[dna_start:dna_end]
                    dna_attention = attention_list[dna_start:dna_end]
                    dna_attention_scores = np.array(dna_attention).reshape(np.array(dna_attention).shape[0],1)
                    # attention_scores[0] = 0
                    dna_seq_scores = self.get_real_score(dna_attention_scores)
                    dna_seq_scores_1 = dna_seq_scores.reshape(1, dna_seq_scores.shape[0])
                    if i == 0:
                        bs_dna_scores = dna_seq_scores_1
                    else:
                        bs_dna_scores = np.vstack((bs_dna_scores,dna_seq_scores_1))
                print(f'shape of attention score for dna sequence in a batch {bs_dna_scores.shape}')
                #avg_bs_dna_score = np.mean(bs_dna_scores, axis=0)

            # merge batches
                if idx == 0:
                    all_bs_dna_scores = bs_dna_scores
                else:
                    all_bs_dna_scores = np.vstack((all_bs_dna_scores, bs_dna_scores))
                print(f'shape of all_bs_dna_score is {all_bs_dna_scores.shape}')

            if pretrainedLM == 'bert':
                all_pretrainedModel_dna_scores = all_bs_dna_scores
            else:
                all_pretrainedModel_dna_scores = all_pretrainedModel_dna_scores + all_bs_dna_scores

        avg_pretrainedModel_dna_scores = np.divide(all_pretrainedModel_dna_scores, len(pretrainedLM_list))
        avg_pretrainedModel_dna_scores = avg_pretrainedModel_dna_scores.tolist()

        return avg_pretrainedModel_dna_scores

    def methylated_positoin_inference(self, row, methy_type):
        if row['pred_label'] == 1:
            pos_max = row['importance_score'].index(max(row['importance_score']))
            if row['seq'][pos_max] == methy_type[-1]:
                methylated_pos = pos_max
            else:
                sorted_index_interval = sorted(range(len(row['importance_score'])), key=lambda k: row['importance_score'][k], reverse=True)[0:5]
                for i in sorted_index_interval:
                    if row['seq'][i] == methy_type[-1]:
                        methylated_pos = i
                        break
                methylated_pos = None
        else:
            methylated_pos = None
        return methylated_pos

    def center_methylation_judgement(self, row):
        pos_max = row['importance_score'].index(max(row['importance_score']))
        print(pos_max)
        if pos_max in range(16, 25):
            center_methylated = 1
        else:
            center_methylated = 0
        return center_methylated






