# -*- coding: utf-8 -*-
'''
load user data and process it to a sentence
'''

import pandas as pd
import os
from Bio import SeqIO

class DataProcesser(object):

    # seq2kmer
    def seq2kmer(self, seq, k):
        """
        Convert original sequence to kmers

        Arguments:
        seq -- str, original sequence.
        k -- int, kmer of length k specified.

        Returns:
        kmers -- str, kmers separated by space
        """
        kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
        kmers = " ".join(kmer)
        return kmers

    # cut sequence if each sample's length is not equal 41
    def cut_sample(self, obj, sec):
        return [obj[i:i+sec] for i in range(0, len(obj), sec)]

    # reshape dataframe if sample's length is not equal 41（here list　approaches）
    def expand_df(self, ini_df):
        ini_df['seq'] = list(map(lambda x: self.cut_sample(x, 41),ini_df['seq']))
        expanded_df = ini_df.explode('seq')
        return expanded_df

    def seed_and_extend(self, obj, sec, methyType):
        seed_dict={}
        if len(obj) >= sec:
            seed_pos = sorted([i for i,x in enumerate(obj) if x == methyType[-1]])
            if len(seed_pos) != 0:
                for ele in seed_pos:
                    if ele in range(int((sec-1)/2), int((len(obj)-(sec-1)/2))):
                        seed_dict[ele] = (int(ele-(sec-1)/2), ele, int(ele+(sec-1)/2))
        else:
            seed_dict=None
        return seed_dict


    def expand_df_seed(self, ini_df, methyType, developer):
        ini_df = ini_df.rename(columns={'seq':'long_seq'})
        ini_df['seed_and_extend'] = list(map(lambda x: self.seed_and_extend(x, 41, methyType), ini_df['long_seq']))
        ini_df = ini_df.dropna()
        ini_df['extend_range'] = list(map(lambda x: list(x.values()), ini_df['seed_and_extend']))
        extended_df = ini_df.explode('extend_range')
        extended_df['seed'] = list(map(lambda x: x[1], extended_df['extend_range']))
        extended_df['seq'] = list(map(lambda x, y: x[y[0]:y[2]+1], extended_df['long_seq'], extended_df['extend_range']))
        extended_df['id'] = list(map(lambda x, y: f'{x}_{y}', extended_df['id'], extended_df['seed']))
        if developer:
            extended_df = extended_df[['id', 'seed', 'seq', 'label']]
        else:
            extended_df = extended_df[['id', 'seed', 'seq']]
        extended_df.reset_index(drop=True, inplace=True)
        return extended_df


    # get sample's lineage
    def get_lineage(self, x):
        # load taxonomy file
        df_taxonomy = pd.read_csv('./data/taxonomy/ncbi_gtdb_processed.csv', low_memory=False) #2190288
        tmp_df = df_taxonomy[df_taxonomy['species']==x][['species', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom', 'domain']]
        if len(tmp_df) >= 1:
            tmp_df.reset_index(drop=True, inplace=True)
            list_ = list(tmp_df.iloc[0,:])
            list_ = list(map(lambda x: x[3:], list_))
        else:
            print(f'failed to track the taxonomy lineage of provided species')
            list_=[]
        return list_

    def description_Creator(self, row, taxonomy_lineage):
        sentence4tax = f'For this organism, its species is {taxonomy_lineage[0]}, its genus is {taxonomy_lineage[1]}, its family is {taxonomy_lineage[2]}, its order is {taxonomy_lineage[3]}, its class is {taxonomy_lineage[4]}, its phylum is {taxonomy_lineage[5]}, its kingdom is {taxonomy_lineage[6]}, its domain is {taxonomy_lineage[7]}'
        sequence = row['sequence_6mer']
        sentence4seq = f'The DNA sequence is {sequence}'
        description = f'{sentence4seq}. {sentence4tax}.'
        return description

    def data_loader(self, dataPath, dataType, customSpecies=False, labelled=True):
        if dataType != 'fasta':
            df = pd.read_csv(dataPath, sep='\t')
            if labelled == False:
                df.columns = ['id', 'species', 'methyl_type', 'seq']
            else:
                df.columns = ['id', 'seq', 'species', 'methyl_type', 'label']
        # generate a sequence of 6mer
        df['sequence_6mer'] = list(map(lambda x: self.seq2kmer(x, 6), df['seq']))
        # mapping species name
        if customSpecies == False:
            species_mapped = pd.read_csv('./data/taxonomy/species_name_mapped.csv')
            species_mapped_dict = dict(zip(species_mapped['abbre_species'], species_mapped['full_species']))
            df = df.rename(columns={'species': 'short_species'})
            df['species'] = list(map(lambda x: species_mapped_dict[x], df['short_species']))
        description4sample = []
        for index, row in df.iterrows():
            # generate taxonomy lineage according to species
            species_ = row['species']
            taxonomy_lineage = self.get_lineage(species_)
            desc = self.description_Creator(row, taxonomy_lineage)
            description4sample.append(desc)
        df['text'] = description4sample
        if labelled:
            processed_df = df[['id', 'text', 'methyl_type', 'species', 'label']]
        else:
            processed_df = df[['id', 'text', 'methyl_type', 'species']]
        return processed_df

    def data_loader_predict(self, dataPath, dataType, varSpecies, customSpecies=False, labelled=True):
        '''
        curPath: current path(project path)
        dataPath: relative path of data
        dataType: data type, fasta or txt
        varSpecies: species that dataset belongs to
        methyType: methylation site type, 6mA, 4mC or 5hmC
        customSpecies: species is not belongs to iDNA-MS, H.sapiens for False, s__Homo sapiens for True
        developer: the input data has label, for developer testing code
        customLength: if the input length is longer than 41
        return: dataset contains sentence
        '''
        if customSpecies == False:
            species_mapped = pd.read_csv('./data/taxonomy/species_name_mapped.csv')
            species_mapped_dict = dict(zip(species_mapped['abbre_species'], species_mapped['full_species']))
            # apply long species name to initial dataset
            varSpecies = species_mapped_dict[varSpecies]
        taxonomy_lineage = self.get_lineage(varSpecies)
        if dataType == 'fasta':
            # read fasta
            index_ = []
            seq = []
            with open(os.path.join(curPath, dataPath)) as f:
                for line in f:
                    if line.startswith('>'):
                        tmp = line.replace('>', '')
                        idx = tmp.replace('\n', '')
                        index_.append(idx)
                    else:
                        seq.append(line.replace('\n', ''))
            df = pd.DataFrame({'id':index_, 'seq': seq})
        else:
            df = pd.read_csv(dataPath, sep='\t')
            if labelled == False:
                df.columns = ['id', 'seq']
            else:
                df.columns = ['id', 'seq', 'label']
        df['sequence_6mer'] = list(map(lambda x: self.seq2kmer(x, 6), df['seq']))
        description4sample = []
        for index, row in df.iterrows():
            desc = self.description_Creator(row, taxonomy_lineage)
            description4sample.append(desc)
        df['text'] = description4sample
        if developer:
            processed_df = df[['id', 'seq', 'text', 'label']]
        else:
            processed_df = df[['id', 'seq', 'text']]
        return processed_df










