# -*- coding: utf-8 -*-
'''
load user data and process it to a sentence
'''

import pandas as pd
import os

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

    # get sample's lineage
    def get_lineage(self, x, curPath):
        # load taxonomy file
        df_taxonomy = pd.read_csv(f'{curPath}/data/taxonomy/ncbi_gtdb_processed.csv', low_memory=False) #2190288
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

    def data_loader(self, curPath, dataPath, varType, varSpecies, customSpecies=False, developer=False):
        '''
        curPath: current path(project path)
        dataPath: relative path of data
        varType: data type, fasta or txt
        varSpecies: species that dataset belongs to 
        customSpecies: species is not belongs to iDNA-MS, H.sapiens for False, s__Homo sapiens for True
        developer: the input data has label, for developer testing code
        return: dataset contains sentence
        '''
        if customSpecies == False:
            species_mapped = pd.read_csv(f'{curPath}/data/taxonomy/species_name_mapped.csv')
            species_mapped_dict = dict(zip(species_mapped['abbre_species'], species_mapped['full_species']))
            # apply long species name to initial dataset
            varSpecies = species_mapped_dict[varSpecies]
        taxonomy_lineage = self.get_lineage(varSpecies, curPath)
        if varType == 'fasta':
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
            df = pd.read_csv(os.path.join(curPath, dataPath), sep='\t')
            if developer == False:
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








