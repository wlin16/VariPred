import pandas as pd
import re
import requests, sys
from tqdm import tqdm

import numpy as np
import pickle
from Bio import Entrez
from Bio import SeqIO


# convert HGVSp variant representation into gene_id, aa_index. wt_aa, and mt_aa
def get_id(input_id):
    txt = re.search( r"([A-Z]P_[0-9]+.[0-9])(:p.)([A-Z][a-z]+)([0-9]+)([A-Z][a-z]+)", input_id)
    NP_id = txt.group(1)
    wt = txt.group(3)
    aa_index = txt.group(4)
    mt = txt.group(5)
    return NP_id, aa_index, wt, mt

# convert amino_acid abbreviations into amino_acid_codes
d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
def shorten(x):
    x = x.upper()
    if len(x) % 3 != 0:
        raise ValueError('Input length should be a multiple of three')

    y = ''
    for i in range(len(x) // 3):
        y += d[x[3 * i : 3 * i + 3]]
    return y

def df_process (input_df):
    input_df['NP_id'], input_df['aa_index'], input_df['wt_aa'], input_df['mt_aa'] = zip(*input_df['target_id'].apply(get_id))
    input_df['wt_aa'] = input_df['wt_aa'].apply(shorten)
    input_df['mt_aa'] = input_df['mt_aa'].apply(shorten)
    if 'label' in input_df.columns:
        input_df = input_df[['target_id','NP_id','aa_index','wt_aa','mt_aa','label']]
    else:
        input_df = input_df[['target_id','NP_id','aa_index','wt_aa','mt_aa']]
    return input_df

# generate mt sequences
def create_mt_sequence(df):
    mt_sequence = []
    length = []
    for index, row in df.iterrows():
        seq_len = len(row['wt_seq'])
        string = row['wt_seq']
        posn = int(row['aa_index'])-1
        nc = row['mt_aa']
        result = string[:posn] + nc + string[posn+1:]
        length.append(seq_len)
        mt_sequence.append(result)
    df['mt_seq'] = mt_sequence
    df['Length'] = length
    df['Length'] = df['Length'].apply(lambda x: int(x))
    df['aa_index'] = df['aa_index'].apply(lambda x: int(x))
    
    if 'label' in df.columns:
        column_order = ['target_id','NP_id','aa_index','Length','wt_aa','mt_aa','wt_seq','mt_seq','label']
    else:
        column_order = ['target_id','NP_id','aa_index','Length','wt_aa','mt_aa','wt_seq','mt_seq']
    
    df = df[column_order]
    return df


def NCBI_crawl(NP_id):
    Entrez.email = 'your@email.com'
    handle = Entrez.efetch(db="protein", id=NP_id,
                            rettype="gb", retmode='text')
    record = handle.read()
    for i in record.split('\n'):
            if "DBSOURCE" in i:
                    NM_id = i.split(' ')[-1]
                    handle = Entrez.efetch(db="nucleotide", id=NM_id, rettype="fasta_cds_aa", retmode='text')
                    record = SeqIO.read(handle, format='fasta')
                #     print(record.seq)
    return str(record.seq)

def fetch_seq(df):
    for index, row in df.iterrows():
        seq = NCBI_crawl(row['NP_id'])
        df.loc[index,'wt_seq'] = seq
    return df
        
def validable(df):
    error_data = len(df.query("aa_index > Length"))
    wt_counter = 0
    mt_counter = 0
    
    filtered_df = df.query("aa_index <= Length")
    
    for i, row in filtered_df.iterrows():
        if row.wt_seq[row.aa_index - 1] != row.wt_aa:
            wt_counter +=1
        if row.mt_seq[row.aa_index - 1] != row.mt_aa:
            mt_counter +=1
    
    total_error = error_data + wt_counter
    return total_error


if __name__ == '__main__':
    # seq provided by 3CNet group
    transcript_seq = pd.read_csv('transcript_seq.csv')
    
    # clinvar variant example with pathogenicity label (only with missense mutation)
    example = pd.read_csv('./example.txt', sep = '\s')
    example = df_process(example)



    ################



    # find the genes not included in the provided transcript list
    difference = example[~example.NP_id.isin(transcript_seq.NP_id)]
    included = example[example.NP_id.isin(transcript_seq.NP_id)]

    # fetch wt_seq for each gene
    if len(difference) > 0:
        # fetch wt_seq from NCBI for not founded ids, and create correponing mt_seq for each variant
        print(f'There are {len(difference)} genes need to find wt_seq from NCBI')
        difference = fetch_seq(difference)
        difference = create_mt_sequence(difference) 

    included = pd.merge(included, transcript_seq, on='NP_id', how='left').dropna(axis = 0)
    included = create_mt_sequence(included)

    if len(difference) > 0:
        df = pd.concat([included,difference])
    else:
        df = included



    ################



    # valid how many wt_seq are not correct:
    error_counter = validable(df)

    if error_counter == 0:
        print('All wt_seq are correct')
        df.to_csv('../example/dataset/target.csv')
    else:
        print(f'There are {error_counter} wt_seq are not correctly fetched from NCBI')
        print('dataframe is not saved')






