import tensorflow as tf
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
import os
import re
import math
import random
from molvecgen import SmilesVectorizer
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.MACCSkeys import GenMACCSKeys
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
import pubchempy as pcp
import enformer
from numpy2tfrecord import Numpy2TFRecordConverter

########################################################################################

anti_cid_dict = {'Amikacin':37768,'Ampicillin':6249,'Ampicillin/Sulbactam':74331798,'Aztreonam':5742832,'Cefazolin':33255,'Cefepime':5479537,'Cefoxitin':441199,'Ceftazidime':5481173,'Ceftriaxone':5479530,'Cefuroxime sodium':23670318,
                'Ciprofloxacin':2764,'Gentamicin':3467,'Imipenem':104838,'Levofloxacin':149096,'Meropenem':441130,'Nitrofurantoin':6604200,'Piperacillin/Tazobactam':9918881,'Tetracycline':54675776,'Tobramycin':36294,'Trimethoprim/Sulfamethoxazole':358641}
seq_dir = './data/amr_genes_aligned/'

########################################################################################

def getSeqMat():
    df = pd.read_csv('./data/data_map.csv')
    seq_data = []
    for index, row in df.iterrows():
       seq_data.append(row['PATRIC ID'])
    seq_data = set(seq_data)
    seq_mat_dict = {}
    for key in tqdm(seq_data):
        seq_mat_dict[key] = convertSeq(key)
    return seq_mat_dict

def getAntiMat(anti_cid_dict):
    anti_smiles_dict = {}
    for key in anti_cid_dict:
        anti_smiles_dict[key] = pcp.Compound.from_cid(anti_cid_dict[key]).isomeric_smiles
    anti_mol_dict = {}
    for key in anti_smiles_dict:
        anti_mol_dict[key] = MolFromSmiles(anti_smiles_dict[key])
    anti_mat_dict = {}
    encoder = SmilesVectorizer()
    for key in tqdm(anti_mol_dict):
        mol = [anti_mol_dict[key]]
        anti_mat_dict[key] = encoder.transform(mol)[0].reshape(650,4)
        for i in range(8):
            anti_mat_dict[key] = np.vstack((anti_mat_dict[key], anti_mat_dict[key]))
        anti_mat_dict[key] = (anti_mat_dict[key][68096:])
        assert len(anti_mat_dict[key]) == 196608 // 2
    return anti_mat_dict

def convertMIC(s):
    new_s = re.sub('\>([0-9]+\.*[0-9]*)/*.*$', lambda m: str(float(m.group(1))*2), s)
    new_s = re.sub('\<([0-9]+\.*[0-9]*)/*.*$', lambda m: str(float(m.group(1))/2), new_s)
    new_s = re.sub('\<=([0-9]+\.*[0-9]*)/*.*$', lambda m: str(float(m.group(1))), new_s)
    new_s = re.sub('^([0-9]+\.*[0-9]*)/*.*$', lambda m: str(float(m.group(1))), new_s)
    return float(new_s)

def convertSeq(filename):
    with open(seq_dir + '{:0.5f}'.format(filename) , 'r') as data_file:
        seq = data_file.read()
    mat = enformer.one_hot_encode(seq).astype(np.float32)
    return mat

def getListData():
    df = pd.read_csv('./data/data_map.csv')
    df['MIC'] = df['Actual MIC'].apply(convertMIC)
    lst_data = []
    for index, row in df.iterrows():
       lst_data.append({'PATRIC ID':row['PATRIC ID'], 'MIC': int(math.log2(row['MIC'])), 'ANTI': row['Antibiotic']})
    return lst_data

def getMatData(lst_data):
    random.shuffle(lst_data)
    mic_label_lst = [[1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0.5, 1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0.5, 1, 0.5, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0.5, 1, 0.5, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0.5, 1, 0.5, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0.5, 1, 0.5, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0.5, 1, 0.5, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1, 0.5],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1]]
    mat_lst = [
        {
            'seq_mat': (seq_mat_dict[item['PATRIC ID']] + anti_mat_dict[item['ANTI']]) / 2,
            'mic_label': np.array(mic_label_lst[item['MIC'] + 3]).astype(np.float32)
        }
        for item in tqdm(lst_data)
    ]
    return mat_lst

if __name__=='__main__':
    print()
    lst_data = getListData()
    print("lst_data: ", len(lst_data))
    random.shuffle(lst_data)
    antibiotics_lst = list(anti_cid_dict.keys())
    divide_by_anti = [[] for i in range(len(antibiotics_lst))]
    for item in lst_data:
        divide_by_anti[antibiotics_lst.index(item['ANTI'])].append(item)
    splits = [[] for i in range(11)]
    for item_lst in divide_by_anti:
        for idx, item in enumerate(item_lst):
            splits[idx % 11].append(item)
    anti_mat_dict = getAntiMat(anti_cid_dict)
    print('anti_key_dict: ' + str(len(anti_mat_dict)))
    seq_mat_dict = getSeqMat()
    print('seq_mat_dict: ' + str(len(seq_mat_dict)))
    print()

    test_set = splits.pop(5)
    train_val_set = []

    for idx, split in enumerate(splits):
        val_set = splits[idx]
        train_set = []
        for i in range(len(splits)):
            if i != idx:
                train_set += splits[i]
        train_val_set.append((val_set, train_set))

    mat_test_lst = getMatData(test_set)
    with Numpy2TFRecordConverter('./data/tfrecord_data/test_data.tfrecord') as converter:
        converter.convert_list(mat_test_lst)

    # for idx, item in enumerate(train_val_set):
    item = train_val_set[0]
    mat_val_lst = getMatData(item[0])
    with Numpy2TFRecordConverter('./data/tfrecord_data/val_data.tfrecord') as converter:
        converter.convert_list(mat_val_lst)
    mat_train_lst = getMatData(item[1])
    with Numpy2TFRecordConverter('./data/tfrecord_data/train_data.tfrecord') as converter:
        converter.convert_list(mat_train_lst)