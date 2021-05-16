import gym
import itertools
import numpy as np

from rdkit import Chem  # TODO(Bowen): remove and just use AllChem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import qed, MolLogP
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog
# import gym_molecule
import copy
import networkx as nx
from gym_molecule.envs.sascorer import calculateScore
from gym_molecule.dataset.dataset_utils import gdb_dataset,mol_to_nx,nx_to_mol
import random
import time
import matplotlib.pyplot as plt
import csv

from contextlib import contextmanager
import sys, os

from rdkit import RDLogger
from rdkit.rdBase import DisableLog
from rdkit.rdBase import BlockLogs

class Vocab(object):
    def __init__(self, vocab_list):
        self.vocab_list = list(vocab_list)
        self.vmap = {x: i for i, x in enumerate(self.vocab_list)}
        self.length = len(self.vocab_list)
        # if one_hot_perpare:

    def get_vocab_idx(self, vocab):
        return self.vmap.get(vocab, -1)

    def __getitem__(self, idx):
        return self.vocab_list[idx]

    def size(self):
        return self.length

def get_vocab_counter(file_path):
    counter = {}
    with open(file_path) as f:
        try:
            for line in f:
                line = line.strip()
                smi, times = line.split('\t')
                counter[smi] = int(times)
        except:
            pass
    return counter

def get_vocab_set(counter, threshold):
    vocab_set = set()
    for k, v in counter.items():
        if v >= threshold:
            vocab_set.add(k)
    return vocab_set

# 利用从数据中拆分的motif，用频率高的motif形成vocab
frag_counter = get_vocab_counter("./dataset/moses/counter1_leaf/fragment_counter.txt")
ring_counter = get_vocab_counter("./dataset/moses/counter1_leaf/ring_counter.txt")
frag_set = get_vocab_set(frag_counter, 500)
ring_set = get_vocab_set(ring_counter, 50)
vocab = Vocab(frag_set.union(ring_set))
count = 0
for v in vocab.vocab_list:
    mol = Chem.MolFromSmiles(v)
    if mol:
        atom_num = mol.GetNumAtoms()
        if atom_num > count:
            count = atom_num
print(count)