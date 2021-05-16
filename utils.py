import copy
import torch
import sys
import re

import rdkit
from rdkit import Chem
from rdkit.Chem import rdBase
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import BRICS, Recap
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

import dgllife
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer


'''
chemical utils
'''
# TODO
ATOM_CANDIDATE = [('C', 0, 0), ('N', 0, 0), ('O', 0, 0), ('S', 0, 0), ('F', 0, 0), ('Cl', 0, 0), ('N', 0, 1), ('Br', 0, 0)]
ATOM_CANDIDATE_DICT = {x:i for i, x in enumerate(ATOM_CANDIDATE)}
ATOM_CANDIDATE_LEN = len(ATOM_CANDIDATE)

BOND_CANDIDATE = [
    None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC
]
BOND_CANDIDATE_DICT = {x:i for i, x in enumerate(BOND_CANDIDATE)}
BOND_CANDIDATE_LEN = len(BOND_CANDIDATE)

MAX_VALENCE = {'B': 3, 'Br':1, 'C':4, 'Cl':1, 'F':1, 'I':1, 'N':5, 'O':2, 'P':5, 'S':6, 'Se':4, 'Si':4}

# more tolerant way
def get_mol_by_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_KEKULIZE^Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)
            return mol
        except:
            return None
    return mol

def valence_check(atom, bt):
    cur_val = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
    return cur_val + bt <= MAX_VALENCE[atom.GetSymbol()]

def valence_remain(atom):
    cur_val = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
    return MAX_VALENCE[atom.GetSymbol()] - int(cur_val)

def get_candidate_atoms(mol):
    potential_atoms = {}
    for atom in mol.GetAtoms():
        max_bt = min(valence_remain(atom), 3)
        if max_bt > 0:
            aid = atom.GetIdx()
            potential_atoms[aid] = max_bt
    return potential_atoms

# def add_atom_indices(mol):
#     for i, a in enumerate(mol.GetAtoms()):
#         a.SetAtomMapNum(i)

def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

def mol_with_atom_prop(mol, prop):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetProp(prop))
    return mol

# mol should be rwmol
# copy a sub-molecule from original molecule by given atom indices
def get_sub_mol(mol, sub_atoms, prop=None):
    new_mol = Chem.RWMol()
    atom_map = {}
    # sub_atoms = set(sub_atoms)
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())
        
        if atom.HasProp('global_idx'):
            new_atom.SetProp('global_idx', atom.GetProp('global_idx'))

        atom_map[idx] = new_mol.AddAtom(new_atom)

    for idx in sub_atoms:
        a = mol.GetAtomWithIdx(idx)
        for b in a.GetNeighbors():
            if b.GetIdx() not in sub_atoms: continue
            bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
            bt = bond.GetBondType()
            if a.GetIdx() < b.GetIdx(): #each bond is enumerated twice
                new_mol.AddBond(atom_map[a.GetIdx()], atom_map[b.GetIdx()], bt)
    return new_mol.GetMol()

def decode_stereo(smiles2D):
    mol = Chem.MolFromSmiles(smiles2D)
    dec_isomers = list(EnumerateStereoisomers(mol))
    dec_isomers = [Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=True)) for mol in dec_isomers]
    smiles3D = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in dec_isomers]
    chiralN = [atom.GetIdx() for atom in dec_isomers[0].GetAtoms() if int(atom.GetChiralTag()) > 0 and atom.GetSymbol() == "N"]
    if len(chiralN) > 0:
        for mol in dec_isomers:
            for idx in chiralN:
                mol.GetAtomWithIdx(idx).SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            smiles3D.append(Chem.MolToSmiles(mol, isomericSmiles=True))
    return smiles3D

# from RDkit cookbook
def GetRingSystems(mol, includeSpiro=False):
    ri = mol.GetRingInfo()
    systems = []
    for ring in ri.AtomRings():
        ringAts = set(ring)
        nSystems = []
        for system in systems:
            nInCommon = len(ringAts.intersection(system))
            if nInCommon and (includeSpiro or nInCommon>1):
                ringAts = ringAts.union(system)
            else:
                nSystems.append(system)
        nSystems.append(ringAts)
        systems = nSystems
    return systems

# return ring indices and ring smiles
def get_ring_systems(mol):
    ring_system = GetRingSystems(mol)
    detected_rings = [Chem.MolFragmentToSmiles(mol, atom_indices) for atom_indices in ring_system]
    return ring_system, detected_rings

# return single rings from smiles
# usually used to detach multi-ring
def get_single_rings(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return [], []
    ri = mol.GetRingInfo()
    systems = []
    for ring in ri.AtomRings():
        ringAts = set(ring)
        systems.append(ringAts)
    detected_rings = [Chem.MolFragmentToSmiles(mol, atom_indices) for atom_indices in systems]
    return systems, detected_rings

# get the fragment mols via some decomposing methods
def get_fragment_mols(mol, t='BRICS', min_frag_size=2):
    if t == 'BRICS':
        frags = BRICS.BRICSDecompose(mol,keepNonLeafNodes=False,minFragmentSize=min_frag_size)
        # replace dummy atom with hydrogen atom
        frags = [re.sub(r"(\[[0-9]*\*\])", "[H]", smi) for smi in frags]
        frags = [Chem.MolFromSmiles(s) for s in frags]
    elif t == 'Recap':
        frags = [Chem.MolToSmiles(x.mol) for x in Recap.RecapDecompose(mol,minFragmentSize=min_frag_size).GetLeaves().values()]
        frags = [re.sub(r"(\*)", "[H]", smi) for smi in frags]
        frags = [Chem.MolFromSmiles(s) for s in frags]
    else:
        frags = []
    return frags


    
'''
nn utils
'''
def get_atom_dict_idx(atom):
    symbol = atom.GetSymbol()
    charge = atom.GetFormalCharge()
    exp_hs = atom.GetNumExplicitHs()
    idx = ATOM_CANDIDATE_DICT.get((symbol, charge, exp_hs), -1)
    return idx

def get_bond_dict_idx(bond):
    bt = bond.GetBondType()
    idx = BOND_CANDIDATE_DICT.get(bt, -1)
    return idx

canonical_node_featurizer = CanonicalAtomFeaturizer(atom_data_field='atom_feat')
canonical_edge_featurizer = CanonicalBondFeaturizer(bond_data_field='bond_feat')

def custom_mol_to_graph(mol):
    if mol is None:
        return None
    
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_KEKULIZE^Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)

    g = dgllife.utils.mol_to_bigraph(mol, node_featurizer=canonical_node_featurizer, edge_featurizer=canonical_edge_featurizer, canonical_atom_order=False)

    return g

    # update 4.27 put these into ssvae's encoder

    # # no need to deal with motif feat
    # if atom_mapping is None:
    #     return g
    
    # motif_feat_mapping = {}
    # for k,v in motif_feat_idx.items():
    #     motif_feat = torch.zeros(merge_vocab_size)
    #     motif_feat[v] = 1
    #     motif_feat_mapping[k] = motif_feat

    # motif_feats = []
    # for atom_id in range(g.num_nodes()):
    #     motif_id = atom_mapping[atom_id]
    #     motif_feats.append(motif_feat_mapping[motif_id])
    
    # g.ndata['motif_feat'] = torch.stack(motif_feats)
    # return g

'''
miscellaneous utils
'''
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