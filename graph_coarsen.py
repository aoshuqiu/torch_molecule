import logging
import random
import sys

import dgl
import dgllife
import rdkit
import torch
from rdkit import Chem



# from utils import *
import utils


# an object to memorize the original molecule by setting property
class InitAtomMapping(object):
    def __init__(self, mol):
        self.mol = mol
        for atom in self.mol.GetAtoms():
            atom.SetProp('global_idx', str(atom.GetIdx()))
    
    def get_sub_mol(self, atom_indices):
        sub_mol = utils.get_sub_mol(self.mol, atom_indices)
        return sub_mol
        
        # sub_mol_atom_mapping = {}
        # for atom in sub_mol.GetAtoms():
            

class MolBreaker(object):

    def __init__(self, frag_set, ring_set):
        self.frag_set = frag_set
        self.ring_set = ring_set
    
    def get_frag_candidates(self, mol):
        frag_candidates = {}
        frag_mols = utils.get_fragment_mols(mol)

        # init brics motif
        for patt in frag_mols:
            atom_indices = mol.GetSubstructMatches(patt)
            if len(atom_indices) == 0: continue
            fs = Chem.MolFragmentToSmiles(mol, atom_indices[0])
            frag_candidates[fs] = atom_indices

        # prune
        for fs in list(frag_candidates.keys()):
            if fs not in self.frag_set:
                frag_candidates.pop(fs)
        
        return frag_candidates

    # frag_candidates is used to prune some rings which is only part of fragments in frag_candidates
    def get_ring_candidates(self, mol, frag_candidates={}):
        ring_candidates = {}
        # indicates the fragments which contains ring
        contains_ring = []
        ring_systems, ring_smiles = utils.get_ring_systems(mol)

        # init ring motif
        for rs in set(ring_smiles):
            # keep decompose the multi-ring, if it is
            if rs not in self.ring_set:
                single_ring_systems, single_ring_smiles = utils.get_single_rings(rs)
                ring_systems.extend(single_ring_systems)
                ring_smiles.extend(single_ring_smiles)
        
        # prune the rings
        for i, ring in enumerate(ring_systems):
            rs = ring_smiles[i]
            if rs not in self.ring_set:
                continue

            useless = False
            ring_size = len(ring)

            for fs, atom_indices in frag_candidates.items():
                frag_size = len(atom_indices[0])
                if frag_size < ring_size:
                    continue

                # if the current ring is part of some fragments, the ring will not be used
                for idx in atom_indices:
                    if ring.issubset(idx):
                        contains_ring.append((fs, idx))
                        useless = True
                        break
                
                # can't break the loop here for detecting rings for all fragments
                # if useless:
                #     break

            if useless:
                continue
        
            if rs in ring_candidates:
                ring_candidates[rs].append(tuple(ring))
            else:
                ring_candidates[rs] = [tuple(ring)]

        return ring_candidates, contains_ring

    def get_motif_atom_mapping(self, mol, frag_candidates, ring_candidates, contains_ring):
        covered_atoms = set()
        atom_mapping, motif_mapping = {}, {}
        
        # 1. cover thos atoms in ring_candidates.
        # the reason that we choose the cover the motif in ring_candiates first is
        # the motif in ring_candidates are not part of fragment in frag_candiates.
        # we priorize the cover of ring because ring can be more difficult to generate
        for rs, ring_systems in ring_candidates.items():
            for atoms in ring_systems:

                # if there is any common atoms, ignore current ring
                # because we wanna make sure the connection between motifs will be a set of real chemical keys
                if set(atoms) & covered_atoms:
                    continue
                
                motif_id = len(motif_mapping)
                motif_mapping[motif_id] = (rs, atoms)
                covered_atoms.update(atoms)
                for n in atoms:
                    atom_mapping[n] = motif_id
        
        # 2. cover those atoms in fragments in contains_ring
        # the reason is similar to above mentioned.
        # the fragments which contains ring is more important.
        contains_ring.sort(key=lambda x:len(x[1]), reverse=True) # sorted by the fragment_size
        for fs, atoms in contains_ring:
            if set(atoms) & covered_atoms:
                continue
            motif_id = len(motif_mapping)
            motif_mapping[motif_id] = (fs, atoms)
            covered_atoms.update(atoms)
            for n in atoms:
                atom_mapping[n] = motif_id
        
        # 3. cover other fragments
        atom_num = mol.GetNumAtoms()
        reversed_index = {i:[] for i in range(atom_num)} # to sort the fragments via priority
        for fs, atom_indices in frag_candidates.items():
            # add randomness
            atom_indices = list(atom_indices)
            random.shuffle(atom_indices)
            for atoms in atom_indices:
                if (fs, atom_indices) in contains_ring:
                    continue
                for n in atoms:
                    reversed_index[n].append((fs, atoms))
        
        priority = sorted(reversed_index, key=lambda k: len(reversed_index[k]))
        for i in priority:
            if i in atom_mapping:
                continue
            
            for fs, atoms in reversed_index[i]:
                if set(atoms) & covered_atoms:
                    continue

                motif_id = len(motif_mapping)
                motif_mapping[motif_id] = (fs, atoms)
                covered_atoms.update(atoms)
                for n in atoms:
                    atom_mapping[n] = motif_id

        # 4. turn those uncovered atoms to single motif
        atom_num = mol.GetNumAtoms()
        uncovered_atoms = set(range(atom_num)) - covered_atoms
        for i in uncovered_atoms:
            atom = mol.GetAtomWithIdx(i)
            atom_symbol = atom.GetSymbol()
            atom_charge = atom.GetFormalCharge()
            atom_explicit_hs = atom.GetNumExplicitHs()

            motif_id = len(motif_mapping)
            motif_name = str((atom_symbol, atom_charge, atom_explicit_hs))
            motif_mapping[motif_id] = (motif_name, (i,))
            atom_mapping[i] = motif_id
        
        return motif_mapping, atom_mapping

    def get_edge_mapping(self, mol, atom_mapping):
        edge_mapping = {}
        # kekulize the molecule at first to turn the AROMATIC key to SINGLE or DOUBLE
        Chem.Kekulize(mol)
        for bond in mol.GetBonds():
            atom1 = bond.GetBeginAtomIdx()
            atom2 = bond.GetEndAtomIdx()
            motif1 = atom_mapping[atom1]
            motif2 = atom_mapping[atom2]

            if motif1 != motif2:
                if (motif1, motif2) in edge_mapping:
                    edge_mapping[(motif1, motif2)].append((atom1, atom2, bond.GetBondType()))
                    edge_mapping[(motif2, motif1)].append((atom2, atom1, bond.GetBondType()))
                else:
                    edge_mapping[(motif1, motif2)] = [(atom1, atom2, bond.GetBondType())]
                    edge_mapping[(motif2, motif1)] = [(atom2, atom1, bond.GetBondType())]

        Chem.SanitizeMol(mol)

        # motif_neis = {i:set() for i in range(len(motif_mapping))}
        # for m1, m2 in edge_mapping:
        #     motif_neis[m1].add(m2)
        
        return edge_mapping

    def get_random_bfs_order(self, motif_num, edge_mapping):
        motif_neis = {i:set() for i in range(motif_num)}
        for m1, m2 in edge_mapping:
            motif_neis[m1].add(m2)
        
        init_node = random.randint(0, motif_num-1)
        queue = []
        queue.append(init_node)

        motif_order = []
        ed_motif_bond_mapping = {}
        visited_order = []

        while queue:
            cur = queue.pop(0)
            visited_order.append(cur)

            neis = list(motif_neis[cur])
            random.shuffle(neis)

            for nxt in neis:
                if nxt not in visited_order and nxt not in queue:
                    queue.append(nxt)
                    motif_order.append((cur, nxt))
                    ed_motif_bond_mapping[nxt] = edge_mapping[(cur, nxt)]

                    # if z is before nxt in current order, update the connection of nxt should get
                    for z in motif_neis[nxt]:

                        # first version (deprecated) 
                        if z != cur and (z in queue or z in visited_order):
                        # if z != cur and z in visited_order:
                            ed_motif_bond_mapping[nxt].extend(edge_mapping[(z, nxt)])

        return motif_order, ed_motif_bond_mapping, motif_neis

    def get_train_data(self, mol, motif_mapping, motif_order, motif_neis, ed_bond_mapping):
        
        iam = InitAtomMapping(mol)
        # motif_atoms and partial_atoms shows the atom in these motif or partial graph
        motif_atoms, partial_atoms = {}, {}
        # motif_mols, partial_mols = {}, {}
        for motif_id, (_, atoms) in motif_mapping.items():
            # to control the id mapping
            atoms = sorted(list(atoms))
            motif_atoms[motif_id] = atoms
            # motif_mol = ssvae.utils.get_sub_mol(mol, atoms)
            # motif_mols[motif_id] = motif_mol
                   
        covered_atoms = set()
        # cur_set indicates the component of partial graph
        # the motif_id in cur_set will make up a partial graph
        cur_set = set()
        # nxt_set indicates the next motif connected to the current partial graph
        nxt_set = set()

        '''
        node_train_data is a list of (partial_id(int), nxt_motifs(set(int)))
        
        '''
        node_train_data = []
        
        '''

        '''
        edge_train_data = []

        
        for i, (st_motif_id, ed_motif_id) in enumerate(motif_order):
            st_motif_atoms = motif_atoms[st_motif_id]
            ed_motif_atoms = motif_atoms[ed_motif_id]

            if i == 0:
                # --- data for node predict ---
                # consturct current partial graph
                partial_id = len(motif_atoms) + len(partial_atoms)
                covered_atoms.update(st_motif_atoms)
                used_atoms = sorted(list(covered_atoms))
                partial_atoms[partial_id] = used_atoms
                # partial_mol = ssvae.utils.get_sub_mol(mol, used_atoms)
                # partial_mols[partial_id] = partial_mol

                # update the motif composition
                cur_set.add(st_motif_id)
                nxt_set.update(motif_neis[st_motif_id])
                nxt_set = nxt_set - cur_set

                node_train_data.append((partial_id, tuple(nxt_set)))

            # --- data for edge predict ---
            attachments = []
            motif_edges = ed_bond_mapping[ed_motif_id]

            # construct attachments between partial graph and motif
            for st, ed, bt in motif_edges:

                a1, a2 = -1, -1

                if st in used_atoms:
                    a1 = used_atoms.index(st)
                if ed in ed_motif_atoms:
                    a2 = ed_motif_atoms.index(ed)

                if a1 >= 0 and a2 >= 0:
                    attachments.append((a1, a2, bt))

            edge_train_data.append((partial_id, ed_motif_id, attachments))

            partial_id = len(motif_atoms) + len(partial_atoms)
            covered_atoms.update(motif_atoms[ed_motif_id])
            used_atoms = sorted(list(covered_atoms))
            partial_atoms[partial_id] = used_atoms
            # partial_mol = ssvae.utils.get_sub_mol(mol, used_atoms)
            # partial_mols[partial_id] = partial_mol

            cur_set.add(ed_motif_id)
            nxt_set.update(motif_neis[ed_motif_id])
            nxt_set = nxt_set - cur_set
            node_train_data.append((partial_id, tuple(nxt_set)))

        # merge motif_mols and partial_mols
        sub_mol_atoms = []
        for i in range(len(motif_atoms) + len(partial_atoms)):
            if i in motif_atoms:
                sub_mol_atoms.append(motif_atoms[i])
            else:
                sub_mol_atoms.append(partial_atoms[i])
        
        sub_mols = []
        for i, atoms in enumerate(sub_mol_atoms):
            sub_mols.append(iam.get_sub_mol(atoms))

        # print('motif_atoms', motif_atoms)
        # print('partial_atoms', partial_atoms)

        return node_train_data, edge_train_data, sub_mols, sub_mol_atoms

    def train_prepare(self, smiles, debug=False):
        mol = utils.get_mol_by_smiles(smiles)
        info = {}
        if mol is None:
            return info
        
        iam = InitAtomMapping(mol)
        frag_candidates = self.get_frag_candidates(mol)
        ring_candidates, contains_ring = self.get_ring_candidates(mol, frag_candidates)
        motif_mapping, atom_mapping = self.get_motif_atom_mapping(mol, frag_candidates, ring_candidates, contains_ring)
        edge_mapping = self.get_edge_mapping(mol, atom_mapping)
        motif_order, ed_bond_mapping, motif_neis = self.get_random_bfs_order(len(motif_mapping), edge_mapping)
        node_train_data, edge_train_data, sub_mols, sub_mol_atoms = self.get_train_data(mol, motif_mapping, motif_order, motif_neis, ed_bond_mapping)
        
        dgl_subgraphs = [utils.custom_mol_to_graph(sub_mol) for sub_mol in sub_mols]
        dgl_subgraph_size = [g.num_nodes() for g in dgl_subgraphs]
        batch_dgl_subgraphs = dgl.batch(dgl_subgraphs)

        info = {
            'smiles': smiles,
            'mol_with_prop': iam.mol,
            'atom_mapping': atom_mapping,
            'motif_mapping': motif_mapping,
            'sub_mols': sub_mols,
            'batch_dgl_subgraphs': batch_dgl_subgraphs,
            'dgl_subgraph_size': dgl_subgraph_size,
            'node_train_data': node_train_data,
            'edge_train_data': edge_train_data,
        }

        if debug:
            info['frag_candidates'] = frag_candidates
            info['ring_candidates'] = ring_candidates
            info['contains_ring'] = contains_ring
            info['edge_mapping'] = edge_mapping
            info['motif_order'] = motif_order
            info['ed_bond_mapping'] = ed_bond_mapping
            info['motif_neis'] = motif_neis
            info['sub_mol_atoms'] = sub_mol_atoms
    
        return info

        # except Exception as e:
        #     print('in turning_for_train', smiles, e)