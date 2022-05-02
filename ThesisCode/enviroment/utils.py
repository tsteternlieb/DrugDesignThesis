from rdkit import Chem
import numpy as np
import dgl
import dgllife
import torch
import heapq
import wandb




def fixOrder(mol):
    l = [atom.GetDegree() for atom in mol.GetAtoms()]
    if 0 in l:
        idx = l.index(0)
        mol = permuteAtomToEnd(mol,idx)
    return mol



def bondTypeInt(bond_type):
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.AROMATIC]
    return bond_types.index(bond_type)


def permute_mol(mol, permute):
    new_mol = Chem.RWMol()
    atoms = [atom for atom in mol.GetAtoms()]
    new_atom_list = [0 for _ in range(mol.GetNumAtoms())]
    for atom in mol.GetAtoms():
        new_atom_list[permute(atom.GetIdx())] = atom
    for atom in new_atom_list:
        new_mol.AddAtom(atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(permute(a1), permute(a2), bt)
    return new_mol


def permute_rot(domain, rot_num=1):
    l1 = list(range(domain))
    l2 = list(range(domain))
    l2 = l2[rot_num:] + l2[:rot_num]
    p = dict(zip(l1, l2))

    def _permute(num):
        return p[num]
    return _permute


def _permute_atom_to_end(domain, atom_idx):
    l1 = list(range(domain))
    l2 = list(range(domain))
    l2[atom_idx], l2[-1] = l2[-1], l2[atom_idx]
    p = dict(zip(l1, l2))

    def _permute(num):
        return p[num]
    return _permute


def permuteAtomToEnd(mol, atom_idx):
    return permute_mol(mol, permute=_permute_atom_to_end(mol.GetNumAtoms(), atom_idx))


def SanitizeNoKEKU(mol):
    s_dict = {'SANITIZE_ADJUSTHS': Chem.rdmolops.SanitizeFlags.SANITIZE_ADJUSTHS,
              'SANITIZE_ALL': Chem.rdmolops.SanitizeFlags.SANITIZE_ALL,
              'SANITIZE_CLEANUP': Chem.rdmolops.SanitizeFlags.SANITIZE_CLEANUP,
              'SANITIZE_CLEANUPCHIRALITY': Chem.rdmolops.SanitizeFlags.SANITIZE_CLEANUPCHIRALITY,
              'SANITIZE_FINDRADICALS': Chem.rdmolops.SanitizeFlags.SANITIZE_FINDRADICALS,
              'SANITIZE_KEKULIZE': Chem.rdmolops.SanitizeFlags.SANITIZE_KEKULIZE,
              'SANITIZE_NONE': Chem.rdmolops.SanitizeFlags.SANITIZE_NONE,
              'SANITIZE_PROPERTIES': Chem.rdmolops.SanitizeFlags.SANITIZE_PROPERTIES,
              'SANITIZE_SETAROMATICITY': Chem.rdmolops.SanitizeFlags.SANITIZE_SETAROMATICITY,
              'SANITIZE_SETCONJUGATION': Chem.rdmolops.SanitizeFlags.SANITIZE_SETCONJUGATION,
              'SANITIZE_SETHYBRIDIZATION': Chem.rdmolops.SanitizeFlags.SANITIZE_SETHYBRIDIZATION,
              'SANITIZE_SYMMRINGS': Chem.rdmolops.SanitizeFlags.SANITIZE_SYMMRINGS}

    #mol = Chem.SanitizeMol(mol,s_dict['SANITIZE_KEKULIZE'])
    Chem.SanitizeMol(mol, s_dict['SANITIZE_ADJUSTHS'] | s_dict['SANITIZE_SETAROMATICITY'] |
                     s_dict['SANITIZE_CLEANUP'] | s_dict['SANITIZE_CLEANUPCHIRALITY'] |
                     s_dict['SANITIZE_FINDRADICALS'] | s_dict['SANITIZE_NONE'] |
                     s_dict['SANITIZE_PROPERTIES'] | s_dict['SANITIZE_SETCONJUGATION'] |
                     s_dict['SANITIZE_SETHYBRIDIZATION'] | s_dict['SANITIZE_SYMMRINGS']
                     )


def FeatToAtomAroFULL(feat):
    atom_list = ['N', 'C', 'O', 'S', 'F', 'Cl',
                 'Na', 'P', 'Br', 'Si', 'B', 'Se', 'K']
    atom_type_slice = feat[0:14]

    atom_type_idx = np.where(atom_type_slice.cpu() == 1)
    atom_type_idx = atom_type_idx[0][0]
    atom_type = atom_list[atom_type_idx]

    return atom_type


def MolFromGraphsFULL(graph):

    # create empty editable mol object
    feat_list = graph.ndata['atomic']

    node_list = []
    for feat in range(feat_list.shape[0]):
        node_list.append(FeatToAtomAroFULL(feat_list[feat]))

    mol = Chem.RWMol()

    # add atoms to mol and keep track of index
    node_to_idx = {}
    for i in range(len(node_list)):
        a = Chem.Atom(node_list[i])
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    for u in range(len(node_list)-1):
        for v in range(u+1, len(node_list)):
            if graph.has_edges_between(u, v):
                bond_type = int(graph.edges[u, v][0]
                                ['type'].cpu().numpy()[0][0])
                if bond_type == 1:
                    bond = Chem.rdchem.BondType.SINGLE
                elif bond_type == 2:
                    bond = Chem.rdchem.BondType.DOUBLE
                elif bond_type == 3:
                    bond = Chem.rdchem.BondType.AROMATIC
                else:
                    print("graelfsdjhkarg", bond_type, "asdf")
                mol.AddBond(u, v, bond)
    mol = mol.GetMol()

    return mol


def CustomAtomFeaturizer_full(mol):
    '''
    atom type, bond_number, formal charge, chirality, 
    number of bonded h atoms, hybridization, aromaticity,
    atomic mass scaled

    '''
    feats = []

    atom_bond_dict = {'N': [1, 0, 5], 'C': [2, 0, 4], 'O': [3, 0, 6], 'S': [4, 0, 6],
                      'F': [5, 0, 7], 'Cl': [6, 0, 7], 'Na': [7, 0, 7], 'P': [8, 0, 5],
                      'Br': [9, 0, 7], 'Si': [10, 0, 4], 'B': [11, 0, 5], 'Se': [12, 0, 6],
                      'K': [13, 0, 7]}

    hybridization_ids = ["SP", "SP2", "SP3"]

    Chem.SetHybridization(mol)

    for atom in mol.GetAtoms():
        atom.UpdatePropertyCache()
        atom_symbol = atom.GetSymbol()
        atom_idx = atom_bond_dict[atom_symbol][0]-1

        atom_oh = np.zeros(15)
        atom_oh[atom_idx] = 1

        max_valence = atom_bond_dict[atom_symbol][-1]
        max_valence_oh = np.zeros(8)
        max_valence_oh[max_valence] = 1

        degree = dgllife.utils.atom_degree_one_hot(atom)

        hybridization = dgllife.utils.atom_hybridization_one_hot(atom)

        is_aromatic = np.expand_dims(np.asarray(
            atom.GetIsAromatic()).astype(int), 0)  # Bool

        exp_valence = dgllife.utils.atom_explicit_valence_one_hot(atom)
        imp_valence = dgllife.utils.atom_implicit_valence_one_hot(atom)

        mass = np.expand_dims(np.asarray(atom.GetMass()) /
                              127, 0)  # max val of 126.904

        feat = np.concatenate((atom_oh, max_valence_oh, degree,
                               hybridization, is_aromatic, exp_valence, imp_valence, mass))

        feats.append(feat)
    return {'atomic': torch.tensor(feats).float()}


def edge_featurizer_full(mol, add_self_loop=False):
    feats = []
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.AROMATIC]
    for bond in mol.GetBonds():
        btype = bond_types.index(bond.GetBondType())+1
        # One bond between atom u and v corresponds to two edges (u, v) and (v, u)
        feats.extend([btype, btype])
    return {'type': torch.tensor(feats).reshape(-1, 1).float()}


def selfLoop(graph):
    return dgl.add_self_loop(dgl.remove_self_loop(graph))


def mol_to_graph_full(mol):
    graph = dgllife.utils.mol_to_bigraph(mol, node_featurizer=CustomAtomFeaturizer_full, edge_featurizer=edge_featurizer_full,
                                         canonical_atom_order=False)
    graph = selfLoop(graph)
    return graph
