import sys
sys.path.append('..')

from enviroment.utils import permuteAtomToEnd, mol_to_graph_full, permute_mol, bondTypeInt, selfLoop, fixOrder
from enviroment.Keku import AdjustAromaticNs
import dgllife
import os, torch, dgl, networkx as nx
from rdkit import Chem
from collections import Counter

device = None


def __remove_edge(mol, atom1_idx, atom2_idx,len_node_actions=17):
    if atom1_idx == mol.GetNumAtoms()-1:
        atom1_idx, atom2_idx = atom2_idx, atom1_idx
    new_mol = permuteAtomToEnd(mol,atom2_idx)
    b = new_mol.GetBondBetweenAtoms(new_mol.GetNumAtoms()-1,atom1_idx)
    bt_int = b.GetBondTypeAsDouble() - 1
        
    new_mol.RemoveBond(new_mol.GetNumAtoms()-1,atom1_idx)
    action =  2*atom1_idx + len_node_actions + bt_int
    
    return new_mol, action

def __remove_node(mol):
    
    atom_list = ['N','C','O','S','F','Cl','Na','P','Br','Si','B','Se','K', 'Aro']    
    for atom in mol.GetAtoms():
        if atom.GetDegree() == 0:
            atom_idx = atom.GetIdx()
            atom_symbol = atom.GetSymbol() 

            action = atom_list.index(atom_symbol) + 1
            mol.RemoveAtom(atom_idx)
            
            
            try:
                Chem.SanitizeMol(mol)
                graph = mol_to_graph_full(mol)
                last_action_node = torch.zeros((1,1)).to(device)
                obs = graph, last_action_node, torch.unsqueeze(graph.ndata['atomic'][-1], dim = 0)                
                return mol, obs, action, True
            except Chem.rdchem.KekulizeException:
                return (None,None,None), -1, False 

    return None, (None,None,None), -1, False
        
def __remove_aro_ring(mol):
    #decide on aromatic ring ordering 
    #ORDERING WILL BE OFF UNTILL DECIDING ON HOW MANY RINGS TO ALLOW
    aro_dict_list = [{'C': 6}, {'N': 1, 'C': 5}, {'N': 1, 'C': 4}]
    aro_list = ['Benz', 'Pyri', 'Pyrr']
    cycles = list(mol.GetRingInfo().AtomRings())
    
    for cycle in cycles:
        if all([mol.GetAtomWithIdx(atom_id).GetIsAromatic() for atom_id in cycle]) and all([mol.GetAtomWithIdx(atom_id).GetDegree() == 2 for atom_id in cycle]):
            aro_cycle = cycle
            atom_types_dict = (Counter([mol.GetAtomWithIdx(atom_id).GetSymbol() for atom_id in cycle]))
            
            try:
                aro_idx = aro_dict_list.index(atom_types_dict)
            except:
                continue
            aro_idx_name = aro_list[aro_idx]
    
            for atom_idx in sorted(aro_cycle, reverse = True):
                mol.RemoveAtom(atom_idx)
            action = 1 + 13 + aro_idx
            
            if mol.GetNumAtoms() == 0:
                    return None, (None,None,None), -1, False 
            try:
                Chem.SanitizeMol(mol)
                graph = mol_to_graph_full(mol)
                last_action_node = torch.zeros((1,1)).to(device)
                obs = graph, last_action_node, torch.unsqueeze(graph.ndata['atomic'][-1], dim = 0)   
                
                return mol, obs, action, True
            except Chem.rdchem.KekulizeException:
                return None, (None,None,None), -1, False 
        
    
    return None, (None,None,None), -1, False 
            
def __remove_node_edge(mol):    
    for atom in mol.GetAtoms():
        if atom.GetDegree() == 1:
            atom_idx = atom.GetIdx()
            b = list(atom.GetBonds())[0]
            other_atom_idx = b.GetOtherAtom(atom).GetIdx()
            new_mol, action = __remove_edge(mol,other_atom_idx,atom_idx)
            
            try:
                Chem.SanitizeMol(new_mol)
                graph = mol_to_graph_full(new_mol)
                last_action_node = torch.ones((1,1)).to(device)
                obs = graph, last_action_node, torch.unsqueeze(graph.ndata['atomic'][-1], dim = 0)                
                return new_mol, obs, action, True
            except Chem.rdchem.KekulizeException:
                return None, (None,None,None), -1, False 
            
                
    
    return None, (None,None,None), -1, False              
    
def __remove_ring_edge(mol):
    cycles = list(mol.GetRingInfo().AtomRings())
    for cycle in cycles:
        degrees = [mol.GetAtomWithIdx(cycle_idx).GetDegree() for cycle_idx in cycle] 
        is_aromatic = [mol.GetAtomWithIdx(cycle_idx).GetIsAromatic() for cycle_idx in cycle] 
        if sum(degrees) == (len(degrees) * 2) + 1 and all(is_aromatic):
            idx_in_cycle = degrees.index(3)
            
            atom_idx = cycle[idx_in_cycle]
            atom = mol.GetAtomWithIdx(atom_idx)
            b_list = list(atom.GetBonds())
            
            for b in b_list:
                bt = b.GetBondType()
                if bt != Chem.rdchem.BondType.AROMATIC:
                    break
            else:
                continue

            other_atom_idx = b.GetOtherAtom(atom).GetIdx()
            new_mol, action = __remove_edge(mol,atom_idx,other_atom_idx)
            
            
            
            try:
                Chem.SanitizeMol(new_mol)
                graph = mol_to_graph_full(new_mol)
                last_action_node = torch.ones((1,1)).to(device)
                obs = graph, last_action_node, torch.unsqueeze(graph.ndata['atomic'][-1], dim = 0)                
                return new_mol, obs, action, True
            except Chem.rdchem.KekulizeException:
                return None, (None,None,None), -1, False 
                
        
    
    return None, (None,None,None), -1, False 
            
def __break_cycle(mol):
        for atom_i in range(mol.GetNumAtoms()):
            if mol.GetAtomWithIdx(atom_i).GetIsAromatic():
                continue
            for atom_j in range(atom_i+1, mol.GetNumAtoms()):
                if mol.GetAtomWithIdx(atom_j).GetIsAromatic():
                    continue
                bond = mol.GetBondBetweenAtoms(atom_i,atom_j)
                if bond == None:
                    continue
                
                bond_type_int = bondTypeInt(bond.GetBondType())
                mol_copy = permute_mol(mol,lambda x: x)
                mol_copy.RemoveBond(atom_i,atom_j)
                if not nx.is_connected(mol_to_graph_full(mol_copy).to_networkx().to_undirected()):
                    continue
                new_mol, action = __remove_edge(mol,atom_i,atom_j)
                graph = mol_to_graph_full(new_mol)
                last_action_node = torch.ones((1,1)).to(device)
                obs = graph, last_action_node, torch.unsqueeze(graph.ndata['atomic'][-1], dim = 0)
                Chem.SanitizeMol(new_mol)
                return new_mol, obs, action, True
        else:
            return None, (None,None,None), -1, False 
            
            
def __molDecompStep(mol):
    '''
    input: RDKit Mol 
    Output: (RDKit Mol, last action node, node features) 
    
    Priority:
    
    1. Remove Lone Node/Aromatic ring
    2. Disconnect Aromatic Ring
    3. Disconnect Atom
    4. Break Cycle
    '''
    mol_copy = permute_mol(mol, lambda x: x) 
    Chem.SanitizeMol(mol_copy)
    success = False
    

    
    
    if not success:
        mol, obs, action, success = __remove_node(mol_copy)

    if not success:
        mol, obs, action, success = __remove_aro_ring(mol_copy)
        
    if not success:
        mol, obs, action, success = __remove_ring_edge(mol_copy)
        
    if not success:
        mol, obs, action, success = __remove_node_edge(mol_copy)    
              
    if not success:
        mol, obs, action, success = __break_cycle(mol_copy)
      
    
    g = dgllife.utils.mol_to_bigraph(mol)
    if g != None:
        g = g.to_networkx().to_undirected()        
        num_connected = nx.number_connected_components(g)    

        if success and mol.GetNumAtoms()>1: #and num_connected<3:
            AdjustAromaticNs(mol) #added
            if mol != None:
                obs = selfLoop(obs[0]),obs[1],obs[2]
                return mol, obs, action, success
            else:
                return None,None,None, False 
    
    return None, None, None, False
    
def __fullMolDecomp(mol, return_mol = False):
    

    graph = selfLoop(mol_to_graph_full(mol))
    last_action_node = torch.zeros((1,1)).to(device)
    
    #states = [graph, last_action_node, torch.unsqueeze(graph.ndata['atomic'][-1],dim=0)]
    obs = (graph, last_action_node, torch.unsqueeze(graph.ndata['atomic'][-1],dim=0))
    states = [obs]
    actions = [torch.zeros(1)]
    mol_list = [mol]
    
    
    i = 1
    success = True
    while success:
        mol, obs, action, success = __molDecompStep(mol)
        
        if success:
            states.append(obs)
            actions.append(action)
            mol_list.append(mol)
    if return_mol:
        return mol_list, states, actions
    
    return states,actions
        
        
def DataBaseGenerationONEFILE(smiles):
    i = 0
    j = 0
    atom_types = ['N','C','O','S','F','Cl','Na','P','Br','Si','B','Se','K']
    
    state_list = []
    action_list = []

    for i,smile in enumerate(smiles):
        if i%1000 == 0:
            print(i)
        try:
            mol = Chem.MolFromSmiles(smile, False)
            AdjustAromaticNs(mol)
        except:
            continue
        if all([atom.GetSymbol() in atom_types for atom in mol.GetAtoms()]) and all([bond.GetBondType() != Chem.BondType.TRIPLE for bond in mol.GetBonds()]):
            states,actions = __fullMolDecomp(mol)
            state_list.extend(states)
            action_list.extend(actions)
            
    graph_list, last_action_node_list, last_atom_feat_list = zip(*state_list)
    
    graph_labels = {"last_action" : torch.tensor(last_action_node_list),
                    "last_atom_feats" : torch.cat(last_atom_feat_list,dim=0),
                    "actions" : torch.tensor(action_list)}


    if not os.path.isdir('./graph_decomp'):
        os.mkdir('./graph_decomp')
    dgl.save_graphs('./graph_decomp/graphs',list(graph_list), graph_labels)

