import sys
from torch.utils.tensorboard import SummaryWriter

sys.path.append("..")
import copy
import torch, random, dgl
import networkx as nx
from rdkit import Chem
import dgl.data

from .utils import (
    fixOrder,
    permute_mol,
    permute_rot,
    SanitizeNoKEKU,
    mol_to_graph_full,
    MolFromGraphsFULL,
    permuteAtomToEnd,
)

from .Keku import AdjustAromaticNs

from Rewards.rewards import FinalRewardModule
import os

import wandb

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ChemEnv(object):
    """
    Class which holds logic for graph generation environment
    Observations come in the form of (current_graph, last action was node addition, last node features)


    """

    def __init__(
        self,
        num_node_feats: int,
        RewardModule: FinalRewardModule,
        mol_featurizer,
        writer,
        wandb_log = True
    ):

        """
        ENV_Atoms
        """
        self.mol_featurizer = mol_featurizer
        self.atom_list = [
            "N",
            "C",
            "O",
            "S",
            "F",
            "Cl",
            "Na",
            "P",
            "Br",
            "Si",
            "B",
            "Se",
            "K",
            "Benz",
            "Pyri",
            "Pyrr",
        ]
        self.atom_bond_dict = {
            "N": [1, 0, 5],
            "C": [2, 0, 4],
            "O": [3, 0, 6],
            "S": [4, 0, 6],
            "F": [5, 0, 7],
            "Cl": [6, 0, 7],
            "Na": [7, 0, 7],
            "P": [8, 0, 5],
            "Br": [9, 0, 7],
            "Si": [10, 0, 4],
            "B": [11, 0, 5],
            "Se": [12, 0, 6],
            "K": [13, 0, 7],
        }

        self.wandb_log = wandb_log

        """
        ENV_Attributes
        """
        self.num_atom_types = self.atom_list.__len__()
        self.batch_dim = 1

        self.StateSpace = Chem.RWMol()

        """ENV_State"""
        self.Done = False
        self.num_node_feats = num_node_feats
        self.reward = 0

        self.log = ""

        self.completed_mols = []

        self.episode_step_reward = 0
        self.num_episodes = 0
        self.episode_length = 0
        
        self.idx = 0 

        """
        External_Rewards
        """
        self.RewardModule = RewardModule
        self.model_list = []

        self.writer = writer

    def __len__(self):
        return self.StateSpace.GetNumAtoms()

    @property
    def n_nodes(self):
        return self.StateSpace.GetNumAtoms()

    def Log(self):
        self.RewardModule.LogRewards()

    def clear(self):
        self.StateSpace = Chem.RWMol()

    def addStructure(self, mol2):
        mol1 = self.StateSpace
        add_dif = mol1.GetNumAtoms()
        for atom in mol2.GetAtoms():
            new_atom = Chem.Atom(atom.GetSymbol())
            mol1.AddAtom(new_atom)
        for bond in mol2.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            bt = bond.GetBondType()
            mol1.AddBond(add_dif + a1, add_dif + a2, bt)
            mol1.UpdatePropertyCache()

    def addBenzine(self):
        mol = Chem.MolFromSmiles("c1ccccc1")
        self.addStructure(mol)
    def addPyridine(self):
        mol = Chem.MolFromSmiles("N1=CC=CC=C1")
        mol = permute_mol(mol, permute_rot(mol.GetNumAtoms()))
        SanitizeNoKEKU(mol)
        self.addStructure(mol)

    def addPyrrole(self):
        mol = Chem.MolFromSmiles("N1C=CC=C1")
        mol = permuteAtomToEnd(mol, 0)
        self.addStructure(mol)

    def addNaptholene(self):
        mol = Chem.MolFromSmiles("C1=CC=C2C=CC=CC2=C1")
        self.addStructure(mol)

    def assignMol(self, mol):
        mol = Chem.RWMol(mol)
        mol = fixOrder(mol)
        self.StateSpace = mol
        self.getObs()

    def resetStateSpace(self):
        while True:
            graph_id = random.randint(1, 500000)  ###so lazy erggg
            graph, graph_dict = dgl.load_graphs(
                "../GraphDecomp/graph_decomp/graphs", [graph_id]
            )
            self.idx = graph_id

            try:
                mol = MolFromGraphsFULL(graph[0])
                mol = fixOrder(mol)
                Chem.SanitizeMol(mol)
                break
            except:
                # print("err")
                pass

        graph = graph[0]

        mol = MolFromGraphsFULL(graph)
        # mol = Chem.RWMol(Chem.MolFromSmiles('CCO'))
        SanitizeNoKEKU(mol)
        
        # if nx.is_connected(
        #         mol_to_graph_full(mol).to_networkx().to_undirected()
        #     ):
        
        
        self.assignMol(Chem.RWMol(mol))

    def reset(self):
        self.resetStateSpace()

        self.reward = 0
        self.log = ""

        self.episode_step_reward = 0
        self.episode_length = 0

        self.Done = False

        obs = self.getObs()
        return obs  
        
    def addNode(self, node_choice, give_reward=True):
        #####figure out last features
        if not nx.is_connected(
                mol_to_graph_full(self.StateSpace).to_networkx().to_undirected()
            ):
            if give_reward:
                self.reward -= 0.1
            return
        
        if give_reward:
            self.reward += 0.1
        if node_choice == "Benz":
            self.addBenzine()
        elif node_choice == "Pyri":
            self.addPyridine()
        elif node_choice == "Pyrr":
            self.addPyrrole()
        else:
            self.StateSpace.AddAtom(Chem.Atom(node_choice))

    def valiateMol(self, mol):
        """method for validating molecules

        Args:
            mol (Chem.RDMol): Chem molecule

        Returns:
            bool: whether the molecule is good or not
        """

        # check connected
        try:
            if not nx.is_connected(
                mol_to_graph_full(mol).to_networkx().to_undirected()
            ):
                return False
        except:
            return False

        # check kekulization
        try:
            Chem.SanitizeMol(mol)
        except Chem.rdchem.KekulizeException:
            return False

        return True

    def addEdge(self, edge_type, atom_id, give_reward=True):
        """
        Method for calculating new graph after adding an edge between the last node added and nodes[atom_id]
        returns nothing as we mutate in place
        """

        try:
            atom_id = (atom_id).item()
        except:
            pass

        if edge_type == 1:
            bond = Chem.rdchem.BondType.SINGLE
        elif edge_type == 2:
            bond = Chem.rdchem.BondType.DOUBLE

        mol_copy = copy.deepcopy(self.StateSpace)
        mol_copy.UpdatePropertyCache()
        SanitizeNoKEKU(mol_copy)

        addable = True

        # perform checks

        # add bond to complete the rest of the checks
        try:
            mol_copy.AddBond(atom_id, self.StateSpace.GetNumAtoms() - 1, bond)
            mol_copy.UpdatePropertyCache()
            SanitizeNoKEKU(mol_copy)
        except:
            addable = False

        validated = self.valiateMol(mol_copy)

        if validated and addable:
            self.StateSpace.AddBond(atom_id, self.StateSpace.GetNumAtoms() - 1, bond)
            self.StateSpace.UpdatePropertyCache()
            try:
                Chem.SanitizeMol(self.StateSpace)
            except:
                self.StateSpace = AdjustAromaticNs(self.StateSpace)
                Chem.SanitizeMol(self.StateSpace)
            self.reward += 0.1

            self.log += "edge added \n"
        else:
            self.reward -= 0.1

    def removeUnconnected(self, mol, sanitize=True):
        # need to remove rings as well
        if mol.GetAtomWithIdx(mol.GetNumAtoms() - 1).GetDegree() == 0:
            mol.RemoveAtom(mol.GetNumAtoms() - 1)

        elif mol.GetNumAtoms() > 6:
            num_atoms = mol.GetNumAtoms()
            if all([atom.GetDegree() == 2 for atom in list(mol.GetAtoms())[-5:]]):
                for i in range(5):
                    mol.RemoveAtom(num_atoms - (1 + i))
            elif all([atom.GetDegree() == 2 for atom in list(mol.GetAtoms())[-6:]]):
                for i in range(6):
                    mol.RemoveAtom(num_atoms - (1 + i))

        AdjustAromaticNs(mol)
        if sanitize:
            Chem.SanitizeMol(mol)

        self.StateSpace = mol

    def checkValence(self, atom_id, edge_type):
        atom = self.StateSpace.GetAtomWithIdx(atom_id)
        currValence = atom.GetExplicitValence()
        maxValence = 8 - self.atom_bond_dict[atom.GetSymbol()][-1]
        return currValence + edge_type > maxValence

    def modelRewards(self, mol):
        return self.RewardModule.GiveReward(mol)

    def graphObs(self):
        self.StateSpace.UpdatePropertyCache()
        return dgl.add_self_loop(
            dgl.remove_self_loop(self.mol_featurizer(self.StateSpace))
        ).to(DEVICE)

    def getObs(self):

        graph = self.graphObs()
        last_atom_feats = torch.unsqueeze(graph.ndata["atomic"][-1], dim=0)

        if nx.is_connected(graph.cpu().to_networkx().to_undirected()):
            last_action_node = torch.zeros((1, 1)).to(DEVICE)
        else:
            last_action_node = torch.ones((1, 1)).to(DEVICE)

        return graph, last_action_node, last_atom_feats

    def step(self, action, final_step=False, verbose=False):
        """
        Function for a single step in our trajectory
        Expect action to be an int indexing
        [terminate, add_atom1,...,add_atomN, node1_edge, ... ,nodeN_edge]
        """
        self.TempSmiles = Chem.MolToSmiles(self.StateSpace)

        self.episode_length += 1
        reward_dict_info = {
            "model_reward": 0,
            "property_reward": 0,
            "step_reward": 0,
        }  # info for different rewards for logging

        self.reward = 0
        self.log = ""
        terminated = False

        # case for termination
        if action == 0:
            self.log += "terminating \n"
            self.Done = True
            terminated = True
            """final rewards """

        # case for adding a node
        elif action > 0 and action < self.num_atom_types + 1:
            self.log += "------adding " + self.atom_list[action - 1] + " atom------ \n"
            self.addNode(self.atom_list[action - 1])
            SanitizeNoKEKU(self.StateSpace)

        # case for edge addition
        elif action < 1 + self.num_atom_types + (2 * self.__len__()):

            destination_atom_idx = (action - len(self.atom_list) - 1) // 2
            edge_type = (action - self.num_atom_types - 1) % 2 + 1

            self.log += (
                "------attempting to add "
                + str(edge_type)
                + " bond between last atom added and atom "
                + str(destination_atom_idx)
                + "------ \n"
            )
            self.addEdge(edge_type, destination_atom_idx)
        else:
            self.log += "------action id is too large for state space------ \n"

        reward_dict_info["step_reward"] = self.reward

        # self.episode_step_reward += self.reward
        if self.wandb_log:
            wandb.log({"step reward": self.reward})

        if final_step:
            terminated = True

        if terminated:
            # print(Chem.MolToSmiles(self.StateSpace))
            self.removeUnconnected(self.StateSpace, sanitize=False)
            AdjustAromaticNs(self.StateSpace)
            self.reward += self.modelRewards(self.StateSpace)
        if verbose:
            print(self.log)

        obs = self.getObs()
        return obs, self.reward, self.Done, reward_dict_info


class DefaultEnv(ChemEnv):
    def __init__(self, cpu = True):
        # if cpu:
        #     DEVICE = torch.device("cpu")
        wr = SummaryWriter
        rewardModule = FinalRewardModule(wr,[],False)
        super().__init__(54, rewardModule, mol_to_graph_full,wr,False)
