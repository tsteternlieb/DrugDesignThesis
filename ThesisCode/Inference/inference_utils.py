"""
FILE FOR INFERENCE UTILS
"""
from __future__ import annotations
import copy
from dis import dis
import os
import random
import sys
from tkinter import S
from unittest.util import strclass
from matplotlib import pyplot
import networkx as nx
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from collections import deque
import torch
from torch import nn
import yaml
import numpy as np
import pandas as pd
from random import sample

sys.path.append("..")
from torch.distributions import Categorical
from Rewards.rewards import FinalRewardModule

from enviroment.Keku import AdjustAromaticNs


from enviroment.ChemEnv import ChemEnv, DefaultEnv
from Architectures.models import Actor, dropout_inference
from TrainingMain.config_utils import generateActor, generateRewardModule


class MolTree:
    """Class for holding molecular iteration tree"""

    def __init__(self, root_mol: Chem.RWMol, idx: int):
        """init fn

        Args:
            root_mol (Chem.RWMol): the molecule to use as the node
        """
        self.root_mol = root_mol
        self.idx = idx
        self.children = []

    def addChild(self, mol: Chem.RWMol):
        """add a child molecule

        Args:
            mol (Chem.RWMol): [description]
        """
        child = MolTree(mol)
        self.children.append(child)

    def addChildren(self, mols: "list[Chem.RWMol]", i: int):
        """adds children molecules

        Args:
            mols (list[Chem.RWMol]): mols to add
            i (int): starting idx for node numbering

        Returns:
            int: number of children added
        """
        # self.children += list(map(lambda mol: MolTree(mol), mols))
        for j, mol in enumerate(mols):
            self.children.append(MolTree(mol, i + j))

        return len(self.children)

    def ConvertToGraph(self):
        mol = self
        g = nx.graph.Graph()
        queue = deque([mol])

        while queue:
            mol_tree = queue.pop()
            mol = mol_tree.root_mol
            if g.number_of_nodes() == 0:
                g.add_node(
                    mol_tree.idx,
                    mol=Chem.MolToSmiles(mol),
                    photo_path=f"./{mol_tree.idx}.jpg",
                )  # , img=smi2svg(mol), hac=mol.GetNumAtoms())

            for child in mol_tree.children:
                child_mol = child.root_mol
                g.add_node(
                    child.idx,
                    mol=Chem.MolToSmiles(child_mol),
                    photo_path=f"./{child.idx}.jpg",
                )  # , img = smi2svg(mol))
                g.add_edge(mol_tree.idx, child.idx)
                queue.appendleft(child)

        return g

    def traverse(self, func):
        func(self)
        for child in self.children:
            child.traverse(func)

    def __save_image(self):
        print(Chem.MolToSmiles(self.root_mol), "root")
        pil = rdkit.Chem.Draw.MolToImage(self.root_mol)
        pil.save(f"{self.idx}.jpg")

    def __save_tree_images(self):
        self.traverse(MolTree.__save_image)

    def __convert_to_cytoscape(self, graph):
        style = [
            {
                "style": [
                    {
                        "css": {
                            "label": "data(mol)",
                            # 'background-color': 'blue',
                            "shape": "rectangle",
                            "width": 800,
                            "height": 400,
                            "background-image": "data(photo_path)",
                            "background-fit": "contain",
                            "font-size": "60px",
                        },
                        "selector": "node",
                    },
                    {
                        "css": {
                            "width": 20.0,
                        },
                        "selector": "edge",
                    },
                ],
            }
        ]
        # cy_g = cytoscape_data(graph)
        # return Cytoscape(data=cy_g, visual_style=style[0]['style'])

    def DisplayTree(self):
        # graph = self.ConvertToGraph()
        # cyobj = self.__convert_to_cytoscape(graph)

        # return cyobj
        g = GraphFromMolTree(self)
        labels = nx.get_node_attributes(g, "mol")
        nx.draw(g, labels=labels)


class Handler:
    """Class for handling model. inference and that sort of stuff"""

    def __init__(self, path, load_model=True, dropout=False, fine_tuned=False):
        """Class for handling model inference

        Args:
            path (str): location of folder with model state dict as well as config,
        """

        # if type(path) == tuple:
        #     model_path = path[0]
        #     config_path = path[1]

        # else:
        #     config_path = os.path.join(path, "config.yaml")
        #     if fine_tuned:
        #         version = "fine_tuned_model"
        #     else:
        #         version = "supervised_model"

        #     model_path = os.path.join(path, version)

        # with open(config_path) as file:
        #     config = yaml.safe_load(file)

        self.model = generateActor(None, path).cuda()

        # print(model_path)

        # if load_model:
        #     self.model.load_state_dict(torch.load(model_path)["model_state_dict"])
        # self.model.cuda()

        if not dropout:
            self.model.apply(dropout_inference)

        self.env = DefaultEnv()

    def produceMols(self, num):
        l = []
        for i in range(num):
            if i%10 == 0:
                print(f"generated {i} molecules")
            self.env.reset()
            for _ in range(40):
                obs = self.env.getObs()
                distribution = Categorical(self.model(*obs))
                action = distribution.sample()
                if action == 0:
                    break
                self.env.step(int(action))

            AdjustAromaticNs(self.env.StateSpace)
            self.env.removeUnconnected(self.env.StateSpace, False)

            l.append(self.env.StateSpace)

        return l

    def sample(self, mol):
        self.env.assignMol(mol)
        obs = self.env.getObs()
        predictions = self.model(*obs, softmax=True)
        dist = Categorical(probs=predictions)
        return predictions

    def _get_n_best(self, mol: Chem.RWMol, n: int):
        """gets the top n most likely actions given mol

        Args:
            mol (Chem.RWMol): mol to set as state
            n (int): number of actions to return

        Returns:
            Torch.tensor: tensor containing the actions
        """
        # mol = Chem.RWMol(Chem.MolFromSmiles('CC-N'))
        self.env.assignMol(mol)
        obs = self.env.getObs()
        predictions = self.model(*obs)
        _, actions = torch.topk(predictions, n)
        return actions

    def __run_actions(self, mol: Chem.RWMol, actions: "list[int]"):
        """calculates new mols updated by actions

        Args:
            mol (Chem.RWMol): starting structure
            actions (list[int]): actions to take

        Returns:
            list[Chem.RWMol]: newly generated molecules
        """
        new_mols = []
        for action in torch.squeeze(actions):
            action_int = int(action)
            mol_copy = copy.deepcopy(mol)
            self.env.assignMol(mol_copy)
            _, _, _, reward_dict = self.env.step(action_int)

            if reward_dict["step_reward"] > 0:
                new_mols.append(self.env.StateSpace)

        return new_mols

    def iterate(self, mol, n):
        """Expands the passed molecule by one step

        Args:
            mol (Chem.RWMol): base molecule to iterate on
            n (int): How many different one step iterations to make

        Returns:
            list[Chem.RWMol]: The mutated molecules
        """
        actions = self._get_n_best(mol, n)
        mols = self.__run_actions(mol, actions)

        return mols

    def Explore(self, mol, horizon=4):
        mols = []
        for i in range(horizon):
            actions = self._get_n_best(mol, 4).cpu()
            action = 0
            while action == 0:
                action = np.random.choice(actions[0], p=np.array([8, 4, 2, 1]) / 15)

            mols_ = self.__run_actions(mol, torch.tensor([action, action]))[0]

            print(Chem.MolToSmiles(mols_))
            mols.append(mols_)
            mol = mols_

        return mols

    def treeSearch(
        self, initial_mol: Chem.RWMol, width: int, size: int, save_images=False
    ):
        """search chemical space around the initial molecule

        Args:
            initial_mol (Chem.RWMol): starting
            width (int): how many branches to make at each step
            size (int): total size of the tree

        Returns:
            [type]: [description]
        """

        molTree = MolTree(initial_mol, 0)
        queue = deque([molTree])
        i = 1

        while queue:
            if size <= 0:
                break
            mol_node = queue.pop()
            children = self.iterate(mol_node.root_mol, width)
            j = mol_node.addChildren(children, i)
            i = i + j

            for child in mol_node.children:
                print(Chem.MolToSmiles(child.root_mol))

                if save_images:
                    PIL = Draw.MolToImage(child.root_mol)
                    PIL.save(f"./Imagess/{Chem.MolToSmiles(child.root_mol)}.png")
                queue.appendleft(child)

            size -= 1

        return molTree

    def inference():
        pass


def GraphFromMolTree(mol: MolTree):
    """Function for transforming a Molecule Tree to a nx Graph for use with cytoscape

    Args:
        mol (MolTree): Tree to be converted

    Returns:
        nx.graph.Graph: converted graph
    """
    g = nx.graph.Graph()
    queue = deque([mol])

    while queue:
        mol_tree = queue.pop()
        mol = mol_tree.root_mol
        if g.number_of_nodes() == 0:
            g.add_node(
                mol_tree.idx, mol=Chem.MolToSmiles(mol)
            )  # , img=smi2svg(mol), hac=mol.GetNumAtoms())

        for child in mol_tree.children:
            child_mol = child.root_mol
            g.add_node(
                child.idx, mol=Chem.MolToSmiles(child_mol)
            )  # , img = smi2svg(mol))
            g.add_edge(mol_tree.idx, child.idx)
            queue.appendleft(child)

    return g


def testModelReward(
    handler: Handler, path: str, reward_names: list[str], num: int, plot_title
):
    """function for testing performance of model optimized on a reward over random samples from chem database

    Args:
        handler (Handler): mol handler
        path (str): path to drug csv
        reward (list[str]): which rewards were used for optimization
        num (int): how many samples to pull
    """

    drugs = pd.read_csv(path, error_bad_lines=False, delimiter=";")
    drugs.dropna()
    smiles_values = drugs["Smiles"].values
    samples_true = random.sample(list(smiles_values), num)
    samples_true = []

    i = 0
    while i < num:

        smile = random.sample(list(smiles_values), 1)[0]
        if type(smile) == str:
            mol = Chem.MolFromSmiles(smile)
            legal_mols = ["N", "C", "O", "S", "F", "Cl", "Na", "P", "Br", "Se", "K"]

            atom_types = [atom.GetSymbol() for atom in mol.GetAtoms()]
            if all([atom.GetSymbol() in legal_mols for atom in mol.GetAtoms()]):
                samples_true.append(smile)
                i += 1

    samples_gen = handler.produceMols(num)
    reward_module = FinalRewardModule(
        None, generateRewardModule(reward_names, False), False
    )

    true_rewards = []
    gen_rewards = []

    for i in range(num):
        mol = samples_gen[i]
        smiles = Chem.MolToSmiles(mol)
        if "." in smiles:
            s = smiles[: smiles.index(".")]
            mol = Chem.MolFromSmiles(s)
        try:
            reward = reward_module.GiveReward(mol)
        except:
            pass

        gen_rewards.append(float(reward))
        
        if i %100 == 0:
            print(i)

    for i in range(num):
        # print(samples_true[i])
        mol = Chem.MolFromSmiles(samples_true[i])
        reward = reward_module.GiveReward(mol)
        # if reward > -2:
        #     true_rewards.append(float(reward))
        true_rewards.append(float(reward))

    true_rewards.sort()
    gen_rewards.sort()

    # true_rewards = true_rewards[int(.7*num):]
    # gen_rewards = gen_rewards[int(.7*num):]

    print(len(true_rewards), len(gen_rewards))
    bins = 30

    bins = np.histogram(np.hstack((true_rewards, gen_rewards)), bins=60)[1]

    pyplot.hist(
        true_rewards, bins, alpha=0.5, label="Real", color="cadetblue", histtype="bar"
    )
    pyplot.axvline(
        np.mean(true_rewards), color="darkslategrey", linestyle="dashed", linewidth=1
    )

    pyplot.hist(
        gen_rewards,
        bins,
        alpha=0.5,
        label="Synthetic",
        color="olive",
        histtype="bar",
    )
    pyplot.axvline(
        np.mean(gen_rewards), color="darkolivegreen", linestyle="dashed", linewidth=1
    )

    min_ylim, max_ylim = pyplot.ylim()

    pyplot.text(
        np.mean(true_rewards) * 1.1,
        max_ylim * 0.9,
        "Real Mean: {:.2f}".format(np.mean(true_rewards)),
    )
    pyplot.text(
        np.mean(gen_rewards) * .9,
        max_ylim * 0.7,
        "Synthetic Mean: {:.2f}".format(np.mean(gen_rewards)),
    )

    pyplot.legend(loc="lower right")
    pyplot.title(plot_title)

    title = "_".join(reward_names)

    pyplot.savefig("hist_dock_with_RL")


if __name__ == "__main__":
    # handler = Handler(('../TrainingMain/DOCK_RESET/ckpt_5/actor_checkpoint','../TrainingMain/DOCK_RESET/config.yaml'))
    # handler = Handler(('../TrainingMain/QED+SYNTH/ckpt_2/actor_checkpoint','../TrainingMain/QED+SYNTH/config.yaml'))  QED_WEIGHT13_400000
    # handler = Handler(
    #     (
    #         "../TrainingMain/QED+SYNTH/fine_tuned_model",
    #         "../TrainingMain/QED+SYNTH/config.yaml",
    #     )
    # )
    # handler = Handler(('../TrainingMain/QED_WEIGHT13_400000/supervised_model','../TrainingMain/QED_WEIGHT13_400000/config.yaml')) #PRETRAINED
    handler = Handler(('../TrainingMain/DOCK_RESET/ckpt_5/actor_checkpoint','../TrainingMain/DOCK_RESET/config.yaml')) #fine_tuned dock

    testModelReward(
        handler,
        "../GraphDecomp/SmallDrug.csv",
        ['DOCK'],
        800,
        "Real vs Generated Docking with RL ",
    )
