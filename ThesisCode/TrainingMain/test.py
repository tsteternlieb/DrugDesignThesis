import sys
sys.path.append('..')
from Rewards.autoDock import VinaWrapper


from vina import Vina


receptor_path = '../Rewards/y220c_av.pdbqt'
va = VinaWrapper(receptor_path)
va._generatePDBQT('Oc1ccc(-n2cccc2)cc1Nc1cc(F)c([Na])c(Br)c1')
v = Vina(sf_name='vina')
v.set_receptor(receptor_path)
v.set_ligand_from_file('./mol.pdbqt')

v.compute_vina_maps(center=[54.797*.375, 71.04*.375, 45.391*.375], box_size=[108*.375, 78*.375, 110*.375])
energy = v.score()
energy_minimized = v.optimize()
v.dock(exhaustiveness=32, n_poses=3)
print (v.energies())
# docker = VinaWrapper('../Rewards/y220c_av.pdbqt')
# docker.CalculateEnergies('CS(=O)(=O)C1=NC=C(C(=N1)C(=O)O)Cl')