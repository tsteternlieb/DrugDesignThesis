import subprocess
from time import time
from vina import Vina


class VinaWrapper():
    def __init__(self, receptor_path):
        self.receptor_path = receptor_path
        self.v = Vina(sf_name='vina',verbosity=0)
        
    def _generatePDBQT(self,SMILE):
        print(SMILE)
        SMILE = SMILE.replace('B','C')
        SMILE = SMILE.replace('Cr','Br')
        print(SMILE)        
                
        try:
            subprocess.run(['obabel',f'-:{SMILE}',"--gen3d",
                          '-o', "pdbqt", '-O', './mol.pdbqt'])
        except:
            pass
    def _dock(self, path='./mol.pdbqt'):
        try:
            print('hi')
            self.v.set_ligand_from_file(path)
            
            self.v.compute_vina_maps(center=[54.797, 71.04, 45.391], box_size=[20, 20, 20])
            energy = self.v.score()
            energy_minimized = self.v.optimize()
            self.v.dock(exhaustiveness=1, n_poses=1)
            return (self.v.energies())
        except:
            return [[0]]
        
        
    def CalculateEnergies(self,SMILE):
        #weighted avergage based on energy
        try:
            split_SMILE = SMILE.split('.')[0]
            self.v.set_receptor(self.receptor_path)
            self._generatePDBQT(split_SMILE)
            
            r = sum(self._dock()[:,0])
            return r
        except:
            return 0
        
        