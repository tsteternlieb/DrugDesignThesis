import subprocess
from vina import Vina


class VinaWrapper():
    def __init__(self, receptor_path):
        self.receptor_path = receptor_path
        
        
    def _generatePDBQT(self,SMILE):
        sp = subprocess.Popen(['obabel','-:C-C',"--gen3d",
                          '-o', "pdbqt", '-O', 'PDBQT/Ligands/mol.pdbqt'])
        sp.wait()
        
    def _dock(self, path='PDBQT/Ligands/mol.pdbqt'):
        self.v.set_ligand_from_file(path)
        self.v.compute_vina_maps(center=[54.797*.375, 71.04*.375, 45.391*.375], box_size=[108*.375, 78*.375, 110*.375])
        energy = self.v.score()
        energy_minimized = self.v.optimize()
        self.v.dock(exhaustiveness=32, n_poses=3)
        return (self.v.energies())
        
    def CalculateEnergies(self,SMILE):
        #weighted avergage based on energy
        self.v = Vina(sf_name='vina')
        self.v.set_receptor('./PDBQT/y220c_av.pdbqt')
        self._generatePDBQT(SMILE)
        
        return sum(self._dock()[:,0])/3
        