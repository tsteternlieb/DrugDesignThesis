'''
FILE FOR GENERATING SUPERVISED TRAINING DATA
USAGE: python.py decompy --path [path to csv with smiles column]
'''


import argparse
import pandas as pd
from decomp_utils import DataBaseGenerationONEFILE

def main():
    """call with the path of a file containing csv with smiles columns to generate the dataset
    """
    print('123123')
    parser = argparse.ArgumentParser(description="graph_decomp")
    parser.add_argument("--path",type=str,required=True)
    args = parser.parse_args()
    drugs = pd.read_csv(args.path,error_bad_lines=False,delimiter=';')
    smiles_values = drugs['Smiles'].values
    DataBaseGenerationONEFILE(smiles_values)
    
main()