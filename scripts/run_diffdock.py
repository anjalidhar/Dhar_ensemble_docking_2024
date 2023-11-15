#!/usr/bin/env python3

#########################
import numpy as np 
import os, sys 
import argparse
import pandas as pd
import subprocess as sp
from openbabel import openbabel
import mdtraj as md
##########################

import warnings
import functools

def num_str(s, return_str=True, return_num=True):
    s = ''.join(filter(str.isdigit, s))
    if return_str and return_num:
        return s, int(s)
    if return_str:
        return s
    if return_num:
        return int(s)

def sort_strs(strs: list, max=False, indexed: bool=False):
    """ strs ::: a list or numpy array of strings.
        max ::: bool, to sort in terms of greatest index to smallest.
        indexed ::: bool, whether or not to filter out strings that don't contain digits.
                    if set to False and string list (strs) contains strings without a digit, function 
                    will return unsorted string list (strs) as an alternative to throwing an error."""
    
    # we have to ensure that each str in strs contains a number otherwise we get an error
    check = np.vectorize(lambda s : any(map(str.isdigit, s)))
    if isinstance(strs, list):
        strs = np.array(strs)
    # the indexed option allows us to filter out strings that don't contain digits.
    ## This prevents an error
    if indexed:
        strs = strs[check(strs)]
    #if indexed != True, then we don't filter the list of input strings and simply return it
    ##because an attempt to sort on indices (digits) that aren't present results in an error
    else:
        if not all(check(strs)):
            
            warnings.warn("Not all strings contain a number, returning unsorted input list to avoid throwing an error. "
                        "If you want to only consider strings that contain a digit, set indexed to True ")
            return strs
    get_index = np.vectorize(functools.partial(num_str, return_str=False, return_num=True))
    indices = get_index(strs).argsort()
    
    if max:
        return strs[np.flip(indices)]
    
    else:
        return strs[indices]

def lsdir(dir, keyword: "list or str" = None,
          exclude: "list or str" = None,
          indexed:bool=False):
    
    """ full path version of os.listdir with files/directories in order
    
        dir ::: path to a directory (str), required
        keyword ::: filter out strings that DO NOT contain this/these substrings (list or str)=None
        exclude ::: filter out strings that DO contain this/these substrings (list or str)=None
        indexed ::: filter out strings that do not contain digits.
                    Is passed to sort_strs function (bool)=False"""

    if dir[-1] == "/":
        dir = dir[:-1] # slicing out the final '/'

    listed_dir = os.listdir(dir) # list out the directory 

    if keyword is not None:
        listed_dir = keyword_strs(listed_dir, keyword) # return all items with keyword
    
    if exclude is not None:
        listed_dir = keyword_strs(listed_dir, keyword=exclude, exclude=True) # return all items without excluded str/list

    # Sorting (if possible) and ignoring hidden files that begin with "." or "~$"
    return [f"{dir}/{i}" for i in sort_strs(listed_dir, indexed=indexed) if (not i.startswith(".")) and (not i.startswith("~$"))] 

def keyword_strs(strs: list, keyword: "list or str", exclude: bool = False):
    if isinstance(keyword, str): # if the keyword is just a string 
        if exclude:
            filt = lambda string: keyword not in string
        else:
            filt = lambda string: keyword in string
    else:
        if exclude:
            filt = lambda string: all(kw not in string for kw in keyword)
        else:
            filt = lambda string: all(kw in string for kw in keyword)
    return list(filter(filt, strs))


def get_filename(filepath):
    """ returns a string of the file name of a filepath """
    return filepath.split('/')[-1].split('.')[0]


def chk_mkdir(newdir, nreturn=False):
    """ Checks and makes the directory if it doesn't exist. If True, nreturn will return the new file path. """
    isExist = os.path.exists(newdir)
    if not isExist:
        os.mkdir(newdir)
    if nreturn == True:
        return newdir


# PREPARING DIFFDOCK FILES 

def make_pdbs_from_traj(outdir, trajfile, pdb, dock_name): 
    """ Will save a pdb for each frame of a trajectory and then return a list of the pdbs for docking and the protein trajectory. """
    pdbdir = chk_mkdir(f'{outdir}/predock_pdbs', nreturn=True)
    trajpdbs = chk_mkdir(f'{pdbdir}/{dock_name}', nreturn=True)
    traj = md.load(trajfile, top=pdb, stride=1) 
    protein_traj = traj.atom_slice(traj.topology.select("protein"))
    for frame in range(protein_traj.n_frames): 
        protein_traj[frame].save_pdb(f'{trajpdbs}/{dock_name}_{frame}.pdb')
    return lsdir(trajpdbs) , protein_traj 

 
def diffdock_csv(cwd, pdb_list, lig_path, dock_name): # n_frames, indices, cluster):
    """ Makes the appropriate csv file to run diffdock fron a list of pdbs and a ligand. """
    str_list = lambda x, n: [x]*n
    csv_dir = chk_mkdir(f'{cwd}/run_csvs', nreturn=True)
    out_framedir_paths = [f'{dock_name}_out/{dock_name}_{x}' for x in range(len(pdb_list))] # strings for all the frame dirs for each pdb given
    df = pd.DataFrame({'complex_name': out_framedir_paths, 'protein_path': pdb_list, 'ligand_description': str_list(lig_path, len(pdb_list)), 'protein_sequence': str_list('', len(pdb_list))})
    df.to_csv(f'{csv_dir}/{dock_name}.csv', index=False, header=True, sep=',')
    return f'{csv_dir}/{dock_name}.csv', [f'{cwd}/{x}' for x in out_framedir_paths]



# DOCKING

def run_diffdock(outdir, git_repo, trajfile, pdb, ligfile): 
    """ Runs Diffdock for all frames of the given trajectory with the ligand. """
    dock_name = get_filename(trajfile)

    predockpdbs, protein_traj = make_pdbs_from_traj(outdir, trajfile, pdb, dock_name)   
    run_csv, framedirs = diffdock_csv(outdir, predockpdbs, ligfile, dock_name)
    print(f"Docking on {protein_traj.n_frames} now. Hang tight! :) ")

    # Actually running diffdock: 
    sp.run(['python3', f'{git_repo}/inference.py', '--protein_ligand_csv', run_csv, '--out_dir', outdir, '--samples_per_complex', '15', '--model_dir', f'{git_repo}/workdir/paper_score_model', '--confidence_model_dir', f'{git_repo}/workdir/paper_confidence_model']) 

    # Now, parsing out files
    ligout_pdbdir, success_ind = parse_dd_outfiles(outdir, framedirs, dock_name) 

    # Making & saving the docked trajectory for successful frames
    make_traj(outdir, protein_traj, ligout_pdbdir, success_ind, dock_name) 
    print(f"Docking complete! You can find the docked trajectory in {outdir}/trajoutfiles and all docking data in {outdir}/{dock_name}_out. Happy docking!") 


# POST-PROCESSING

def sdf_to_pdb(sdf, out_file_path): 
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("sdf", "pdb")
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, sdf) 
    obConversion.WriteFile(mol, out_file_path)
    return out_file_path

def add_hydrogens(in_file, out_file_path): 
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdb", "pdb")
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, in_file) 
    mol.AddHydrogens()
    mol.AddPolarHydrogens()
    obConversion.WriteFile(mol, out_file_path)


def parse_dd_outfiles(outdir, framedirs, dock_name): 
    """ Will generate indices of failed docking frames and a list of successfully docked ligand pdbs. Returns directory containing ligand pdbs.""" 
# Making dirs for: 
    # lig out pdbs
    alloutpdbs = chk_mkdir(f'{outdir}/ligand_outpdbs', nreturn=True) 
    outpdbs = chk_mkdir(f'{alloutpdbs}/{dock_name}', nreturn=True)
    # cscores 
    csc_dir = chk_mkdir(f'{outdir}/scores', nreturn=True)
    # failed ind
    faileddir = chk_mkdir(f'{outdir}/failed_ind', nreturn=True)

    # Initializing arrays...
    failed = []
    score = []
    success = []

    # Iterating over all frame directories 
    for frame, framedir in enumerate(framedirs): 
        isExist = os.path.exists(f'{framedir}/rank1.sdf')
        if isExist: # only if docking did not fail
            ligpdb = sdf_to_pdb(f'{framedir}/rank1.sdf', f'{outpdbs}/ligout_{frame}.pdb')
            add_hydrogens(ligpdb, ligpdb) 
            score.append(get_score(framedir))   
            success.append(frame)
        else: # if failed, 
            failed.append(int(frame))

    np.savetxt(f'{faileddir}/{dock_name}_failed.txt', np.array(failed))
    np.save(f'{csc_dir}/{dock_name}_scores.npy', np.array([score, success]))

    return outpdbs, np.array(success)

def make_traj(outdir, protein_traj, ligpdbs, success_ind, dock_name): 
    """ Makes a trajectory given a list of ligand pdbs, indices of the failed frames, and the original trajectory. """
    # Slicing the original protein traj to match 
    sliced_protein_traj = protein_traj[success_ind]
     
    # loading a ligand trajectory from the pdbs and stacking with the sliced protein trajectory 
    ligtraj = md.load(lsdir(ligpdbs))
    docked_traj = sliced_protein_traj.stack(ligtraj)
 
    # Saving...
    outtrajs = chk_mkdir(f'{outdir}/trajoutfiles', nreturn=True) 
    docked_traj.save_xtc(f'{outtrajs}/{dock_name}_out.xtc')
    docked_traj[0].save_pdb(f'{outtrajs}/{dock_name}_out.pdb')  

def get_score(ddoutdir): 
    """ Will return the confidence score for the rank 1 position for a given diffdock out directory. """ 
    rank1file = lsdir(ddoutdir, keyword="rank1_")
    split1 = rank1file[0].split('e')
    split2 = split1[-1].split('.')
    return float(f'{split2[0]}.{split2[1]}')
   


if __name__ == '__main__': 

#################
    parser = argparse.ArgumentParser(description = "Runs DiffDock on all frames of a given trajectory (will dock on only the protein, and slice out all other atom types). Docks with a provided ligand mol2 file. Will save docked trajectories and all dock data to a specified out directory.") 

    parser.add_argument("--outdir", required=True, type=str, help='The directory where all the out directories and files will be written :) ') 

    parser.add_argument('--traj', required=True, type=str, help='The path to the traj file you want to use. Will dock for all frames of the trajectory provided.') 
    parser.add_argument('--pdb', required=True, type=str, help='The pdb topology file to properly load the provided trajectory file.') 
    parser.add_argument('--lig_file', required=True, type=str, help='The path to the ligand mol2 file you wish to dock with.') 

    parser.add_argument('--git_repo', required=False, default= '/dartfs-hpc/rc/home/x/f004r0x/Anjali/diffdock/DiffDock', help='absolute path to the github repo of DiffDock :) it has all the training data we will need!')

    args = parser.parse_args()
################

    run_diffdock(args.outdir, args.git_repo, args.traj, args.pdb, args.lig_file)
    
