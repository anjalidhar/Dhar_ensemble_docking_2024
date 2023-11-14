#!/usr/bin/env python3
# coding: utf-8

import mdtraj as md
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os, sys
import matplotlib.colors as colors
# from IPython.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))


# Directory // file comprehension 
# Excludes all hidden files 

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


def source_module(module_file: str, local_module_name: str = None):

    """to add a module from a user defined python script into the local name space"""

    #
    if local_module_name is None:
        local_module_name = module_file.split("/")[-1].replace(".py", "")

    if len(module_file.split("/")) == 1 or module_file.split("/")[-2] == ".":
        module_dir = os.getcwd()
    else:
        module_dir = "/".join(module_file.split("/")[:-1])

    sys.path.insert(0, module_dir)

    module = importlib.import_module(module_file.split("/")[-1].replace(".py", ""))

    g = globals()
    g[local_module_name] = module


    pass


# For MD trajectory analysis

def vec_angles(trj,atom_indices=[]):
    """    Calculates the angle between 3 atoms for each frame of a given trajectory. Returns 
        a np array of shape (n_frames,) 
        
    """
    xyz=[]
    for  atom_idx in atom_indices :
        a=[]
        for frame_idx in range(trj.n_frames):
            a.append(trj.xyz[frame_idx, atom_idx,:].astype(float))
        xyz.append(a)    
    xyz=np.array(xyz)
    xyz.shape
    
    #Define vectors with 2nd atom as starting point
    V=[]
    v1=xyz[0]-xyz[1]
    v2=xyz[2]-xyz[1]
    V.append(v1)
    V.append(v2)

    #Compute angles between two vectors
    angles=[]
    for i in range(trj.n_frames):
        a=np.rad2deg(np.arccos(np.dot(V[0][i],V[1][i])/
                               (np.sqrt(np.dot(V[0][i],V[0][i])*np.dot(V[1][i],V[1][i])))))
        angles.append(a)
    angles=np.array(angles)
    return angles


def distance_matrix(sel1,sel2,traj,mat_type,measure,periodic):
    ''' RETURNS: dmat,np.array(pairs),np.array(pairs_index),index,columns,len(sel1)..(x),len(sel2)....(y)
    '''
    offset1 = sel1[0];offset2 = sel2[0]
    if mat_type == "inter":
        pair_distances = []
        pairs = []
        pairs_index = []
        if measure == "residues":
            index = [traj.topology.residue(i) for i in sel1]
            columns = [traj.topology.residue(j) for j in sel2]
            for i in sel1:
                for j in sel2:
                    pairs.append("{},{}".format(traj.topology.residue(i),traj.topology.residue(j)))
                    pairs_index.append([i,j])
                    if i==j:
                        dist = np.zeros(traj.n_frames)
                        pair_distances.append(dist)
                    else:
                        dist = md.compute_contacts(traj,[[i,j]],periodic=periodic)[0][:,0]
                        pair_distances.append(dist)
        if measure == "atoms":
            index = [traj.topology.atom(i) for i in sel1]
            columns = [traj.topology.atom(j) for j in sel2]
            for i in sel1:
                for j in sel2:
                    pairs.append("{},{}".format(traj.topology.atom(i),traj.topology.atom(j)))
                    pairs_index.append([i,j])
                    if i==j:
                        dist = np.zeros(traj.n_frames)
                        pair_distances.append(dist)
                    else:
                        dist = md.compute_distances(traj,[[i,j]],periodic=periodic)[:,0]
                        pair_distances.append(dist)
        dist_feat_arr = np.stack(pair_distances,axis=1)
        dmat = dist_feat_arr.reshape((traj.n_frames,len(sel1),len(sel2)))
    if mat_type == "intra":
        pair_distances = []
        pairs = []
        pairs_index = []
        dmat = np.zeros((traj.n_frames,len(sel1),len(sel2)))
        if measure == "residues":
            index = [traj.topology.residue(i) for i in sel1]
            columns = [traj.topology.residue(j) for j in sel2]
            for i in sel1:
                for j in sel2:
                    if i<j:
                        pairs.append("{},{}".format(traj.topology.residue(i),traj.topology.residue(j)))
                        pairs_index.append([i,j])
                        dist = md.compute_contacts(traj,[[i,j]],periodic=periodic)[0][:,0]
                        pair_distances.append(dist)
                        dmat[:,i-offset1,j-offset2] = dist
                        dmat[:,j-offset2,i-offset1] = dist
        if measure == "atoms":
            index = [traj.topology.atom(i) for i in sel1]
            columns = [traj.topology.atom(j) for j in sel2]
            for i in sel1:
                for j in sel2:
                    if i<j:
                        pairs.append("{},{}".format(traj.topology.atom(i),traj.topology.atom(j)))
                        pairs_index.append([i+offset1,j+offset2])
                        dist = md.compute_distances(traj,[[i,j]],periodic=periodic)[:,0]
                        pair_distances.append(dist)
                        dmat[:,i-offset1,j-offset2] = dist
                        dmat[:,j-offset2,i-offset1] = dist
        dist_feat_arr = np.stack(pair_distances,axis=1)
    return dmat,dist_feat_arr,np.array(pairs),np.array(pairs_index),index,columns,np.array([len(sel1),len(sel2)])


def get_residues(traj): 
    """ Returns mdtraj residue objects for residues in an mdtraj object :) """
    return [res for res in traj.topology.residues]


def reindex_traj_obj(traj):
    """ Given a traj object, returns a sorted traj object where the atoms are in order of chainID and resSeq """
    table, bonds = traj.topology.to_dataframe()
    # sort by chain and residue index (sequence)
    table.sort_values(["chainID","resSeq"], ignore_index=True, inplace=True)
    # get the resorted index 
    reindex = table["serial"].to_numpy()
    # replace the old index to be ascending i.e 1,2,3....N
    table["serial"] = np.arange(1,len(table)+1)
    # make "fixed" MDtraj topology object
    top = md.Topology.from_dataframe(table,bonds)
    #make new trajectory
    traj = md.Trajectory(xyz = traj.xyz[:,reindex-1], topology=top)
    
    return traj
    
    return None


# For plotting // data maintenance 

import pickle
def load_dict(file):
    """
    Loads a dictionary from a file using pickle. 
    """
    with open(file, "rb") as handle:
        dic_loaded = pickle.load(handle)
    return dic_loaded
def save_dict(file, dict):
    """
    Saves a dictionary to a file using pickle. 
    """
    with open(file, "wb") as handle:
        pickle.dump(dict, handle)
    return None


def get_color_list(n_colors: int, cmap: str, trunc=0, pre_trunc=0):
    """
    Generates a list of colours using a matplotlib colour map. Has optional truncation parameters. 
    """
    cmap = getattr(plt.cm, cmap)
    cl = [cmap(i) for i in range(cmap.N)]
    return [cl[i] for i in np.linspace(1 + pre_trunc, len(cl) - 1 - trunc, n_colors).astype(int)]
color_list = get_color_list(6, "spring", trunc = 30, pre_trunc=30)


def PolyRegression1D(x, y, degree = 1, intercept = True):
    """one dimensional (y is a single variable function of x) fitting using an n-degree polynomial

    RETURNS: a dictionary
    KEY  , VALUE

    ypred , estimated y values based on regression result (np.ndarray)
    co-effs , co-efficients for each degree of the polynomial in increasing order (np.ndarray)
    r2 , the r-squared value from the fit (float)
    y_residuals , difference between actual y data and the predicted y data

    ADDITIONAL RETURNS FOR DEGREE == 1 (fitting a line)

    slope_error , uncertainty in the predicted slope
    int_error , uncertainty in the predicted y-intercept

    """
    X = []
    if intercept:
        start = 0
    else:
        start = 1
    for i in range(start,degree+1):
        X.append(x**i)
    X = np.stack(X, axis = 1) #make matrix to solve co-efficients of
    b = np.linalg.inv(X.T@X)@X.T@y #solve for co-efficients
    yhat = (X@b.reshape(-1,1)).flatten() #predict y-values based on co-efficients
    yres = y-yhat #get y residuals
    yres2 = yres**2 #get squared residuals
    yvar = (y-y.mean())**2 #get un-normalized variance of y
    r2 = 1 - (yres2.sum())/(yvar.sum()) #compute r-squared
    if degree == 1:
        """compute error of slope and intercept if applicable"""
        x2 = x**2
        slope_error = np.sqrt(np.sum(yres2)/((len(x)-2)*np.sum((x-x.mean())**2)))
        int_error = np.sqrt((np.sum(yres2)*x2.sum())/((len(x)-2)*len(x)*np.sum((x-x.mean())**2)))

        dic = dict(zip("ypred,co-effs,r2,y_residuals,slope_error,int_error".split(","),
                        [yhat,b,r2,yres,slope_error,int_error]))
        fitter = fit_new_data(b)
        return dic,fitter

    else:
        return dict(zip("ypred,co-effs,r2,y_residuals".split(","),[yhat, b, r2, yres]))


class fit_new_data:

    def __init__(self,coeffs):
        self.coeffs = coeffs

    def __call__(self,data):
        fit_data = np.stack([np.ones(len(data)),data],axis = 1)
        return fit_data@self.coeffs.reshape(-1,1)






