# This file is part of SpikeyTree.
# Copyright (C) 2015 Paula Tataru

# SpikeyTree is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import math

import numpy as np
import scipy.special as special


def get_bins(K, surr=True):
    '''construct the bins of length K+1 and corresponding mid points'''
    step = 1.0/K
    mid = None
    bins = np.concatenate(([0], np.arange(step/2, 1, step), [1]))
    if surr:
        # I am running beta with spikes, I need the spikes
        mid = np.concatenate(([0], [step/4],
                              np.arange(step, 1, step),
                              [1-step/4], [1]))
    else:
        mid = np.arange(0, 1+step, step)
    return mid, bins


def pos(m, bins):
    '''find position of number in bins,
    by treating the boundaries in a special way'''
    if m <= 0:
        return 0
    if m >= 1:
        return -1
    return np.digitize([m], bins)


def read_data(filename):
    '''read data from file'''
    f = open(filename)
    # read tree
    tree = f.readline()
    # read sample sizes
    aux = f.readline()
    n = np.array(map(float, aux.split()))
    # read labels
    labels = f.readline().split()
    
    row_data = np.empty(len(labels))
    data_matrix = np.empty([0, len(labels)])
    freq_matrix = np.empty([0, len(labels)])
    # read the entries in the data matrix
    for line in f:
        row_data = map(int, line.split())
        data_matrix = np.vstack((data_matrix, row_data))
        freq_matrix = np.vstack((freq_matrix, row_data/n))
    f.close()
    
    # identify unique rows in the data
    bins = np.ascontiguousarray(data_matrix)
    bins = bins.view(np.dtype((np.void, 
                         data_matrix.dtype.itemsize * data_matrix.shape[1])))
    _, idx, count = np.unique(bins, return_index=True, return_counts=True)
    
    data = {}
    sample = {}
    for i,l in enumerate(labels):
        data[l] = []
        sample[l] = n[i]
    # store unique data
    for j in idx:
        for i, l in enumerate(labels):
            data[l].append(data_matrix[j, i]) 
    
    # add fake fixed and lost sites
    for l in labels:
        data[l].append(0)
        data[l].append(sample[l])
    
    return tree, [data, count, sample, [np.mean(freq_matrix), np.var(freq_matrix)]]


def read_tree(filename):
    '''read tree from file'''
    f = open(filename)
    # read tree
    tree = f.readline()
    f.close()
    return tree


def to_inf(param): 
    '''transform positive params to [-inf, +inf]'''
    return np.log(param)

    
def to_pos(param): 
    '''transform params in [-inf, +inf] to positive'''    
    return np.exp(param)


def branch_str(bins, l, scaled=True):
    '''print scaled or unscaled information for tree'''
    tree_str = ''
    if scaled:
        tree_str += (': %.5f' % (float(l) / (2*bins)))
    else:
        tree_str += (' %d : %d' % (bins, l))
    return tree_str


def fst_dist(pop1, n1, pop2, n2):
    '''calculate distance based on Fst'''
    num = 0
    den = 0
    for x, y in zip(pop1, pop2):
        p1 = x / float(n1)
        p2 = y / float(n2)
        # average p
        p = (x + y) / float(n1 + n2)
        num += (p1 - p2)**2
        den += 2 * (p - p1*p2)
    return -math.log(1 - num/den)


def write_tree(output, tree, params=True):
    '''write tree and parameters to file'''
    if type(output) is str:
        f = open(output, 'w')
    else:
        f = output
    tree_str = tree.to_str()
    f.write(tree_str)
    f.write(';\n')
    f.write("Shape parameters at root: %.4e %.4e\n" 
            % (tree.shapes[0], tree.shapes[1]))
    if params:
        params_bran = tree.get_params()
        f.write('Branch lengths in DF order\n')
        for bran in params_bran:
            f.write('%.6e\t' % (bran))
        f.write('\n')
    if type(output) is str:
        f.close()


def read_correct(corr):
    '''read correct parameters on tree from file'''
    f = open(corr)
    param = None
    scaled_bran = None
    count = 0
    for line in f:
        if count == 1:
            # shape parameters
            param = map(float, line.split(":")[1].split())
        if count == 3:
            # branch lengths
            scaled_bran = map(float, line.strip().split())
        count += 1
    return param, scaled_bran
