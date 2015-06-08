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

import os

from numpy.core.multiarray import dot
from optparse import OptionParser

import math
import newick.tree
import numpy as np
import numpy.random as nprand
import scipy.stats as stats

from tree import FullTree

import utils


def sim(N, p0, r):
    '''simulate from Wright Fisher with pure drift'''
    p = p0
    for _ in range(r):
        p = nprand.binomial(N, p, 1)
        p = p[0] / float(N)
    return p


def sim_tree(tree, p0, root=True):
    '''simulate from Wright Fisher on a tree'''
    # if leaf, return
    if type(tree) is newick.tree.Leaf:
        return [p0]
    res = []
    for (sub_tree, N, l) in tree._edges:
        p = sim(2 * N[0], p0, l)
        res.extend(sim_tree(sub_tree, p, False))
    if root:
        # make the return to be a dictionary
        leaves = tree.get_leaves_identifiers()
        sim_dict = {}
        for leaf, p in zip(leaves, res):
            sim_dict[leaf] = p
        return sim_dict
    return res


def sim_sample(res, leaves, samples):
    '''simulate samples from given population frequency'''
    x = []
    for l in leaves:
        p = res[l]
        count = nprand.binomial(samples[l], p, 1)[0]
        x.append('%d' % count)
    return '\t'.join(x)


def write_data(output, tree, loci, samples):
    '''simulate and write data to file'''
    f = open(output, 'w')
    tree_str = tree.to_str(True, False)
    f.write(tree_str)
    f.write(';\n')
    
    leaves = list(tree.tree.get_leaves_identifiers())
    full_samples = {}
    for i,l in enumerate(leaves):
        full_samples[l] = samples[i % len(samples)]
        f.write('%d ' % full_samples[l])
    f.write('\n')
    f.write('\t'.join(leaves))
    f.write('\n')
    
    # all 0 or all 1 in data
    lost = []
    fixed = []
    for l in leaves:
        lost.append('0')
        fixed.append('%d' % full_samples[l])
    lost = '\t'.join(lost)
    fixed = '\t'.join(fixed)
    
    # simulate polymorphic loci
    i = 0
    while i < loci:
        p0 = float('nan')
        while math.isnan(p0):
            p0 = nprand.beta(tree.shapes[0], tree.shapes[1])
        res = sim_tree(tree.tree, p0)
        counts = sim_sample(res, leaves, full_samples)
        if counts != lost and counts != fixed:
            f.write(counts)
            f.write('\n')
            f.flush()
            i += 1
    f.close()


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option('-f', dest='file',
                      help='file with tree to simulate on')
    parser.add_option('-s', dest='shapes', default='1,1',
                      help='shape parameters for root distribution [default: %default]')
    parser.add_option('-n', dest='sample', default='100',
                      help='sample sizes for populations [default: %default]')
    parser.add_option('-l', dest='loci', type='int', default=100,
                      help='number of loci/sites to simulate [default: %default]')
    parser.add_option('-o', dest='output',
                      help='file to write simulation to')

    (opt, args) = parser.parse_args()

    init_tree = utils.read_tree(opt.file)
    # initialize tree structure
    tree = FullTree(init_tree)
    tree.update_pop_sizes()

    output = os.path.basename(opt.output)
    path = os.path.dirname(opt.output)
    
    tree.shapes = map(float, opt.shapes.split(','))
    
    # extract sample sizes
    samples = map(int, opt.sample.split(','))
    
    write_data(opt.output, tree, opt.loci, samples)
    utils.write_tree('%s/tree_%s' % (path, output), tree)
