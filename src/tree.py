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

from sets import Set

import math

from numpy.core.multiarray import dot

import newick.tree
import numpy as np
import scipy.stats as stats

import beta
import utils


def trans_prob(bins, N, all_p0, r, spikes):
    '''calculate transition probabilities
    using either the beta or the beta with spikes'''
    n = len(all_p0)
    trans = np.empty((n, n))
    for i, p0 in enumerate(all_p0):
        trans[i, :] = beta.distr(bins, N, p0, r, spikes)
    return trans


class VisitorLength(newick.tree.TreeVisitor):
    '''Extract all the edge lengths in a tree'''

    def post_visit_edge(self, _, __, length, ___):
        if length is None:
            length = 0
        self.edges.append(length)

    def get_lengths(self, tree):
        self.edges = []
        tree.dfs_traverse(self)
        return self.edges


class FullTree:
    '''Store a tree representing populations'''

    def __init__(self, treeString, data=None, all_bins=None, all_heights=None,
                 spikes=True):
        self.tree = newick.tree.parse_tree(treeString)
        # update the tree to using lists
        self._to_list(self.tree)
        
        if all_heights is not None:
            self.spikes = spikes
            self.edgeLengths = VisitorLength()
            self.no_pop = len(self.edgeLengths.get_lengths(self.tree))
            # initialize the proper branch lengths in generations
            # in self.lengths I save the edge lengths in dfs order
            self.lengths = []
            self.height = all_heights
            self._set_lengths(self.tree, self.height)
        
        if data is not None:
            self.data = data
            self.p0, self.bins = utils.get_bins(all_bins, self.spikes)

    def _to_list(self, tree):
        if type(tree) is not newick.tree.Leaf:
            for i in xrange(len(tree._edges)):
                self._to_list(tree._edges[i][0])
            tree._edges = map(list, tree._edges)

    def _dist_to_leaves(self, tree):
        if type(tree) is newick.tree.Leaf:
            return 0
        # store distance to leaves
        dist = [0]*len(tree._edges)
        for i in xrange(len(tree._edges)):
            dist[i] = self._dist_to_leaves(tree._edges[i][0]) + 1
        for i in xrange(len(tree._edges)):
            if tree._edges[i][2] is None:
                tree._edges[i][2] = 0
            else:
                tree._edges[i][2] = int(tree._edges[i][2])
            tree._edges[i][1] = dist[i]
        return max(dist)

    def _set_lengths(self, tree, all_heights):
        if all_heights == self.height:
            # I am in the root, calculate distance to leaves
            self._dist_to_leaves(tree)
        if type(tree) is newick.tree.Leaf:
            return
        for i in xrange(len(tree._edges)):
            tree._edges[i][2] = all_heights/tree._edges[i][1]
            tree._edges[i][1] = None
            self._set_lengths(tree._edges[i][0], all_heights-tree._edges[i][2])
            self.lengths.append(tree._edges[i][2])

    def update_pop_sizes(self, tree=None, pos=None):
        if tree is None:
            # I am in the root, initialize
            self.update_pop_sizes(self.tree)
            return
        if type(tree) is newick.tree.Leaf:
            return
        for i in xrange(len(tree._edges)):
            self.update_pop_sizes(tree._edges[i][0])
            # update parameters
            tree._edges[i][1] = [tree._edges[i][1], []]
            tree._edges[i][2] = int(tree._edges[i][2])

    def update_param(self, param, tree=None, pos=None):
        if tree is None:
            # I am in the root, initialize            
            self.update_param(param, self.tree, -1)
            return
        if type(tree) is newick.tree.Leaf:
            return pos
        for i in range(len(tree._edges)):
            pos = self.update_param(param, tree._edges[i][0], pos)
            # update parameters
            pos += 1
            N = param[pos]
            tree._edges[i][1] = [N, []]
        return pos

    def update_tran(self, tree=None):
        if tree is None:
            # I am in the root, initialize
            self.update_tran(self.tree)
            return
        if type(tree) is newick.tree.Leaf:
            return
        for i in range(len(tree._edges)):
            self.update_tran(tree._edges[i][0])
            N = tree._edges[i][1][0]
            r = tree._edges[i][2]
            tree._edges[i][1][1] = trans_prob(self.bins, 2*N, self.p0, r,
                                              self.spikes)

    def update_shapes(self, param):
        self.shapes = param
        pb = stats.beta.cdf(self.bins, self.shapes[0], self.shapes[1])
        # the probability for loss/fixation is 0
        self.root_dist = np.concatenate(([0], pb[1:]-pb[:-1], [0]))
        
    def update_pop(self, param):
        self.update_param(param)
        self.update_tran()
        
    def update(self, param):
        self.update_shapes(param[0:2])
        self.update_pop(param[2:])

    def get_params(self, tree=None):
        if tree is None:
            return self.get_params(self.tree)
        if type(tree) is newick.tree.Leaf:
            return []
        params_bran = []
        for (sub_tree, bins, l) in tree._edges:
            new_bran = self.get_params(sub_tree)
            if len(new_bran) > 0:
                params_bran.extend(new_bran)
            N = bins[0]
            params_bran.append(float(l)/(2*N))
        return params_bran
    
    def get_branch_len(self):
        return self.edgeLengths.get_lengths(self.tree)

    def _calculate_branches(self, dist, tree=None, all_leaves=None, bran=None):
        all_pop_sizes = []
        if tree is None:
            all_leaves = Set(self.tree.get_leaves_identifiers())
            for (sub_tree, _, l) in self.tree._edges:
                aux_pop_sizes = self._calculate_branches(dist, sub_tree,
                                                         all_leaves, l)
                all_pop_sizes.extend(aux_pop_sizes)
            return all_pop_sizes
        if type(tree) is not newick.tree.Leaf:
            this_all_leaves = Set(tree.get_leaves_identifiers())
            for (sub_tree, _, l) in tree._edges:
                aux_pop_sizes = self._calculate_branches(dist, sub_tree,
                                                         this_all_leaves, l)
                all_pop_sizes.extend(aux_pop_sizes)
        leaves = tree.get_leaves_identifiers()
        no_leaves = len(leaves)
        remaining_leaves = list(all_leaves.difference(Set(leaves)))
        no_remaining = len(remaining_leaves)
        dist_between = 0
        for leaf in leaves:
            for remaining in remaining_leaves:
                aux = sorted([leaf, remaining])
                dist_between += dist[(aux[0], aux[1])]
        dist_between /= (no_leaves*no_remaining)
        dist_within = 0
        if no_leaves > 1:
            for i in range(no_leaves-1):
                for j in range(i+1, no_leaves):
                    aux = sorted([leaves[i], leaves[j]])
                    dist_within += dist[(aux[0], aux[1])]
            dist_within /= (no_leaves*(no_leaves-1)/2.0)
        # this is the length of the branch going above node
        # dist_between - dist_within
        all_pop_sizes.append(bran/(2*(dist_between - dist_within)))
        return all_pop_sizes

    def guess_param(self):
        # calculate empirical distribution
        obs_alpha, obs_beta = beta.shapes(self.data[3])
        # initialize distance matrix
        dist = {}
        names_pop = sorted(self.data[0].keys())
        no_pop = len(names_pop)
        # start calculating the pairwise distances
        for i in range(no_pop-1):
            for j in range(i+1, no_pop):
                aux = utils.fst_dist(self.data[0][names_pop[i]],
                                     self.data[2][names_pop[i]],
                                     self.data[0][names_pop[j]],
                                     self.data[2][names_pop[j]])
                dist[(names_pop[i], names_pop[j])] = aux
        param = self._calculate_branches(dist)
        param.insert(0, obs_beta)
        param.insert(0, obs_alpha)
        return param

    def __str__(self):
        return self.to_str()

    def to_str(self, scaled=True, params=True):
        def lstr(tree):
            if type(tree) is newick.tree.Leaf:
                return tree.identifier
            tree_str = '('
            sep = ''
            for (n, bins, l) in tree._edges:
                tree_str += sep + lstr(n)
                if params:
                    if type(bins) is list:
                        tree_str += utils.branch_str(bins[0], l, scaled)
                    elif bins is not None:
                        tree_str += ' ' + str(bins) + ' '
                        tree_str += ' : ' + str(l)
                sep = ', '
            return tree_str + ')'
        return lstr(self.tree)


class VisitorLikelihood(newick.tree.TreeVisitor):
    '''Calculate likelihood of data on a tree
    using Felsenstein's peeling algorithm'''

    def _calc_U(self, dst):
        '''extract and combine all the Us of the children'''
        U = np.ones((self.sites, self.states))
        for child, prob, _ in dst.get_edges():
            node = ' '.join(child.get_leaves_identifiers())
            for i in xrange(self.sites):
                for j in xrange(self.states):
                    U[i, j] *= dot(prob[1][j, :], self.U[node][i, :])
        return U

    def post_visit_edge(self, _, __, ___, dst):
        node = ' '.join(dst.get_leaves_identifiers())
        if type(dst) is newick.tree.Leaf:
            self.U[node] = np.ones((self.sites, self.states))
            for i in xrange(self.sites):
                # at the leaf, I have the binomial distribution
                prob = stats.binom.pmf(self.data[0][node][i],
                                       self.data[2][node],
                                       self.p0)
                self.U[node][i, :] = prob
            return
        self.U[node] = self._calc_U(dst)

    def get_lk(self, tree, recalc=True):
        self.p0 = tree.p0
        self.states = len(self.p0)
        self.data = tree.data
        # the number of unique sites, plus the fake lost and fixed sites
        self.sites = len(self.data[1]) + 2
        
        if recalc:
            self.U = {}
            # calculate U's for inner nodes
            tree.tree.dfs_traverse(self)
            # finish off with the root
            self.lk = self._calc_U(tree.tree)
        
        # add the probability distribution at the root
        self.lk_with_root = np.zeros(self.lk.shape)
        for j in xrange(self.states):
            self.lk_with_root[:, j] = self.lk[:, j] * tree.root_dist[j]
        # polymorphism probability
        poly_prob = (1 - np.sum(self.lk_with_root[-1, :]) 
                       - np.sum(self.lk_with_root[-2, :]))
        
        logLK = 0
        for i in xrange(len(self.data[1])):
            s = np.sum(self.lk_with_root[i, :]) / poly_prob
            if math.isnan(s) or s < 0:
                err_msg = ('Found NaN/negative likelihood at site %d.\nTree: '
                           '%s\nLikeliood: %s'
                           % (i, tree.to_str(),
                              ', '.join(map(str, self.lk_with_root[i, :]))))
                raise Exception(err_msg)
            if s == 0:
                return -1e306
            # use the multiplicity of the current site
            logLK += self.data[1][i] * math.log(s)
            
        return logLK
