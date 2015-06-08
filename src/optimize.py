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

from optparse import OptionParser

import os
import time

from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import numpy.random as nprand
import scipy.optimize as scioptim

from tree import FullTree
from tree import VisitorLikelihood

import utils


class Optim:
    '''Optimize and store extra logging information'''

    def __init__(self, rep, info_tree, mutex, param):
        self.mutex = mutex
        # current likelihood
        self.curr_lk = 0
        # number of function evaluations and iterations
        self.no_eval = 1
        self.no_iter = 1
        # current replication
        self.curr_rep = rep
        self.lk = VisitorLikelihood()
        self.tree = FullTree(info_tree[0], info_tree[1], info_tree[2],
                             info_tree[3], info_tree[4])
        # parameters
        self.init = np.array(param)
        self.init_tran = utils.to_inf(self.init)

    def run(self):
        self.mutex.acquire()
        print 'Starting repetition ', self.curr_rep
        self.mutex.release()
        start_time = time.time()
        optim = scioptim.minimize(self.get_lk, self.init_tran, (),
                                  'L-BFGS-B', callback=self.status)
        end_time = time.time()
        running_time = int(end_time - start_time)
        to_print = ('Finished repetition %d with lk %g in %g min and %g sec'
                    % (self.curr_rep, -optim.fun,
                       running_time / 60, running_time % 60))
        if not optim.success:
            to_print = ('%s\nOptimization did not converge: %s' 
                        % (to_print, optim.message)) 
        self.mutex.acquire()
        print to_print
        self.mutex.release()
        optim.x = utils.to_pos(optim.x)        
        return optim, self.init, self.no_iter, self.no_eval

    def get_lk(self, param):
        untran = utils.to_pos(param)
        self.tree.update(untran)
        self.curr_lk = -self.lk.get_lk(self.tree)
        self.no_eval += 1
        return self.curr_lk

    def status(self, _):
        to_print = ('Iteration %d.%3d.%4d:\t%6.15f' %
                    (self.curr_rep, self.no_iter, self.no_eval, -self.curr_lk))
        self.mutex.acquire()
        print to_print
        self.mutex.release()
        self.no_iter += 1


def threaded_optimize(mutex, runs_queue, optim_queue, info_tree):
    '''call optimization from queue with initial parameters,
    until encountered the sentinel None'''
    while True:
        this_run = runs_queue.get()
        if this_run is None:
            runs_queue.task_done()
            break
        try:
            opt = Optim(this_run[0]+1, info_tree, mutex, this_run[1])
            this_optim, init_param, no_iter, no_eval = opt.run()
            optim_queue.put([this_run[0], this_optim, init_param,
                             no_iter, no_eval])
        except Exception as inst:
            mutex.acquire()
            print 'Exception encountered in repetition', this_run[0]+1
            print inst
            optim_queue.put([this_run[0], None])
        runs_queue.task_done()


def optimize(rep, all_bins, filename, all_heights, spikes, output,
             no_threads, mode):
    '''initialize and start several optimizations'''
    init_tree, data = utils.read_data(filename)    
    info_tree = [init_tree, data, all_bins, all_heights, spikes]
    tree = FullTree(info_tree[0], info_tree[1], info_tree[2], info_tree[3],
                    info_tree[4])
    
    no_pop = tree.no_pop
    
    mutex = multiprocessing.Lock()
    runs_queue = multiprocessing.JoinableQueue()
    optim_queue = multiprocessing.Queue()
    for i in range(no_threads):
        p = multiprocessing.Process(target=threaded_optimize,
                                    args=(mutex, runs_queue, optim_queue,
                                          info_tree, ))
        p.deamon = True
        p.start()
        
    # put the runs in the queue
    param = tree.guess_param()
    runs_queue.put([0, param])    
    for i in range(1, rep):
        # generate random initial values around the guessed ones
        init_param = [p + nprand.uniform(-p, p) for p in param]
        runs_queue.put([i, init_param])
    # put sentinel for each process
    for i in range(no_threads):
        runs_queue.put(None)
    
    runs_queue.join()
    
    # I am done, report results
    report_results(rep, optim_queue, tree, output, mode)


def report_results(rep, optim_queue, tree, output, mode):
    '''report results for all repetitions'''
    best_lk = 1e307
    best_rep = -1
    optim = [None] * rep
    for i in range(rep):
        aux = optim_queue.get()
        optim[aux[0]] = [aux[1], aux[2], aux[3], aux[4]]
        if aux[1] is None:
            print 'Repetition', aux[0], 'terminated with an exception.'
        else:
            if optim[aux[0]][0].fun < best_lk:
                best_lk = optim[aux[0]][0].fun
                best_rep = aux[0]

    if best_rep == -1:
        print 'All repetitions terminated with exceptions. No results.'
    else:
        write_result(tree, optim, best_rep, best_lk, output, mode)


def write_result(tree, optim, best_rep, best_lk, output, mode):
    '''write the result of optimization to file'''
    f = open('%s.txt' % (output), mode=mode)
    f.write('Overall best likelihood %6.15f found at repetition %d\n'
            % (-best_lk, best_rep+1))
            
    for i, opt in enumerate(optim):
        if opt is not None:
            f.write('\n------------- Repetition %d\n' % (i+1))
            f.write(('Maximum log likelihood %6.15f found after %d iterations '
                     'and %d function evaluations\n')
                    % (-opt[0].fun, opt[2], opt[3]))
            tree.update(opt[1])
            f.write('Starting tree:\n')
            utils.write_tree(f, tree, False)
            f.write('Optimized tree:\n')
            tree.update(opt[0].x)
            utils.write_tree(f, tree)
    f.write('\n')
    f.close()


if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option('-f', dest='file',
                      help='input file containing data and tree')
    parser.add_option('-o', dest='output',
                      help='output file to write optimization result')
    parser.add_option('-T', dest='height', type='int', default=30,
                      help='tree height [default: %default]')
    parser.add_option('-K', dest='bins', type='int', default=20,
                      help='number of bins [default: %default]')
    parser.add_option('-B', dest='beta', action='store_true', default=False,
                      help='run beta; otherwise, run beta with spikes [default: %default]')
    parser.add_option('-r', dest='rep', type='int', default=1,
                      help='number of repetitions [default: %default]')
    parser.add_option('-t', dest='threads', type='int', default=1,
                      help='number of threads to run [default: %default]')

    (opt, args) = parser.parse_args()
    
    if not opt.beta:
        print 'Running optimization using beta with spikes'
        opt.output += '_spikes'
    else:
        print 'Running optimization using beta'

    if opt.threads > opt.rep:
        opt.threads = opt.rep
        print 'Number of threads given if larger than required repetitions.'
        print 'Setting number of threads to number of repetitions, ', opt.rep

    optimize(opt.rep, opt.bins, opt.file, opt.height, not opt.beta,
             opt.output, opt.threads, 'w')
