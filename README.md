# SpikeyTree
Inference of divergence times under pure drift and isolation, using the beta with spikes approximation


## What is SpikeyTree?
SpikeyTree infers divergence times between populations that share a common 
history described by a tree. SpikeyTree assumes that the populations have
been evolving under isolation, according to the Wright-Fisher with pure drift.
SpikeyTree can use the Beta or the Beta with Spikes approximation to the allele
frequency distribution under the Wright-Fisher.

More details can be found in 
**Tataru et al.** Inference under a Wright-Fisher model using an accurate beta approximation *Genetics* (2015) 201(3):1133–1141

If using this software in a publication, please cite the above article.


## Contact
Paula Tataru, paula@cs.au.dk


## Requirements
SpikeTree is implemented in python 2.7 and uses third-party python packages.
These should be downloaded and installed accordingly
- SciPy http://sourceforge.net/projects/scipy/files/scipy/
- NumPy http://sourceforge.net/projects/numpy/files/NumPy/
- newick http://users-birc.au.dk/mailund/newick.html


## How to run SpikeyTree
SpikeyTree can be used to
- simulate data on a tree under the Wright-Fisher (src/wf.py)
- infer divergence times on simulated/real data (src/optimize.py)


## Simulating data (src/wf.py)
Running 
```
python src/wf.py
```
will display a brief help:

```
Usage: wf.py [options]

Options:
  -h, --help  show this help message and exit
  -f FILE     file with tree to simulate on
  -s SHAPES   shape parameters for root distribution [default: 1,1]
  -n SAMPLE   sample sizes for populations [default: 100]
  -l LOCI     number of loci/sites to simulate [default: 100]
  -o OUTPUT   file to write simulation to
```
  
FILE should contain a tree in Newick format, containing both the number of 
generations per branch, but also the population size N. For example, 
files/test.txt contains ((C 500 : 44, E 500 : 132) 250 : 14, W 250 : 300); 
This says that population C has been evolving for 44 generations since the 
split from the common ancestor with population E, and its size during these 
44 generations was 500 individuals.

The simulation draws the allele frequency at the root from a beat distribution
with shape parameters given by SHAPES (separated by comma).

SAMPLE specifies how many sequences are sampled from the populations in the
present (i.e. at the leaves). If only one number is given, then all populations
have the same sample size. Otherwise, different sample sizes are separated by
commas. The number of the samples should equal the number of populations in the
present.

LOCI determines the number of simulated (unlinked) loci.

The result of the simulation will be written to OUTPUT. The corresponding 
scaled tree (i.e. with branch lengths given by #gen / 2N) is written to 
tee_OUTPUT.

wf.py simulates only polymorphic data by using rejection samples: simulated
loci that that are fixed or lost in all samples are rejected.

For example, running
```
python src\wf.py -f files\test.txt -s 0.01,0.008 -n 10,15,7 -l 300 -o files\sim.txt
```
will produce two files
- files\sim.txt contain the simulated data
```
((C, E), W);
```
the first line shows the topology (tree without branch lengths) in Newick format
```
10 15 7
```
the second line shows the sample sizes for each population
```
C	E	W
```
the third line is a header with the population names for 
the lines that follow, containing the counts for each locus
- files\tree_sim.txt contains
```
((C: 0.04400, E: 0.13200): 0.02800, W: 0.60000);
```
the tree in Newick format with scaled branches followed by the 
shape parameters used in the simulation and the branch 
lengths in the depth first order; 
the branch above population C is given by 44 / (2*500) = 0.044



## Inference (src/optimize.py)
Running 
```
python src/optimize.py
```
will display a brief help:

```
Usage: optimize.py [options]

Options:
  -h, --help  show this help message and exit
  -f FILE     input file containing data and tree
  -o OUTPUT   output file to write optimization result
  -T HEIGHT   tree height [default: 30]
  -K BINS     number of bins [default: 20]
  -B          run beta; otherwise, run beta with spikes [default: False]
  -r REP      number of repetitions [default: 1]
  -t THREADS  number of threads to run [default: 1]
```

FILE contains the input data, in the same format as the simulated data 
described before.

The results of the optimization will be written to OUTPUT.

The tree height and the number of bins used in the discretization are given
using HEIGHT and BINS. For details of this, see the Supplementary Material of
**Tataru et al.** *Inference under a Wright-Fisher model using an accurate beta approximation*

The option -B toggles between the beta and beta with spikes approximations. By
default, the beta with spikes approximation is used.

The numerical optimization needs a starting point. The starting point can 
affect the resulting optimum found. Therefore, the optimization can be run 
multiple times, each with a different starting point, specified through REP. 
The first point is automatically calculated from Fst obtained from the data, 
while the remaining REP-1 runs will have random generated starting points.

The code can run the multiple runs in parallel, and the number of desired
parallel runs is specified in THREADS.

For example, the command for running the optimization on the previously
simulated data using 4 repetitions on 4 threads is
```
python src\optimize.py -f files\sim.txt -o files\out -r 4 -t 4
```
The program will print to screen the progress of the optimization runs. Each 
line is of the format
```
Iteration x. y. z:   lk
```
where x corresponds to the current run (from 1 to REP), y is the iteration
in the optimization call, and z is the number of calls used to the likelihood
function. lk contains the current log likelihood. The result will be one file,
files\out_spikes.txt. Running the same command with option -B will produce a
file named files\out.txt

The output files contain a summary for all REP runs:
- the maximum log likelihood found
- the starting tree with the scaled branch lengths and shape parameters
- the optimized tree and the shape parameters
- the branch lengths in depth first order of the optimzied tree


## License information
See the file LICENSE.txt for information on terms & conditions for usage,
and a DISCLAIMER OF ALL WARRANTIES.
