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

import numpy as np
import scipy.special as special
import scipy.stats as stats

import utils


def shapes(mom):
    '''calculate the alpha and beta shape parameters
    from mean and variance'''
    m = mom[0]
    m1 = 1 - m
    v = mom[1]
    aux = m * m1
    
    if v == 0 or v > aux:
        return -1, -1

    aux = aux / v - 1
    alpha = aux * m
    beta = aux * m1

    return alpha, beta


def shapes_arr(mom):
    '''calculate the alpha and beta shape parameters
    from a list of mean and variance'''
    m = mom[0, :]
    m1 = 1 - m
    v = mom[1, :]
    aux = m * m1
    pos = (v > 0) & (v <= aux)
    aux[pos] = aux[pos] / v[pos] - 1

    mom[2:4, :] = - np.ones((2, len(v)))
    mom[2, pos] = aux[pos] * m[pos]
    mom[3, pos] = aux[pos] * m1[pos]

    return mom


def cond_mom(mom, poly_prob):
    '''calculate conditional mean and variance
    from mean, variance and probabilities of loss and fixation'''
    if poly_prob < 1e-5:
        mom[4:8] = (-1, -1, -1, -1)
        return mom

    # conditional mean
    mom[4] = (mom[0] - mom[9]) / poly_prob
    # conditional variance
    mom[5] = (mom[1] + mom[0]**2 - mom[9]) / poly_prob - mom[4]**2

    # correct almost 0 negative numbers
    if mom[4] < 0 and mom[4] > -1e-5:
        mom[4] = 0
    if mom[5] < 0 and mom[5] > -1e-5:
        mom[5] = 0

    # check if I get moments outside [0, 1]
    if mom[4] < 0 or mom[4] > 1 or mom[5] < 0 or mom[5] > 1:
        err_msg = (('Found illegal conditional moments '
                    'for the beta distribution.\n'
                    'polymorphism probability: %.6e\nmoments: %s')
                   % (poly_prob, ', '.join(map(str, mom))))
        raise Exception(err_msg)

    # calculate conditional shapes
    mom[6:8] = shapes(mom[4:6])

    return mom


def loss_fix(aux, N, alpha, beta):
    '''calculate the loss/fixation probabilities'''
    loss = aux * special.beta(alpha, beta+N)
    fix = aux * special.beta(alpha+N, beta)
    return loss, fix


def moments(N, p0, r, u=0, v=0):
    '''calculate the moments of the allele frequency distribution
    for pure drift and/or mutation
    the moments are stored in an array in the order
    0=mean, 1=var, 2=alpha, 3=beta'''
    mom = np.empty((4, r))

    # some necessary pre-calculations
    N1 = 1 - 1.0 / N

    # mean and variance
    if u == 0 and v == 0:
        mom[0, :] = p0
        mom[1, :] = p0 * (1 - p0) * (1 - np.power(N1, range(1, r + 1)))
    else:
        aux1 = v / (u+v)
        r_range = range(1, r+1)
        aux2 = np.power(1-u-v, r_range)
        aux3 = aux2 * (p0 - aux1)
        aux4 = np.power(N1, r_range)
        mom[0, :] = aux1 + aux3
        mom[1, :] = (aux1 * (1 - aux1) * (1 - aux2**2 * aux4)
                     / (N - (1-u-v)**2 * (N - 1))
                     + aux3 * (1 - 2 * aux1) * (1 - aux2 * aux4)
                     / (N - (1-u-v) * (N - 1))
                     - aux3**2 * (1 - aux4))

    # shape parameters
    mom = shapes_arr(mom)

    return mom

def moments_spikes(N, p0, r, u=0, v=0):
    '''calculate the moments and spikes of the allele frequency distribution
    for pure drift and/or mutations
    the moments are stored in an array in the order
    0=mean, 1=var, 2=alpha, 3=beta,
    4=c.mean, 5=c.var, 6=c.alpha, 7=c.beta, 8=p.0, 9=p.1'''
    mom = np.empty((10, r))
    
    mom[0:4, :] = moments(N, p0, r, u, v)

    # some necessary pre-calculations
    gp = (1-u-v) * p0 + v
    uN = u**N
    vN = v**N
    u1N = (1-u)**N
    v1N = (1-v)**N
    uv1N = (1-u-v)**N

    mom = shapes_arr(mom)
    # fixation probabilities
    mom[8, 0] = (1 - gp)**N
    mom[9, 0] = gp**N
    # probability of polymorphism
    poly_prob = 1 - mom[8, 0] - mom[9, 0]
    # conditional mean and variance
    mom[:, 0] = cond_mom(mom[:, 0], poly_prob)
    
    for i in range(1, r):
        # fixation probabilities
        mom[8, i] = v1N * mom[8, i-1] + uN * mom[9, i-1]
        mom[9, i] = vN * mom[8, i-1] + u1N * mom[9, i-1]
        b_term = special.beta(mom[6, i-1], mom[7, i-1])
        if np.isfinite(b_term) and b_term != 0 and mom[6, i-1] != -1:
            loss, fix = loss_fix(uv1N, N, mom[6, i-1], mom[7, i-1])
            mom[8, i] += poly_prob * loss / b_term
            mom[9, i] += poly_prob * fix / b_term

        # probability of polymorphism
        poly_prob = 1 - mom[8, i] - mom[9, i]
        # conditional mean and variance
        mom[:, i] = cond_mom(mom[:, i], poly_prob)

    return mom


def distr_mom(bins, mean, alpha, beta, loss=None, fix=None):
    '''calculate a discretized distribution
    from the two shape parameters, loss and fixation probabilities'''
    if fix is None:
        # beta distribution
        if alpha == -1 or beta == -1:
            # set full probability in the mean
            res_distr = np.zeros(len(bins) - 1)
            res_distr[utils.pos(mean, bins)] = 1
        else:
            prob = stats.beta.cdf(bins, alpha, beta)
            res_distr = prob[1:] - prob[:-1]
    else:
        # beta distribution with spikes
        poly_prob = 1 - loss - fix
        res_distr = np.zeros(len(bins) + 1)
        res_distr[0] = loss
        res_distr[-1] = fix
        if alpha == -1 or beta == -1:
            # set the rest of the probability in the mean
            res_distr[utils.pos(mean, bins)] += poly_prob
        else:
            prob = stats.beta.cdf(bins, alpha, beta)
            res_distr[1:-1] = poly_prob * (prob[1:] - prob[:-1])

    return res_distr


def distr(bins, N, p0, r, spikes=False):
    '''calculate a beta discretized distribution'''
    if spikes:
        mom = moments_spikes(N, p0, r)
        return distr_mom(bins, mom[4, r-1], mom[6, r-1], mom[7, r-1],
                     mom[8, r-1], mom[9, r-1])
    
    mom = moments(N, p0, r)
    return distr_mom(bins, mom[0, r-1], mom[2, r-1], mom[3, r-1])
