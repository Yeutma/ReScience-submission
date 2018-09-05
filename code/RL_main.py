#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Replicate (Schweighofer & Doya, 2003) meta-learning results.

The purpose of this project is to replicate (Schweighofer & Doya,
2003), in which a meta-learning reinforcement learning agent solves a
Markovian Decision Problem (MDP) for different initial meta-parameter
values, environment sizes, reinforcement learning algorithms, etc.
The main function in this file performs the simulation and data logging,
graphing and analysis for every different specified condition.
"""

try:
    import os.path, sys
    import time
    #Custom modules
    from RL_sim import *
    from RL_plot import *
    from RL_classes import *
    from RL_data import *
except ImportError, err:
    print "couldn't load module. %s" % (err)
    sys.exit(2)


def main(simulate=True, analyse=True, parallel=False,
         log={'sim':True, 'stats':True},
         graph={'sim':True, 'stats':True}):
    """Simulate, log, graph, & analyse for each specified condition.

    Parameters
    ----------
    simulate : bool, optional
        Simulate the agent learning. Default = True
    analyse : bool, optional
        Analyse the logged data. Default = True
    parallel : int or bool, optional
        Use N cores (false doesn't multithread). Default = False
    log : dict of bool, optional
        Log the resulting simulation / analysis data. Default = True
    graph : dict of bool, optional
        Graph the resulting simulation / analysis data. Default = True
    """
    simulation_start = time.time()

    # Initialize parameters
    root = '/Users/matt/Documents/RL meta/'
    folder = '2018-09-05 - Test/'
    # root = '../../'
    # folder = 'RL_meta/'
    path = {'graph': root+folder+'Graphs/', 'log': root+folder+'Data/'}
    # Path structure: beta_ver, fig, meta_params, states
    v = Version(meta=[1,1,1], noise=1, datatype='t', nbRuns=300,
                alpha=0.1, data='all', frequency=100,
                subtitle='v1.1, New algorithm (affine beta)')
    plots=[#'average reward',
          ["reward short average", "reward long average"],
          ['maximum reward occurence', 'minimum reward occurence'],
          'choice probability', 'Qvals', 'state',
          'alpha', ['alpha_b', 'alpha_b0'],
          'beta', ['beta_b', 'beta_b0'], 'gamma', ['gamma_b', 'gamma_b0']]
    stats = {}


    # Iterate over each different possibility:
    # - Different beta versions (exp, affine)
    # - Different figure versions (Fig 1, Fig 2)
    # - If Fig 1:
    #     - Different initial beta & gamma ((1, 0.99), (10, 0.5))
    #     - Different states & RL ((4, 6), (8, 12), (10, 15))
    for beta_ver in ['exp', 'affine']:
    # for beta_ver in ['exp']:
        beta_version = ('Exponential Beta' if beta_ver == 'exp' else
                  'Affine 1xBeta+1')
        stats[beta_version] = {}

        for fig_version in ['Fig 2', 'Fig 1']:
            stats[beta_version][fig_version] = {}

            if fig_version == 'Fig 1':
                for (beta, gamma) in [(1, 0.99), (10, 0.5)]:
                # for (beta, gamma) in [(1, 0.99)]:
                    meta_version = 'Beta=%i, Gamma=%.2f' % (beta, gamma)
                    stats[beta_version][fig_version][meta_version] = {}

                    for (states, RL) in [(4, 6), (8, 12), (10, 15)]:
                    # for (states, RL) in [(10, 15)]:
                        state_version = '%i States' % states
                        (stats[beta_version][fig_version]
                         [meta_version][state_version]) = {}

                        filename = (
                                'Resimulation (Schweighofer & Doya, 2003) - ' +
                                beta_version + ', ' + fig_version + ', ' +
                                meta_version + ', MDP ' + state_version)

                        newpath = dict(path)
                        newpath['graph'] += filename+'/'

                        for key in newpath:
                            if not os.path.exists(newpath[key]):
                                os.makedirs(newpath[key])
                                print newpath[key]+' created'

                        v.update(meta_config={'beta':{
                                'version':beta_ver, 'slope':1, 'bias':1}},
                                beta=beta, gamma=gamma, path=newpath,
                                W=MDP(states=states, RLs=[RL]),
                                steps=50000, filename=filename)

                        if simulate:
                            if os.path.isfile(v.log_filename):
                                print("%s already exists."
                                      % v.log_filename)
                            else:
                                task_sim(v, log=log['sim'],
                                         parallel=parallel)

                        if graph['sim']:
                            data = Data()
                            data.load(newpath['log']+filename+'.txt',
                                     n="line by line")
                            curves(data, plots, newpath, filename)
                        if analyse:
                            data = Data()
                            data.load(newpath['log']+filename+'.txt',
                                     n="line by line")
                            (stats[beta_version][fig_version]
                                    [meta_version][state_version]) = (
                                    check_convergence(data))

            if fig_version == 'Fig 2':
                beta = 1; gamma = 0.5; states = 8;
                meta_version = 'Beta=%i, Gamma=%.2f' % (beta, gamma)
                state_version = '%i States' % states

                filename = (
                        'Resimulation (Schweighofer & Doya, 2003) - ' +
                        beta_version + ', ' + fig_version + ', ' +
                        meta_version + ', MDP ' + state_version)

                newpath = dict(path)
                newpath['graph'] += filename+'/'

                for key in newpath:
                    if not os.path.exists(newpath[key]):
                        os.makedirs(newpath[key])
                        print newpath[key]+' created'

                v.update(meta_config={'beta':{
                        'version':beta_ver, 'slope':1, 'bias':1}},
                        beta=beta, gamma=gamma, path=newpath,
                        W=MDP(states=states, RLs=[2, 12], rs=[1, 1]),
                        steps=40000, filename=filename)

                if simulate:
                    if os.path.isfile(v.log_filename):
                        print("%s already exists."
                              % v.log_filename)
                    else:
                        task_sim(v, log=log['sim'], parallel=parallel)

                if graph['sim']:
                    data = Data()
                    data.load(newpath['log']+filename+'.txt',
                             n="line by line")
                    curves(data, plots, newpath, filename)
                if analyse:
                    data = Data()
                    data.load(newpath['log']+filename+'.txt',
                             n="line by line")
                    stats[beta_version][fig_version] = (
                            check_convergence(data))


    if log['stats']:
        with open(path['log']+'statistics.txt', 'w') as fileID:
            print >>fileID, stats
    if graph['stats']:
        with open(path['log']+'statistics.txt', 'r') as fileID:
            stats = eval(fileID.readline())
        statistics_plot(path['graph']+'statistics', stats)

    simulation_end = time.time()
    print("Total time elapsed: %.4fs" % (simulation_end-simulation_start))

# ----------------- Main -----------------
if __name__ == '__main__':
    main(simulate=True, analyse=True, parallel=False,
         log={'sim':True, 'stats':True},
         graph={'sim':True, 'stats':True})
