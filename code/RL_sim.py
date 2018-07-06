#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This file serves to simulate the meta-learning reinforcement
learning algorithm, defined in oneAgentoneWorld(), in the task, defined
by the calling program RL.py, and passed into task_sim(), so as to log
the evolution of certain parameters in logfile.

--------- Input ---------

--- ver = algorithm version (S&D / Gamma_Only, Noisefull / Noiseles)
          + simulation task (MDP or Navigation)
--- steps = number of simulation steps
--- nbRuns = number of times the simulation is run
--- paramStr = string containing information on all the parameters
--- a, b, g = meta-parameters' alpha, beta & gamma initial values
--- distance = size of the W World, class defined in calling script
RL.py
--- n_configurations = number of different configurations for the
simulation
--- path = file path for logfile

--------- Output ---------

--- logfile = logs the evolution of the parameters (reward, alpha,
beta, gamma, etc) in float-values, each parameter = 1 column
"""

try:
    import numpy as np
    import os, sys
    from RL_classes import *
    from RL_data import *
    from joblib import Parallel, delayed
except ImportError, err:
    print "couldn't load module. %s" % (err)
    sys.exit(2)


def task_sim(v, log=True, parallel=False):
    """Defines the alpha/beta/gamma & Q-table parameters, conserved
    across simulations.
    Simulates the algorithm in either task (Navigation or MDP), and
    logs the performance in data dictionary which is returned.
    """
    print '-----------------------------------------'
    print v.filename
    print '-----------------------------------------'

    # Define steps at which world switches
    v.update(switches_steps=[int(x) for x in 
                     list(np.linspace(0, v.steps, v.W.nbConfigs+1))[1:-1]])

    # If parallel: write in separate files, then aggregate into one file
    # If linear, write in one file directly
    if parallel is not False:
        Parallel(n_jobs=parallel)(
                delayed(run_task)(v, run, (v.path['log']+'%i.txt' % run), log)
                for run in range(v.nbRuns))
        data = Data()
        for run in range(v.nbRuns):
            data.load(v.path['log']+"%i.txt" % run)
            data.write(v.log_filename)
            os.remove(v.path['log']+"%i.txt" % run)
    else:
        for run in range(v.nbRuns):
            run_task(v, run, v.log_filename, log)
    

def run_task(v, run, filename, log):
    """Agent simulation in class W (world) for config_step steps
    (with Q-table and alpha/beta/gamma defined before), with
    meta-learning algorithms on these meta-parameters alpha/beta/gamma.
    Reward and meta-parameters' evolution logged in list data.
    """

    np.random.seed(None)
    print "run:",run+1,"/",v.nbRuns

    # Initialize variables
    v.W.init()
    gamma = Gamma(v.gamma)
    alpha = Alpha(v.alpha)
    beta = Beta(v.beta, v.meta_config['beta'])
    rew = {'shortmean':0.,'longmean':0,'tau1':100.,'tau2':100.}
    Q = {}
    data = Data(ver=v)
    S = v.W.readState() # S(t), state S = coordinates relative to start point
    Stmo = S # S(t-1)
    Q[S] = np.zeros((len(v.W.act)))

    # Loop over steps
    for step in range(v.steps):
        # check if world has switched
        if step in v.switches_steps:
            v.W.switch()
        # Action selection (softmax)
        choiceVals = np.array(Q[S],dtype=np.float64)
        sumExp = (np.exp(choiceVals*beta.m)).sum()
        for i in range(len(v.W.act)):
            choiceVals[i] = np.exp(choiceVals[i]*beta.m)/sumExp
        # Random draw
        draw = np.random.random()
        choice = -1
        sumProba = 0
        while (draw > sumProba and choice < len(choiceVals)-1):
            choice +=1
            sumProba += choiceVals[choice]
        # Action reward & State management
        r = v.W.move(v.W.act[choice])
        Stmo = S
        S = v.W.readState()
        if S not in Q: Q[S] = np.zeros((len(v.W.act)))
        # Metalearning parameters update
        rew['shortmean'] += (r-rew['shortmean']) / rew['tau1']
        rew['longmean'] += (rew['shortmean'] - rew['longmean']) / rew['tau2']
        [x.update(v, (rew['shortmean']-rew['longmean']), step)
         for x in [alpha,beta,gamma]]
        #Q-value update
        delta = r + gamma.m * max(Q[S]) - Q[Stmo][choice]
        Q[Stmo][choice] += alpha.m * delta
        # Log data into Data class
        if log:
            if step % v.frequency == 0:
                for i in range(len(data.params['data_name'])):
                    data.append(data.params['data_name'][i], 
                        eval(data.params['in_program_name'][i]), 0, step)

    # After having looped over every step, log data into file
    if log:
        data.write(filename)