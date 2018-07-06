#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This file serves to define the classes which will used to simulate
the meta-learning reinforcement learning algorithm, such as the
algorithm version, the metal-learning "World" environments
(Navigation / MDP) and meta-parameters.
"""

try:
    import numpy as np
    import sys
    import ast
    from RL_data import *
except ImportError, err:
    print "couldn't load module. %s" % (err)
    sys.exit(2)


### --- Define the algorithm version class ---

class Version:
    """Contains all the different parameters encoding the algorithm
    version:
    -W (str): simulation World = Navigation / MDP / Pendulum
    -meta (list): [1,1,1] = 1/0 {Alpha, Beta, Gamma} Modulated / Not
    -noise (int): 1/0 = {Noisefull, Noiseless}
    -datatype (str): {'t' = data every time step,'t_eps' = time of
    maxrew-eps, 't+rew' = time & maxrew at convergence}
    -steps (int): number of steps in the environment
    -nbRuns (int): number of runs in the environment
    -nbConfigs (int): number of environment configurations
    -gamma & alpha in ]0;1[, beta in ]0;+inf[
    -meta_config (dict of dicts): define metaparameter configuration
    """
    def __init__(self, **kwargs):
        # Default values:
        #W=MDP(), meta=[1,1,1], metalist=[[1,1,1]],
        #noise=1, datatype='t', steps=1000, nbRuns=10, nbConfigs=1,
        #alpha=0.2, beta=1, gamma=0.95, filename='temp',
        #meta_config={'beta':{'version':'exp'}}
        self.update(**kwargs)

    def update(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        attrs = vars(self)
        attrs_to_not_log = ['filename', 'subtitle', 'path', 'attrs',
                            'log_filename', 'data', 'special_vars',
                            'data_name', 'plotted_name', 'in_program_name',
                            'format', 'attrs_str']
        self.attrs_str = '%r' % {key: attrs[key] for key in attrs.keys()
                             if key not in attrs_to_not_log}
        self.attrs = ast.literal_eval(self.attrs_str)
        if "filename" in kwargs:
            self.log_filename = self.path['log']+self.filename+".txt"

### --- Define the "World' Environment classes ---

class Navigation:
    """World W for the 2D Navigational task: from Starting point S,
    explore 5x5 or 8x8 (distance = 1 or 2) with cost, until reach Goal
    point G and receive reward rew"""

    def __init__(self,distance=1,rew=1,cost=0):
        """World rules : any transition to G generates reward and
        teleports to S. Other transitions may have a cost...
        """
        # distance = 2          distance = 1
        #  _ _ _ _ _ _ _ _       _ _ _ _ _
        # |_|_|_|_|_|_|_|_|     |_|_|_|_|_|
        # |_|_|_|_|_|_|_|_|     |_|_|_|G|_|
        # |_|_|_|_|_|G|_|_|     |_|_|_|_|_|
        # |_|_|_|_|_|_|_|_|     |_|S|_|_|_|
        # |_|_|_|_|_|_|_|_|     |_|_|_|_|_|
        # |_|_|S|_|_|_|_|_|
        # |_|_|_|_|_|_|_|_|
        # |_|_|_|_|_|_|_|_|

        self.distance = distance
        self.worldSize = 3*distance+2
        self.minBound = np.array([0,0])
        self.maxBound = np.array([self.worldSize-1,self.worldSize-1])

        # potential [starting positions, goal positions]
        # Start coordinates = potSG[chosen configuration][0]
        # Goal coordinates  = potSG[chosen configuration][1]
        self.potSG = [
        [np.array([distance,distance]), np.array([2*distance+1,2*distance+1])],
        [np.array([2*distance+1,distance]), np.array([distance,2*distance+1])],
        [np.array([distance,2*distance+1]), np.array([2*distance+1,distance])],
        [np.array([2*distance+1,2*distance+1]), np.array([distance,distance])]]
        self.worldConfig = -1
        self.switch()
        self.allstates = [str(np.array([x, y]))
                          for x in range(-(self.worldSize-1), self.worldSize-1)
                          for y in range(-(self.worldSize-1), self.worldSize-1)]

        self.rew = rew
        self.cost = cost
        self.steps_to_rew = distance*2.+2
        self.maxrew = (self.rew*1/self.steps_to_rew + self.cost*
                             (self.steps_to_rew-1)/self.steps_to_rew)
        self.minrew = 0
        self.act = [np.array([+1,0]), np.array([0,-1]),
                    np.array([-1,0]), np.array([0,+1])]
        self.actNames = ['up', 'left', 'down', 'right'] # to be verified
        self.name = 'Navigation'

    def move(self,action):
        # move agent & return value of transition
        self.A = np.maximum(np.minimum(self.A + action, self.maxBound),
                                        self.minBound)
        if np.all(self.A == self.G):
            self.A = self.S
            return self.rew
        else:
            return self.cost

    def readState(self):
        # current state (in string format)
        return str(self.A - self.S)

    def switch(self):
        self.worldConfig = (self.worldConfig+1) % 4
        self.S = self.potSG[self.worldConfig][0]
        self.G = self.potSG[self.worldConfig][1]
        self.A = self.S #coordinates of Agent

    def __repr__(self):
        attrs = vars(self)
        attrs_to_log = ['name', 'distance', 'rew', 'cost', 'maxrew', 'minrew',
                        'actNames', 'steps_to_rew']
        return '%r' % {key: attrs[key] for key in attrs.keys()
                             if key in attrs_to_log}


class MDP:
    """World W for the Markovian Decision Process task (Tanak et al.,
    2002): from starting point S, explore 2 paths (left/right) with
    -r/+r rewards for n-1 states, and 1 state with +R/-R reward"""

    def __init__(self,states=10,RLs=[15],rs=[1]):
        ## Initialization independent from particular configuration
        # General properties
        self.name = 'MDP'
        self.states = states
        self.allstates = [str(s) for s in range(states)]
        self.steps_to_rew = self.states
        self.center = int(np.floor(states/2.))
        assert len(rs) == len(RLs), \
             'Not same number of configurations for rs and RLs'
        self.nbConfigs = len(RLs)
        # Agent actions: left = 0, right = 1
        self.actNames = ['left', 'right']
        self.act = [0,1]
        # Calculate reward statistics for each case
        self.RLs = RLs
        self.rs = rs
        self.paths_total_reward = [
            [RL-r*(self.states-1), -RL+r*(self.states-1)]
            for (RL, r) in [(RLs[k], rs[k]) for k in range(self.nbConfigs)]]
        self.maxrews = [max(self.paths_total_reward[k])/float(self.states) for
                      k in range(self.nbConfigs)]
        self.minrews = [min(self.paths_total_reward[k])/float(self.states) for
                      k in range(self.nbConfigs)]
#        self.optimal_choices = [
#            self.paths_total_reward[k].index(max(self.paths_total_reward[k]))
#            for k in range(self.nbConfigs)]

        ## Initialize first configuration
        self.init()

    def init(self):
        self.config = -1
        self.switch()

    def move(self,action):
        # move agent & return value of transition
        if(action == 0): #left
            self.A=(self.A-1)%self.states
        elif(action == 1): #right
            self.A=(self.A+1)%self.states
        return self.rew[self.A][action]

    def readState(self):
        # current state (in string format)
        return str(self.A)

    def switch(self):
        # Switch configuration
        self.config += 1
        self.r = self.rs[self.config]
        self.RL = self.RLs[self.config]
        # matrix of arrival rewards
        # left/0:   n-2 <--(-r)-- n-1 <--(+RL)-- 0 <--(-r)-- 1 <--(-r)-- 2
        # right/1:  n-2 --(+r)--> n-1 --(-RL)--> 0 --(+r)--> 1 --(+r)--> 2
        self.rew = ([[-self.r,-self.RL]] +
                   [[-self.r,+self.r] for x in range(self.states-2)] +
                   [[+self.RL,+self.r]])
        self.S = self.center
        self.A = self.S

    def __repr__(self):
        attrs = vars(self)
        attrs_to_log = ['name', 'states', 'RLs', 'rs', 'maxrews', 'minrews',
                    'actNames', 'steps_to_rew', 'nbConfigs', 'allstates']
        return '%r' % {key: attrs[key] for key in attrs.keys()
                             if key in attrs_to_log}


### --- Define the meta-parameter classes ---

class Metaparameter:
    def __init__(self, m):
        self.m=m; self.b=0.; self.b0=self.m_to_b0();
        self.mu=0.2
        self.sigma=0.; self.nu=1; self.pert_len=100;

    def update(self, v, d_rew, step):
        if(step % self.pert_len == 0):
            self.sigma = np.random.normal(0,self.nu)
        if(v.meta[self.num]==1):
            if(v.noise==1):
                self.b0 += self.mu * d_rew * self.sigma
                self.b = self.b0 + self.sigma
                self.m = self.b_to_m(self.b)
            elif(v.noise==0):
                self.b0 += self.mu * d_rew
                self.m = self.b_to_m(self.b0)


class Alpha(Metaparameter):
    num = 0
    def m_to_b0(self):
        return np.log(1./self.m-1)
    def b_to_m(self,b):
        return 1./(1+np.exp(b))

class Beta(Metaparameter):
    num = 1
    def __init__(self, m, meta_config):
        self.version = meta_config['version']
        assert self.version in ['exp', 'affine'], 'Beta version not recognized'
        if self.version == 'affine':
            self.slope = meta_config['slope']
            self.bias = meta_config['bias']
        Metaparameter.__init__(self, m)

    def m_to_b0(self):
        if self.version == 'exp':
            return np.log(self.m)
        elif self.version == 'affine':
            return (self.m-self.bias)/self.slope
    def b_to_m(self,b):
        if self.version == 'exp':
            return np.exp(b)
        elif self.version == 'affine':
            # m restricted to [0:+inf[
            return (self.slope*b + self.bias if self.slope*b > -self.bias
                   else 0)

class Gamma(Metaparameter):
    num = 2
    def m_to_b0(self):
        return np.log(1./(1-self.m)-1)
    def b_to_m(self,b):
        return 1/(1+np.exp(-b))
