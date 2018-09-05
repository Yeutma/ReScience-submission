#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Define the classes of this project.

The classes defined are:
    * the program's Version class, containing information about which
    World, meta-learning combination, noise, number of steps, runs,
    meta-parameter initial values, etc. to choose from.
    * the World class, defining the world environment in which the agent
    learns.
    * the Metaparameter meta-class, defining the generic equations of
    meta-learning.
    * the meta-parameter classes, that inherit from the Metaparameter
    meta-class and complement with equations of their own.
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
    """Contains the different parameters defining the program's version.

    The attributes are left undefined to add flexibility (allowing to
    add / update the class during the program's execution), but need to
    be defined for the program to run properly.

    Parameters
    ----------
    W : World Class
        Defines the World class (the agent's learning environment).
    meta : list of 3 bools
        Specifies which meta-parameters to perform meta-learning on.
        Syntax: [alpha, beta, gamma]
    noise = bool
        Specifies whether the algorithm should be noisefull or noiseless
    datatype : str, {'t'}
        Specifies the type of data to record
    steps : int
        Number of simulation steps
    nbRuns : int
        Number of simulation runs
    nbConfigs : int
        Number of world configurations (the environment's layout
        switching between configurations)
    alpha, beta, gamma : float
        Initial meta-parameter values
        Alpha & Gamma in ]0;1[, Beta in ]0;+inf[
    meta_config : dict of dicts
        Defines the configuration of the meta-learning equations (beta
        in particular)
    data : str
        Specifies the variables to record, either 'all' or dict
    frequency : int
        Data is recorded every X steps
        Unfortunately, certain information needs to be recorded every
        step, so this attribute isn't so useful.
    subtitle : str
        Data & graphe file subtitle.
    path : dict of str
        Specifies the path to which the log & graph files would be saved
    filename : str
        Specifies the str radical for the log & graph file names.
    """
    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        """Update the attrs, attrs_str & log_filename attributes.

        These will be useful when logging the data.
        """
        for key in kwargs:
            setattr(self, key, kwargs[key])
        attrs = vars(self)
        attrs_to_not_log = [
                'path', 'filename', 'log_filename', 'subtitle', 'data',
                'attrs', 'attrs_str'
                ]
        self.attrs_str = '%r' % {key: attrs[key] for key in attrs.keys()
                             if key not in attrs_to_not_log}
        self.attrs = ast.literal_eval(self.attrs_str)
        if "filename" in kwargs:
            self.log_filename = self.path['log']+self.filename+".txt"


### --- Define the "World' Environment classes ---
class MDP:
    """Markovian Decision Process inspired from (Tanaka et al., 2002).

    The world environment in which the agent learns, with N states
    and rewards R associated to each state.
    At each state S, the agent can choose one of two actions (left /
    right), which will bring it to the state (S-1 / S+1) respectively.
    The states wrap onto themselves, for e.g. going right at state N-1
    brings the agent to state 0 (called a wrap transition).
    Every transition between states is associated a reward. Wrap
    transitions are associated +RL / -RL (large rewards), normal
    transitions -r / +r (small rewards) respectively (left / right).

    Summary of the matrix of transition rewards according to action &
    states:
        left/0:   N-2 <-(-r)-- N-1 <-(+RL)-- 0 <-(-r)-- 1 <-(-r)-- 2
        right/1:  N-2 --(+r)-> N-1 --(-RL)-> 0 --(+r)-> 1 --(+r)-> 2

    Parameters
    ----------
    states : int
        Number of states
    RLs : list of int
        Defines the list of large rewards
        For each world configuration c, the large reward RL = RLs[c]
    rs : list of int
        Defines the list of small rewards
        For each world configuration c, the small reward r = rs[c]
    """

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


# class Navigation:
#     """World W for the 2D Navigational task: from Starting point S,
#     explore 5x5 or 8x8 (distance = 1 or 2) with cost, until reach Goal
#     point G and receive reward rew"""
#
#     def __init__(self,distance=1,rew=1,cost=0):
#         """World rules : any transition to G generates reward and
#         teleports to S. Other transitions may have a cost...
#         """
#         # distance = 2          distance = 1
#         #  _ _ _ _ _ _ _ _       _ _ _ _ _
#         # |_|_|_|_|_|_|_|_|     |_|_|_|_|_|
#         # |_|_|_|_|_|_|_|_|     |_|_|_|G|_|
#         # |_|_|_|_|_|G|_|_|     |_|_|_|_|_|
#         # |_|_|_|_|_|_|_|_|     |_|S|_|_|_|
#         # |_|_|_|_|_|_|_|_|     |_|_|_|_|_|
#         # |_|_|S|_|_|_|_|_|
#         # |_|_|_|_|_|_|_|_|
#         # |_|_|_|_|_|_|_|_|
#
#         self.distance = distance
#         self.worldSize = 3*distance+2
#         self.minBound = np.array([0,0])
#         self.maxBound = np.array([self.worldSize-1,self.worldSize-1])
#
#         # potential [starting positions, goal positions]
#         # Start coordinates = potSG[chosen configuration][0]
#         # Goal coordinates  = potSG[chosen configuration][1]
#         self.potSG = [
#         [np.array([distance,distance]), np.array([2*distance+1,2*distance+1])],
#         [np.array([2*distance+1,distance]), np.array([distance,2*distance+1])],
#         [np.array([distance,2*distance+1]), np.array([2*distance+1,distance])],
#         [np.array([2*distance+1,2*distance+1]), np.array([distance,distance])]]
#         self.worldConfig = -1
#         self.switch()
#         self.allstates = [str(np.array([x, y]))
#                           for x in range(-(self.worldSize-1), self.worldSize-1)
#                           for y in range(-(self.worldSize-1), self.worldSize-1)]
#
#         self.rew = rew
#         self.cost = cost
#         self.steps_to_rew = distance*2.+2
#         self.maxrew = (self.rew*1/self.steps_to_rew + self.cost*
#                              (self.steps_to_rew-1)/self.steps_to_rew)
#         self.minrew = 0
#         self.act = [np.array([+1,0]), np.array([0,-1]),
#                     np.array([-1,0]), np.array([0,+1])]
#         self.actNames = ['up', 'left', 'down', 'right'] # to be verified
#         self.name = 'Navigation'
#
#     def move(self,action):
#         # move agent & return value of transition
#         self.A = np.maximum(np.minimum(self.A + action, self.maxBound),
#                                         self.minBound)
#         if np.all(self.A == self.G):
#             self.A = self.S
#             return self.rew
#         else:
#             return self.cost
#
#     def readState(self):
#         # current state (in string format)
#         return str(self.A - self.S)
#
#     def switch(self):
#         self.worldConfig = (self.worldConfig+1) % 4
#         self.S = self.potSG[self.worldConfig][0]
#         self.G = self.potSG[self.worldConfig][1]
#         self.A = self.S #coordinates of Agent
#
#     def __repr__(self):
#         attrs = vars(self)
#         attrs_to_log = ['name', 'distance', 'rew', 'cost', 'maxrew', 'minrew',
#                         'actNames', 'steps_to_rew']
#         return '%r' % {key: attrs[key] for key in attrs.keys()
#                              if key in attrs_to_log}


### --- Define the meta-parameter classes ---
class Metaparameter:
    """Meta-class defining generic meta-learning equations.

    The hidden state self.b0 is changed / learned according to the
    reward differential, self.b is self.b0 with an added Gaussian noise,
    and self.m is the value of the meta-parameter after self.b is passed
    through the meta-parameter's activation equation.
    Each meta-parameter will inherit from this meta-class, and then
    specify their specific b-->m and inverse m-->b transformations.

    In the absence of noise, self.b = self.b0.

    Parameters
    ----------
    m : float
        Initial value of the meta-parameter
    """
    def __init__(self, m):
        self.m=m; self.b=0.; self.b0=self.m_to_b0();
        self.mu=0.2
        self.sigma=0.; self.nu=1; self.pert_len=100;

    def update(self, meta, noise, d_rew, step):
        """Update the meta-parameter.

        Parameters
        ----------
        meta : list of 3 bools
            Specifies which meta-parameters to perform meta-learning on.
            The meta-parameter's number will determine which bool in
            meta is used (alpha = 0, beta = 1, gamma = 2).
        noise : bool
            Specifies whether the activation value self.b is randomized
        d_rew : float
            The reward differential that will be used to learn.
            If d_rew > 0, the meta-parameter's value self.b0 is updated
            in the direction of self.b, else in the opposite direction.
        """
        if(step % self.pert_len == 0):
            self.sigma = np.random.normal(0,self.nu)
        if(meta[self.num]==1):
            if(noise==True):
                self.b0 += self.mu * d_rew * self.sigma
                self.b = self.b0 + self.sigma
                self.m = self.b_to_m(self.b)
            elif(noise==False):
                self.b0 += self.mu * d_rew
                self.m = self.b_to_m(self.b0)


class Alpha(Metaparameter):
    num = 0
    def m_to_b0(self):
        return np.log(1./self.m-1)
    def b_to_m(self,b):
        return 1./(1+np.exp(b))

class Beta(Metaparameter):
    """Beta meta-parameter class

    According to the meta-parameter configuration, beta's equation can
    be exponential or affine.
    In the case beta's equation is affine, meta_config needs to specify
    the slope a & bias b (ax+b).
    """
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
