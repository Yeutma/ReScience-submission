#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions and classes useful for data handling & analysis."""

try:
    import os.path, sys
    import numpy as np
    import ast
    import types
except ImportError, err:
    print "couldn't load module. %s" % (err)
    sys.exit(2)


def check_convergence(imported_data, convergence_data="reward long average"):
    """Obtain average statistics of the agent's convergence.

    Calculate percentage of each of 3 cases: no solution, unstable
    solution, and stable solution.
    To do so, increment 1 to the case found for each run, then divide
    by sum to obtain average.

    Parameters
    ----------
    imported_data : Data class
        Convergence will be checked on imported_data.data
    """
    # Initiailization of variables
    attrs = imported_data.metadata['attrs']
    maxrews = attrs['W']['maxrews']
    config_steps_idx = imported_data.converted['config_steps_idx']

    epsilon = 0.05
    stats = {config:{'No solution':0, 'Unstable solution':0,
                     'Stable solution':0}
            for config in range(attrs['W']['nbConfigs'])}
    nbRuns = attrs['nbRuns']
    frequency = attrs['frequency']

    if isinstance(imported_data.data, types.GeneratorType):
        run = 0
        for data in imported_data.data:
            reward = data[convergence_data][0]
            stats = run_convergence(stats, reward, config_steps_idx, maxrews,
                                    epsilon, frequency)
            run += 1
            print str(run)+"/"+str(nbRuns)+": "+str(stats)
    else:
        for run in range(nbRuns):
            reward = imported_data.data[convergence_data][run]
            stats = run_convergence(stats, reward, config_steps_idx, maxrews,
                                    epsilon, frequency)
            print str(run)+"/"+str(nbRuns)+": "+str(stats)
    for config in stats:
        config_sum = sum([stats[config][key] for key in stats[config]])
        stats[config] = {key:float(stats[config][key])/config_sum for
                       key in stats[config]}
    print "\nFinal results: "+str(stats)
    return stats

def run_convergence(stats, reward, config_steps_idx, maxrews, epsilon,
                    frequency):
    """Check whether algorithm has converged or not on this run.

    The reward signal has converged if it approaches maxrew (threshold
    at (1-epsilon) * maxrew).
    If the reward signal has converged, we test whether it is stable.
    The reward signal is stable if it stays converged at least 90% of
    the time and its average value doesn't dip below 75% of maxrew.
    According to this, the run is categorized:
        * No solution: hasn't converged
        * Unstable solution: has converged & isn't stable
        * Stable solution: has converged & is stable

    Parameters
    ----------
    stats : dict
        Dictionary of
    reward : list of float
        a
    config_steps_idx : list of int
        a
    maxrews : list of float
        a
    epsilon : float
        a
    frequency: int
        a
    """
    first_step_idx = 1000/frequency
    averaged = np.array(reward)
    for config in range(len(config_steps_idx)):
        # for each world configuration
        converged = (averaged[config_steps_idx[config]] >
                     maxrews[config]*(1-epsilon))
        # Classify agent performance
        if not converged[first_step_idx:].any():
            # If hasn't converged in any case
            stats[config]['No solution'] += 1
        else:
            # Find first converged index while correcting for positive start
            # inherited from past if world configuration has changed.
            first_convergence_idx = first_step_idx+(
                    np.where(converged[first_step_idx:]==True)[0][0])
            once_converged_array = (
                    config_steps_idx[config][first_convergence_idx:])
            # If converged less than 90% of time and
            # averaged value dips below 75% of maxrew, unstable
            dips_too_low = (min(averaged[once_converged_array]) <
                            (maxrews[config]*0.75))
            not_always_converged = (
                    np.mean(converged[first_convergence_idx:]) < 0.9)
            if dips_too_low and not_always_converged:
                stats[config]['Unstable solution'] += 1
            else:
                stats[config]['Stable solution'] += 1
    return stats

def sliding_window(array, sliding_window_size):
    """Return list of sliding average with a window of size window_size."""
    half = int(round(sliding_window_size/2))
    len_array = len(array)
    return_array = [np.nan for x in range(len_array)]
    for x in range(len_array):
        # Negative values on the left aren't permitted, but right values are
        if x <= half:
            return_array[x] = np.mean(array[:x+half])
        elif x > (len_array-half):
            return_array[x] = np.mean(array[x-half:])
        else:
            return_array[x] = np.mean(array[x-half:x+half])
    return return_array


class Data:
    """Handles the data and all metadata associated effectively.

    Allows to record data during a simulation.
    Can import all metadata from a Version class or a log file.
    Can load & write the log file's data & metadata to a new file.
    Can load the parameter configurations from the parameter file.

    Attributes
    ----------
    data : dict of list
        Values of each variable for each simulation run & step.
    metadata : dict of dict
        Contains the attrs, subtitle & header dictionaries.
        The attrs entry contains the attributes of the program's
        Version class associated with the past / present simulation.
        The subtitle is recorded in the log & graph file.
        The header is the log file's header.
    parameters : dict
        Stores the information from the parameter file concerning the
        variables to be recorded.
    converted : dict
        Converts the metadata to more easily usable values.
    """
    special_vars = ["Qvals"]
    header_separator = '\t'
    step_separator = '\t'
    step_join = ';'
    run_separator = '\n'
    parameter_file = 'config/params.config'

    def __init__(self, ver=None, filename=None):
        assert (ver is None or filename is None), \
            "Cannot import from two sources simultaneously: merge conflict."

        self.data = {}
        self.metadata = {}
        self.params = {}

        # Import metadata & params from Version or filename
        if ver is not None:
            self.import_ver(ver)
        if filename is not None:
            self.load_metadata(filename)

    def append(self, key, value, run, step):
        """Append data to the Data.data dictionary."""
        step = step/self.metadata['attrs']['frequency']
        if key not in self.special_vars:
            self.data[key][run][step] = value
        elif key == "Qvals":
            for state in value:
                Qval = list(value[state])
                self.data[key][state][run][step] = [round(Qval[act], 2) for
                         act in range(len(Qval))]

    def convert_steps(self):
        """Convert steps according to recording frequency."""
        frequency = self.metadata['attrs']['frequency']
        switches_steps = ([0] + self.metadata['attrs']['switches_steps'] +
                          [self.metadata['attrs']['steps']])
        configs = range(self.metadata['attrs']['W']['nbConfigs'])

        self.converted = {}
        steps = range(0, self.metadata['attrs']['steps'], frequency)
        self.converted['steps'] = steps
        self.converted['steps_idx'] = range(len(steps))

        config_steps = [[s for s in steps if (
                s >= switches_steps[config] and s < switches_steps[config+1])]
                    for config in configs]
        self.converted['config_steps'] = config_steps
        self.converted['config_steps_idx'] = [
                [s/frequency for s in config_steps[config]]
                for config in configs]

    def convert_data(self, lines_data, file_params):
        """Import file data: convert data from str to actual values."""
        # Separate string data into lists
        for run in range(len(lines_data)):
            lines_data[run] = lines_data[run][:-2].split(self.step_separator)
            for s in range(len(lines_data[0])):
                lines_data[run][s] = lines_data[run][s].split(self.step_join)
        runs = range(len(lines_data))
        steps = range(len(lines_data[0]))

        data = {}
        # Import each variable into data dict
        for i in range(len(self.params['data_name'])):
            var = self.params['data_name'][i]
            vartype = self.params['type'][i]
            file_idx = file_params['data_name'].index(var)
            if vartype == "int":
                data[var] = [[int(lines_data[run][s][file_idx])
                              for s in steps] for run in runs]
            elif vartype == "float":
                data[var] = [[float(lines_data[run][s][file_idx])
                              for s in steps] for run in runs]
            elif vartype == "dict":
                data[var] = [[eval(lines_data[run][s][file_idx])
                              for s in steps] for run in runs]
            # Handle Q-values exception
            if var == "Qvals":
                states = self.metadata['attrs']['W']['allstates']
                tempQ = {state:[[data["Qvals"][run][s][state] for s in steps]
                                for run in runs] for state in states}
                data["Qvals"] = tempQ
        return data

    def import_ver(self, ver):
        """Import metadata & params from Version class, initialize data."""
        # Import metadata and params
        self.read_params(ver.data)
        self.metadata = {'attrs':ver.attrs, 'subtitle':ver.subtitle,
                   'header': self.header_separator.join(
                           self.params['data_name'])}
        self.convert_steps()
        # Initialize data
        states = self.metadata['attrs']['W']['allstates']
        actions = self.metadata['attrs']['W']['actNames']
        self.data = {key:[[None for x in self.converted['steps_idx']]]
                for key in self.params['data_name']
                if key not in self.special_vars}
        if "Qvals" in self.params['data_name']:
            self.data['Qvals'] = {state:[[[0. for action in actions]
                             for step in self.converted['steps_idx']]]
                             for state in states}

    def load(self, filename, data_name="all", n="all"):
        """Load only the data asked in data_name from data file."""
        self.load_metadata(filename)

        # Compare parameters loaded from file with those asked in arguments,
        # and only keep variables present in both
        file_params = self.params
        self.read_params(data_name)
        data_name_asked = [var for var in self.params['data_name'] if var in
                           file_params['data_name']]
        self.read_params(data_name_asked)

        ## Load data
        if n == "all":
            self.data = self.read_all_lines(filename, file_params)
        elif n == "line by line":
            self.data = self.read_line_by_line(filename, file_params)

    def read_all_lines(self, filename, file_params):
        """Read all the lines of the data file."""
        with open(filename, "r") as fileID:
            for _ in xrange(3):
                fileID.readline()
            lines_data = fileID.readlines()
        return self.convert_data(lines_data, file_params)

    def read_line_by_line(self, filename, file_params):
        """Read the lines of the data file one by one (memory issues)."""
        with open(filename, "r") as fileID:
            for _ in xrange(3):
                fileID.readline()
            for line_data in fileID:
                lines_data = [line_data]
                yield self.convert_data(lines_data, file_params)

    def load_metadata(self, filename):
        """Only load the filename's metadata, and build params from it."""
        assert os.path.isfile(filename), \
            "File "+filename+" does not exist."
        with open(filename, "r") as fileID:
            self.metadata['attrs'] = ast.literal_eval(fileID.readline()[:-1])
            self.metadata['subtitle'] = fileID.readline()[:-1]
            self.metadata['header'] = fileID.readline()[:-1]
        self.convert_steps()
        self.params['data_name'] = (
                self.metadata['header'].split(self.header_separator))
        self.read_params(self.params['data_name'])

    def read_params(self, data):
        """Load parameter configurations from parameter file."""
        with open(self.parameter_file,'r') as fileID:
            params = np.genfromtxt(fileID, dtype=None, delimiter=', ',
                          names=True)
        # Load indices & converted names
        if data == "all":
            indices = range(len(params['data_name']))
        else:
            if type(data) == str:
                data = [data]
            indices = [list(params['data_name']).index(keys) for
                       keys in data]
        self.params = {key:[params[key][i] for i in indices] for
                 key in params.dtype.names}

    def write(self, filename):
        """Log data in data file."""
        if not os.path.isfile(filename):
            self.write_metadata(filename)

        current_runs = range(len(self.data[self.params['data_name'][0]]))

        with open(filename, "a") as fileID:
            for r in current_runs: # for each run in current data
                for s in self.converted['steps_idx']: # for each step
                    dataline = []
                    for i in range(len(self.params['data_name'])):
                        # for each var
                        var = self.params['data_name'][i]
                        var_format = self.params['format'][i]
                        if var not in self.special_vars:
                            val = self.data[var][r][s]
                        elif var == 'Qvals':
                            val = {state:self.data[var][state][r][s]
                                    for state in
                                    self.metadata['attrs']['W']['allstates']}
                        if val is None:
                            print (val, var, s, r)
                        dataline += [var_format % val]
                    fileID.write(self.step_join.join(dataline) +
                                 self.step_separator)
                fileID.write(self.run_separator)

    def write_metadata(self, filename):
        """Only log metadata into data file."""
        with open(filename, "w") as fileID:
            print >>fileID, self.metadata['attrs']
            print >>fileID, self.metadata['subtitle']
            print >>fileID, self.metadata['header']
