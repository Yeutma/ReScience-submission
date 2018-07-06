#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This file serves to plot the input data into an output pdf file with
rows * columns of either curves, as in the report

--------- Input ---------
   -ver = algorithm version (S&D / Gamma_Only, Noisefull / Noiseless)
          & simulation task (MDP or Navigation)
   -steps = number of simulation steps
   -nbRuns = number of times the simulation is run
   -filenametr = string containing information on all the parameters
   -a, b, g = meta-parameters' alpha, beta & gamma initial values
   -distance = size of the W World, class defined in calling script
               RL.py
   -n_configurations = number of different configurations for the
                       simulation
   -path = file path for logfile
"""

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    import sys
    import types
    from RL_data import *
    from RL_classes import *
except ImportError, err:
    print "couldn't load module. %s" % (err)
    sys.exit(2)


def meta_to_str(meta):
    """Provide a plot string summary of the
    meta-parameter meta-learning state.
    Returns plotstr.
    """
    if(np.sum(meta)==3):
        plotstr = 'Full Meta-learning'
    elif(np.sum(meta)==0):
        plotstr = 'No Meta-learning'
    else:
        plotstr = ''
        if(meta[0] == 1): plotstr += r'$\alpha$ '
        if(meta[1] == 1): plotstr += r'$\beta$ '
        if(meta[2] == 1): plotstr += r'$\gamma$ '
        plotstr += 'Only'
    return plotstr

def plotfig(fig, path, filename, attrs, suffix):
    """Plot figure, with title & subtitle.
    title = [W_name, noise, meta-learning]
    subtitle = [metastr, steps, nbRuns, nbConfigs (, distance)]
    """

    # --- Define title & subtitle ---
    # task
    title = [attrs['W']['name']]

    # noise
    if(attrs['noise'] == 1):
        title += ['Noisefull']
    elif(attrs['noise'] == 0):
        title += ['Noiseless']

    # meta-learning
    title += [meta_to_str(attrs['meta'])]

    # initial values / constants
    subtitle = []
    temp = ''
    temp += r' $\alpha_i$='+str(attrs['alpha']);
    temp += r' $\beta_i$='+str(attrs['beta']);
    temp += r' $\gamma_i$='+str(attrs['gamma']);
    subtitle += [temp[1:]];

    # steps, nbRuns, nbConfigs (, distance)
    temp = ["steps="+str(attrs['steps']), str(attrs['nbRuns'])+" runs",
            str(attrs['W']['nbConfigs'])+" configurations"]
    if(attrs['W']['name'] == 'Navigation'):
        temp += ["distance="+str(attrs['W']['distance'])]
    subtitle += temp

    title = (", ").join(title)
    subtitle = (", ").join(subtitle)

    # --- Plot title & subtitle ---
    fig.suptitle(title, fontsize = 16)
    plt.figtext(.5, .9, subtitle, fontsize=12, ha='center')

    # Define graph filename
    graph_filename = path['graph']+filename+" ("+suffix+").jpg"

    # --- Show / Save fig ---
    # fig = plt.gcf(); plt.show()
    try:
        fig.savefig(graph_filename, dpi=300)
        print graph_filename+" saved"
    except:
        pass
    plt.close(fig)

def curves(imported_data, plots, path, filename):
    # Initialize variables
    colors = {"reward":(1,0,0), "reward short average":(0.7,0,0),
              "reward long average":(0.4,0,0), "average reward":(1,0,0),
              "choice":(1,0,1), "state":(0,0,0),
              "alpha":(0,0,1), "alpha_b":(0,0,0.7), "alpha_b0":(0,0,0.4),
              "beta":(1,0.5,0), "beta_b":(0.7,0.35,0), "beta_b0":(0.4,0.2,0),
              "gamma":(0,1,0), "gamma_b":(0,0.7,0), "gamma_b0":(0,0.4,0),
              "maximum reward occurence":(1,0,0),
              "minimum reward occurence":(0,0,1),
              "choice probability":(1,0,1)}
    linewidths = {"reward":0.05, "reward short average":1,
              "reward long average":1, "average reward":1,
              "choice":0.05, "state":0.05,
              "alpha":1, "alpha_b":1, "alpha_b0":1,
              "beta":1, "beta_b":1, "beta_b0":1,
              "gamma":1, "gamma_b":1, "gamma_b0":1,
              "maximum reward occurence":1,
              "minimum reward occurence":1,
              "choice probability":1}
    metadata = imported_data.metadata
    params = imported_data.params
    converted = imported_data.converted
    # Plot figure one at a time if data = generator, else all together
    # To parallelise
    if isinstance(imported_data.data, types.GeneratorType): # one at a time
        i = 0
        for data in imported_data.data:
            stri = ' %i' % i
            curves_fig(metadata, params, data, converted, plots, path,
                       filename+stri, colors, linewidths)
            i += 1
    else: # all together
        data = imported_data.data
        curves_fig(metadata, params, data, converted, plots, path, filename,
                   colors, linewidths)


def curves_fig(metadata, params, data, converted, plots, path, filename,
               colors, linewidths):

    # Initialize variables
    plots = [[plots[i]] if type(plots[i]) == str else plots[i]
              for i in range(len(plots))]
    flat_plots = [plots[i][j] for i in range(len(plots))
                  for j in range(len(plots[i]))]
    nPlots = len(plots)

    attrs = dict(metadata['attrs'])
    actions = list(attrs['W']['actNames'])
    states = range(attrs['W']['states'])
    steps = list(converted['steps'])
    steps_idx = list(converted['steps_idx'])
    config_steps = list(converted['config_steps'])
    config_steps_idx = list(converted['config_steps_idx'])
    sliding_window_size = round(500./metadata['attrs']['frequency'])
    data_name = list(params['data_name'])
    plotted_name = list(params['plotted_name'])
    data = dict(data)
    runs = range(len(data[data_name[0]]))

    # Initialize figure
    fig, ax = plt.subplots(nPlots, figsize=(10, 20), sharex=True)
    if nPlots==1: ax = [ax]
    plt.tight_layout()
    fig.subplots_adjust(left=0.08, right=0.95, top=0.88, bottom=0.05)

    # --- Transformed data plots ---
    # Average reward
    new = "average reward"
    if new in flat_plots:
        data[new] = [sliding_window(data["reward"][run],
                                   sliding_window_size) for run in runs]
        data_name += [new]
        plotted_name += ["<r>"]

    # Maximum reward occurence
    new = "maximum reward occurence"
    if new in flat_plots:
        if attrs['W']['name'] == 'Navigation':
            maxrew = attrs['W']['rew']
            data[new] = [[1 if data["reward"][run][s]==maxrew
                          else 0 for s in steps_idx]
                         for run in runs]
        elif attrs['W']['name'] == 'MDP':
            maxrews = [abs(x) for x in attrs['W']['RLs']]
            data[new] = [[]]
            for run in runs:
                for config in range(len(config_steps_idx)):
                    data[new][run] += [1
                          if data["reward"][run][s]==maxrews[config]
                          else 0 for s in config_steps_idx[config]]

        data[new] = [sliding_window(data[new][run], sliding_window_size)
                     for run in runs]
        data_name += [new]
        plotted_name += ["P(R)"]

    # Minimum reward occurence
    new = "minimum reward occurence"
    if new in flat_plots:
        if attrs['W']['name'] == 'Navigation':
            minrew = attrs['W']['cost']
            data[new] = [[1 if data["reward"][run][s]==minrew
                          else 0 for s in steps]
                         for run in runs]
        elif attrs['W']['name'] == 'MDP':
            minrews = [-abs(x) for x in attrs['W']['RLs']]
            data[new] = [[]]
            for run in runs:
                for config in range(len(config_steps_idx)):
                    data[new][run] += [1
                          if data["reward"][run][s]==minrews[config]
                          else 0 for s in config_steps_idx[config]]

        data[new] = [sliding_window(data[new][run], sliding_window_size)
                     for run in runs]
        data_name += [new]
        plotted_name += ["P(-R)"]

    # Choice probability
    new = "choice probability"
    if new in flat_plots:
        data[new] = {act:[] for act in actions}
        for a in range(len(actions)):
            act = attrs['W']['actNames'][a]
            data[new][act] = [[True if data['choice'][run][s] == a else
                                False for s in steps_idx] for run in runs]
            data[new][act] = [
                    sliding_window(data[new][act][run], sliding_window_size)
                    for run in runs]
        data_name += [new]
        plotted_name += ["P(choice)"]

    # --- Start plotting ---
    for i in range(len(plots)):
        idx = [data_name.index(plots[i][j]) for j in range(len(plots[i]))]
        for j in range(len(plots[i])):
            name = plots[i][j]
            # --- Plot specifics ---
            # List-type data
            if name not in ["Qvals", "choice probability"]:
                # Plot
                curve_mean = np.mean(data[name], 0)
                curve_std = np.std(data[name], 0)
                # Rewards
                if name in ["average reward", "reward short average",
                                   "reward long average"]:
                    maxrews = attrs['W']['maxrews']
                    minrews = attrs['W']['minrews']
                    ax[i].set_ylim([min(minrews)*1.05, max(maxrews)*1.05])
                    for config in range(len(config_steps)):
                        ax[i].plot(config_steps[config],
                                [minrews[config] for stepi in
                                 config_steps[config]],
                                color=(0,0,0), linewidth=0.5)
                        ax[i].plot(config_steps[config],
                                [maxrews[config] for stepi in
                                 config_steps[config]],
                                color=(0,0,0), linewidth=0.5)
                # Meta-parameters
                elif name in ["alpha", "beta", "gamma"]:
                    if name == "alpha":
                        ylims = [0, 1]
                        start_val = attrs["alpha"]
                        start_txt = str(attrs["alpha"])
                    elif name == "beta":
                        ylims = [0, max(curve_mean)]
                        start_val = attrs["beta"]
                        start_txt = str(attrs["beta"])
                    elif name == "gamma":
                        ylims = [0, 1]
                        start_val = attrs["gamma"]
                        start_txt = str(attrs["gamma"])
                    ylim_span = ylims[1]-ylims[0]
                    ax[i].set_ylim([ylims[0]-ylim_span*0.05,
                                   ylims[1]+ylim_span*0.05])
                    ax[i].plot(0, start_val, 'o')
                    ax[i].text(0, start_val+0.1, start_txt, fontsize=10,
                               color=(0,0,0))
                # Maximum / Minimum rewards
                elif name in ["maximum reward occurence",
                                     "minimum reward occurence"]:
                    height = 1./attrs['W']['steps_to_rew']
                    ax[i].set_ylim([-0.05*height, 1.05*height])
                # State
                elif name in ["state"]:
                    height = max(states)-min(states)
                    ax[i].set_ylim([min(states)-height*0.05,
                                    max(states)+height*0.05])
                # Plot
                ax[i].plot(steps, curve_mean, color=colors[name],
                          linewidth=linewidths[name])
                ax[i].fill_between(steps, curve_mean-curve_std,
                                   curve_mean+curve_std, alpha=0.15,
                                   facecolor=colors[name])
            # Dictionary-type data
            elif name in ["Qvals", "choice probability"]:
                # Q-values
                if name == "Qvals":
                    keys = sorted(data[name].keys())
                    choices = attrs['W']['actNames']
                    labels = [choice+", state "+key
                              for choice in choices for key in keys]
                    colors[name] = cm.rainbow(np.linspace(0, 1,len(labels)))
                    # Plot data
                    for c in range(len(choices)):
                        for k in range(len(keys)):
                            curve_mean = np.mean(
                                [[data[name][keys[k]][run][s][c] for
                                  s in steps_idx]
                                 for run in runs], 0)
                            ax[i].plot(steps, curve_mean,
                                       color=colors[name][c*len(keys)+k])
                    ax[i].legend(labels, ncol=4)
                # Choice Probability
                elif name == "choice probability":
                    colors[name] = cm.rainbow(
                        np.linspace(0, 1,len(attrs['W']['actNames'])))
                    for a in range(len(attrs['W']['actNames'])):
                        act = attrs['W']['actNames'][a]
                        curve_mean = np.mean(data[name][act], 0)
                        ax[i].plot(steps, curve_mean, color=colors[name][a])
                    ax[i].legend(attrs['W']['actNames'])
                    ax[i].set_ylim([-0.05, 1.05])

        ax[i].grid(True)
        ax[i].set_xlim([0, steps[-1]])
        if len(plots[i]) == 1:
            ax[i].set_ylabel(plotted_name[idx[0]])
        else:
            ax[i].legend(ax[i].get_lines()[-len(plots[i]):],
                         [plotted_name[idxi] for idxi in idx])

    ax[-1].set_xlabel('Step number')

    plotfig(fig, path, filename, attrs, 'curves')

def statistics_plot(filename, stats):
    """Plot bar graphs from stats.
    Stats: [beta_version]['Fig 1'][initial parameters][states][configs][sols]
    Stats: [beta_version]['Fig 2'][configs][sols]
    """
    types = ['Affine 1xBeta+1', 'Exponential Beta']
    figs = ['Fig 1', 'Fig 2']
    inits = ['Beta=1, Gamma=0.99', 'Beta=10, Gamma=0.50']
    states = ['4 States', '8 States', '10 States']
    configs = [0, 1]
    sols = ['No solution', 'Unstable solution', 'Stable solution']
    ind = range(len(states))
    x = [0 for k in ind]

    nRows = 3
    nCols = 2
    fig, ax = plt.subplots(nRows, nCols, figsize=(12, 15))
    plt.subplots_adjust(hspace=0.15)

    # Plot Fig 1
    for i in range(len(inits)):
        for t in range(len(types)):
            data = {s:stats[types[t]]['Fig 1'][inits[i]][s][0] for s in states}
            no = [data[s]['No solution']*100 for s in states]
            un = [data[s]['Unstable solution']*100 for s in states]
            st = [data[s]['Stable solution']*100 for s in states]
            x[0] = ax[i][t].bar(ind, no, width=0.35, color='b')
            x[1] = ax[i][t].bar(ind, un, width=0.35, color='g',
                              bottom=no)
            x[2] = ax[i][t].bar(ind, st, width=0.35, color='r',
                             bottom=[no[k]+un[k] for k in ind])
            for s in range(len(states)):
                if no[s] != 0:
                    ax[i][t].text(ind[s], no[s]/2., ('%.2f%%' % no[s]),
                                 horizontalalignment='center',
                                 verticalalignment='center', color='w')
                if un[s] != 0:
                    ax[i][t].text(ind[s], (no[s]*2+un[s])/2., ('%.2f%%' % un[s]),
                                 horizontalalignment='center',
                                 verticalalignment='center', color='w')
                if st[s] != 0:
                    ax[i][t].text(ind[s], ((no[s]+un[s])*2+st[s])/2.,
                                 ('%.2f%%' % st[s]),
                                 horizontalalignment='center',
                                 verticalalignment='center', color='w')
            if i == 1:
                ax[i][t].set_xticks(ind)
                ax[i][t].set_xticklabels([states[k] for k in ind])
            else:
                ax[i][t].tick_params(axis='x', labelbottom='off')
                ax[i][t].set_title(types[t])
            ax[i][0].set_ylabel(inits[i])
            ax[i][t].set_ylim([0, 100])
    # Annotate figure number
    xpos = 0.04
    ypos = [ax[0][0].get_position(), ax[1][0].get_position()]
    ypos = ((ypos[0].y0*2+ypos[0].height)/2. +
      (ypos[1].y0*2+ypos[1].height)/2.)/2.
    fig.text(xpos, ypos, figs[0], horizontalalignment='center',
            verticalalignment='center', color='k', rotation=90,
            fontsize=20)
    # Plot Fig 2
    i = 2
    for t in range(len(types)):
        data = {c:stats[types[t]]['Fig 2'][c] for c in configs}
        no = [data[c]['No solution']*100 for c in configs]
        un = [data[c]['Unstable solution']*100 for c in configs]
        st = [data[c]['Stable solution']*100 for c in configs]
        x[0] = ax[i][t].bar(configs, no, width=0.35, color='b')
        x[1] = ax[i][t].bar(configs, un, width=0.35, color='g',
                          bottom=no)
        x[2] = ax[i][t].bar(configs, st, width=0.35, color='r',
                         bottom=[no[c]+un[c] for c in configs])
        for c in configs:
            if no[c] != 0:
                ax[i][t].text(configs[c], no[c]/2., ('%.2f%%' % no[c]),
                             horizontalalignment='center',
                             verticalalignment='center', color='w')
            if un[c] != 0:
                ax[i][t].text(configs[c], (no[c]*2+un[c])/2., ('%.2f%%' % un[c]),
                             horizontalalignment='center',
                             verticalalignment='center', color='w')
            if st[c] != 0:
                ax[i][t].text(configs[c], ((no[c]+un[c])*2+st[c])/2.,
                             ('%.2f%%' % st[c]),
                             horizontalalignment='center',
                             verticalalignment='center', color='w')
            ax[i][t].set_xticks(configs)
            ax[i][t].set_xticklabels(['Configuration '+str(configs[c])
                    for c in configs])
        ax[i][0].set_ylabel('Beta=1, Gamma=0.5')
        ax[i][t].set_ylim([0, 100])
    # Annotate figure number
    xpos = 0.04
    ypos = ax[2][0].get_position()
    ypos = (ypos.y0*2+ypos.height)/2.
    fig.text(xpos, ypos, figs[1], horizontalalignment='center',
            verticalalignment='center', color='k', rotation=90,
            fontsize=20)
    # Handle legend
    ax[nRows-1][0].legend(x, sols, bbox_to_anchor=(0.5, -0.25, 1.2, .102),
                          loc=3, ncol=3, mode="expand", borderaxespad=0.)

    fig.savefig(filename, dpi=200)


#def Qplot(Q, v):
#    Qkeys = sorted(Q.keys())
#    actionsArray = range(len(Q[Qkeys[1]]))
#    Qarray = [[0 for j in actionsArray] for i in range(len(Qkeys))]
#    for i in range(len(Q)):
#        Qarray[i] = Q[Qkeys[i]]
#    Qarray = np.array(Qarray)
#    Qarray = Qarray.T
#    plt.figure()
#    plt.imshow(Qarray)
#    plt.colorbar()
#    plt.xticks(range(len(Q)), Qkeys)
#    plt.yticks(actionsArray, v.W.actNames)
#    plt.show()


def plot_meta_params(a, b, g, b0=np.arange(-10, 10, 0.1)):
    nPlots = 4
    alpha = Alpha(a)
    exp_beta = Beta(b, {'version':'exp'})
    aff_beta = Beta(b, {'version':'affine', 'slope':1, 'bias':1})
    gamma = Gamma(g)
    fig, ax = plt.subplots(nPlots, sharex=True)
    metas = [alpha, exp_beta, aff_beta, gamma]
    texts = [r'$\alpha$', 'Exponential\n'+r'$\beta$', 'Affine\n'+r'1x$\beta$+1', r'$\gamma$']
    initial_m = [a, b, b, g]
    initial_b = [metas[i].m_to_b0() for i in range(len(initial_m))]
    for i in range(nPlots):
        ax[i].plot(b0, [metas[i].b_to_m(x) for x in b0])
        ax[i].plot(initial_b[i], initial_m[i], 'o')
        ax[i].set_ylabel(texts[i])
    ax[nPlots-1].set_xlabel('b')
    plt.show()

#def rabg_curves(data, attrs, filename, path, type='averaged'):
#    """Plot data curves at every t"""
#    mean=[np.mean(data[i],0) for i in range(len(data))]
#    std=[np.std(data[i],0) for i in range(len(data))]
#    c = [(0,0,1), (0,1,0), (1,0,1), (1,0,0)]
#    n = 1+sum(attrs['meta'])
#    if type == 'individual':
#        runs = range(len(data[0]))
#        maxRew = float(np.max(data[0]))
#        maxRews = [float(np.max(data[0][run])) for run in runs]
#        maxCoeff = [maxRews[run]/maxRew for run in runs]
#        inferno = plt.get_cmap('inferno')
#        cNorm  = colors.Normalize(vmin=0, vmax=maxRew)
#        scalarMap = cm.ScalarMappable(norm=cNorm, cmap=inferno)
#        c = [scalarMap.to_rgba(maxCoeff[run]) for run in runs]
#        suffix = 'RABG individual curves'
#    elif type == 'averaged':
#        suffix = 'RABG averaged curves'
#
#    fig, ax = plt.subplots(n, sharex=True)
#    if n==1: ax = [ax]
#    plt.tight_layout()
#    fig.subplots_adjust(hspace = 0.2, left=0.11,
#                    bottom=0.12, right=0.95, top=0.85)
#    for i in range(n):
#        ax[i].grid(True)
#        ax[i].set_xlim([0,len(mean[i])-1])
#        if type == 'averaged':
#            ax[i].plot(np.arange(len(mean[i])), mean[i], color=c[i])
#            ax[i].fill_between(np.arange(len(mean[i])), mean[i]-std[i],
#                           mean[i]+std[i], alpha=0.15, facecolor = c[i])
#        elif type == 'individual':
#            for run in runs:
#                ax[i].plot(range(len(mean[i])), data[i][run], color=c[run],
#                           linewidth=0.5)
#    ax[-1].set_xlabel('Step number')
#
#    #Specifics
#    i=0
#    ax[i].set_ylabel('Reward Rate')
#    ax[i].set_ylim(attrs['W']['minrew']*1.05,attrs['W']['maxrew']*1.05)
#    ax[i].plot(np.arange(len(mean[i])),[attrs['W']['minrew'] for n in
#    range(len(mean[i]))],'r-',np.arange(len(mean[i])),[attrs['W']['maxrew']
#    for n in range(len(mean[i]))],'r-') #plot maxrew and minrew as red lines
#    ax[i].text(0, attrs['W']['maxrew']*1.1, str(attrs['W']['maxrew']),
#               fontsize=10, color=(1,0,0))
#    if(attrs['meta'][2] == 1):
#        i+=1
#        ax[i].set_ylabel('Gamma')
#        ax[i].set_ylim([-0.05,1.05])
#        ax[i].plot(0, attrs['gamma'], 'o')
#        ax[i].text(0, attrs['gamma']+0.1, str(attrs['gamma']),
#                   fontsize=10, color=(0,0,0))
#    if(attrs['meta'][1] == 1):
#        i+=1
#        ax[i].set_ylabel('Beta')
#        ax[i].set_ylim([-0.05,max(mean[i])*1.05])
#        ax[i].plot(0, attrs['beta'], 'o')
#        ax[i].text(0, min(mean[i])+(max(mean[i])-min(mean[i]))*0.13,
#            str(attrs['beta']), fontsize=10, color=(0,0,0))
#    if(attrs['meta'][0] == 1):
#        i+=1
#        ax[i].set_ylabel('Alpha')
#        ax[i].set_ylim([-0.05,1.05])
#        ax[i].plot(0, attrs['alpha'], 'o')
#        ax[i].text(0, attrs['alpha']+0.1, str(attrs['alpha']),
#                   fontsize=10, color=(0,0,0))
#
#    plotfig(fig, path, filename, attrs, suffix)
#

#def boxplots(data, path, labels, v):
#    """Plot data boxplots of t_rew"""
#    fig, ax = plt.subplots(len(data), figsize=(10,6))
#    fig.subplots_adjust(hspace = 0.3, bottom=0.1, top=0.85, left=0.1,
#        right=0.97)
#    colors = cm.rainbow(np.linspace(0, 1, len(data[0])))
#    for i in range(len(data)):
#        ax[i].grid(True)
#        ax[i].get_xaxis().tick_bottom()
#        ax[i].get_yaxis().tick_left()
#        ax[i].boxplot(data[i], labels = labels, showmeans=True)
#        ax[i].set_title('World configuration n'+str(i), fontsize=12)
#    plotfig(fig, path, v,
#        title=r"Time to (MaxRew-$\epsilon$) - "+", ".join(v.title[:-1]),
#        subtitle=", ".join(v.subtitle))
