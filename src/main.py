from __future__ import unicode_literals
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
from tqdm import tqdm
import operator
from enum import *
import copy
import os
import math
import json

class State(Enum): # define the four states of the nodes.
    Ignorant = 0
    Exposed_r = 1
    Exposed_t = 2
    Spreader_r = 3
    Spreader_t = 4
    Removed = 5
    Out = 6
    
class ES_Params:
    def __init__(self, birth, death, alpha_t, beta_t, gamma_t, 
                                    alpha_r, beta_r, gamma_r, 
                                    epsilon_t, epsilon_r,
                                    st_init, sr_init, et_init, er_init):
        self.birth = birth
        self.death = death
        self.alpha_t = alpha_t
        self.alpha_r = alpha_r
        self.beta_t = beta_t
        self.beta_r = beta_r
        self.gamma_t = gamma_t
        self.gamma_r = gamma_r
        self.epsilon_t = epsilon_t
        self.epsilon_r = epsilon_r
        self.st_init = st_init
        self.sr_init = sr_init
        self.et_init = et_init
        self.er_init = er_init
    modeltype = 'ES'

class SS_Params:
    def __init__(self, birth, death, alpha_t, beta_t, gamma_t, 
                                    alpha_r, beta_r, gamma_r, 
                                    delta_t, delta_r,
                                    st_init, sr_init,et_init, er_init):
        self.birth = birth
        self.death = death
        self.alpha_t = alpha_t
        self.alpha_r = alpha_r
        self.beta_t = beta_t
        self.beta_r = beta_r
        self.gamma_t = gamma_t
        self.gamma_r = gamma_r
        self.delta_t = delta_t
        self.delta_r = delta_r
        self.st_init = st_init
        self.sr_init = sr_init
        self.et_init = et_init
        self.er_init = er_init
    modeltype = 'SS'


def reset(G):
    """ 
    :param G: The graph to reset
    
    Initialise/reset all the nodes in the network to be ignorant. 
    Used to initialise the network at the start of an experiment
    """
    nx.set_node_attributes(G, name='state', values=State.Ignorant)

        
def initialise_rumor_random(G, num_sr,num_st, num_er, num_et):
    """
    :param G: Graph to infect nodes on
    :param num_spreaders: Number of initial spreaders on graph G
    
    Set the state of a random selection of nodes to be infected. 
    num_spreaders specifices how many spreaders to make, the nodes 
    are chosen randomly from all nodes in the network
    """
    spreader_r_nodes = random.sample(G.nodes(), num_sr)
    spreader_t_nodes = random.sample(G.nodes()-spreader_r_nodes, num_st)
    exposed_r_nodes = random.sample(G.nodes()-(spreader_r_nodes+spreader_t_nodes), num_er)
    exposed_t_nodes = random.sample(G.nodes()-(spreader_r_nodes+spreader_t_nodes+exposed_r_nodes), num_et)
    for n in spreader_r_nodes:
        G.nodes[n]['state'] = State.Spreader_r
    for n in spreader_t_nodes:
        G.nodes[n]['state'] = State.Spreader_t
    for n in exposed_t_nodes:
        G.nodes[n]['state'] = State.Exposed_t
    for n in exposed_r_nodes:
        G.nodes[n]['state'] = State.Exposed_r


def initialise_truth_degree(G, num_sr, num_st, num_er, num_et):
    """
    :param G: Graph to infect nodes on
    :param num_to_infect: Number of nodes to infect on G
    
    Set the state of a selection of nodes to be infected. Nodes are
    chosen by degree, lasgest degree first.
    numToInfect specifices how many infections to make, the nodes 
    are chosen randomly from all nodes in the network
    """
    degrees = dict(G.degree()) #get degrees of every node
    #below we sort the nodes in order of their degree, highest degree first.
    spreaders_of_truth = []
    spreaders_of_rumor = []
    nodes_sorted_by_degree = sorted(degrees.items(), key=operator.itemgetter(1), reverse=True) 
    nodes_sorted_by_degree_inc = sorted(degrees.items(), key=operator.itemgetter(1), reverse=False) 
    for x in range(num_st): 
        spreaders_of_truth.append(nodes_sorted_by_degree[x][0])
    for x in range(num_sr):
        spreaders_of_rumor.append(nodes_sorted_by_degree_inc[x][0])

    exposed_r_nodes = random.sample(G.nodes()-(spreaders_of_rumor+spreaders_of_truth), num_er)
    exposed_t_nodes = random.sample(G.nodes()-(spreaders_of_rumor+spreaders_of_truth+exposed_r_nodes), num_et)

    for n in spreaders_of_truth:
        G.nodes[n]['state'] = State.Spreader_t
    for n in spreaders_of_rumor:
        G.nodes[n]['state'] = State.Spreader_r
    for n in exposed_r_nodes:
        G.nodes[n]['state'] = State.Exposed_r
    for n in exposed_t_nodes:
        G.nodes[n]['state'] = State.Exposed_t

def initialise_rumor_degree(G, num_sr, num_st, num_er, num_et):
    """
    :param G: Graph to infect nodes on
    :param num_to_infect: Number of nodes to infect on G
    
    Set the state of a selection of nodes to be infected. Nodes are
    chosen by degree, lasgest degree first.
    numToInfect specifices how many infections to make, the nodes 
    are chosen randomly from all nodes in the network
    """
    degrees = dict(G.degree()) #get degrees of every node
    #below we sort the nodes in order of their degree, highest degree first.
    spreaders_of_rumor = []
    spreaders_of_truth = []
    nodes_sorted_by_degree = sorted(degrees.items(), key=operator.itemgetter(1), reverse=True)
    nodes_sorted_by_degree_inc = sorted(degrees.items(), key=operator.itemgetter(1), reverse=False)
    for x in range(num_sr):
        spreaders_of_rumor.append(nodes_sorted_by_degree[x][0])
    for x in range(num_st):
        spreaders_of_truth.append(nodes_sorted_by_degree_inc[x][0])

    exposed_r_nodes = random.sample(G.nodes()-(spreaders_of_rumor+spreaders_of_truth), num_er)
    exposed_t_nodes = random.sample(G.nodes()-(spreaders_of_rumor+spreaders_of_truth+exposed_r_nodes), num_et)

    for n in spreaders_of_truth:
        G.nodes[n]['state'] = State.Spreader_t
    for n in spreaders_of_rumor:
        G.nodes[n]['state'] = State.Spreader_r
    for n in exposed_r_nodes:
        G.nodes[n]['state'] = State.Exposed_r
    for n in exposed_t_nodes:
        G.nodes[n]['state'] = State.Exposed_t


def initialise_truth_betweenness(G, num_sr, num_st, num_er, num_et):
    #below we sort the nodes in order of their betweenness centrality, highest first.
    spreaders_of_truth = []
    spreaders_of_rumor = []
    bet_cen = dict(nx.betweenness_centrality(G,k=100))
    nodes_sorted_by_betweenness = sorted(bet_cen.items(), key=operator.itemgetter(1), reverse=True)
    nodes_sorted_by_betweenness_inc = sorted(bet_cen.items(), key=operator.itemgetter(1), reverse=False)
    for x in range(num_st): 
        spreaders_of_truth.append(nodes_sorted_by_betweenness[x][0])
    for x in range(num_sr):
        spreaders_of_rumor.append(nodes_sorted_by_betweenness_inc[x][0])

    exposed_r_nodes = random.sample(G.nodes()-(spreaders_of_rumor+spreaders_of_truth), num_er)
    exposed_t_nodes = random.sample(G.nodes()-(spreaders_of_rumor+spreaders_of_truth+exposed_r_nodes), num_et)

    for n in spreaders_of_truth:
        G.nodes[n]['state'] = State.Spreader_t
    for n in spreaders_of_rumor:
        G.nodes[n]['state'] = State.Spreader_r
    for n in exposed_r_nodes:
        G.nodes[n]['state'] = State.Exposed_r
    for n in exposed_t_nodes:
        G.nodes[n]['state'] = State.Exposed_t

def initialise_rumor_betweenness(G, num_sr, num_st, num_er, num_et):
    #below we sort the nodes in order of their betweenness centrality, highest first.
    spreaders_of_rumor = []
    spreaders_of_truth = []
    bet_cen = dict(nx.betweenness_centrality(G,k=100))
    nodes_sorted_by_betweenness = sorted(bet_cen.items(), key=operator.itemgetter(1), reverse=True)
    nodes_sorted_by_betweenness_inc = sorted(bet_cen.items(), key=operator.itemgetter(1), reverse=False)
    for x in range(num_sr):
        spreaders_of_rumor.append(nodes_sorted_by_betweenness[x][0])
    for x in range(num_st):
        spreaders_of_truth.append(nodes_sorted_by_betweenness_inc[x][0])

    exposed_r_nodes = random.sample(G.nodes()-(spreaders_of_rumor+spreaders_of_truth), num_er)
    exposed_t_nodes = random.sample(G.nodes()-(spreaders_of_rumor+spreaders_of_truth+exposed_r_nodes), num_et)

    for n in spreaders_of_truth:
        G.nodes[n]['state'] = State.Spreader_t
    for n in spreaders_of_rumor:
        G.nodes[n]['state'] = State.Spreader_r
    for n in exposed_r_nodes:
        G.nodes[n]['state'] = State.Exposed_r
    for n in exposed_t_nodes:
        G.nodes[n]['state'] = State.Exposed_t


def spread_model_factory_ES(esparams):

    def spread_model(n, G):
        list_to_expose_r = []
        list_spreaders_r = []
        list_to_expose_t = []
        list_spreaders_t = []
        removeMyself = False
        fadeout = False

        if G.nodes[n]['state'] == State.Spreader_r:
            for k in G.neighbors(n):
                if G.nodes[k]['state'] == State.Ignorant:
                    if random.random() <= esparams.alpha_r:
                        list_to_expose_r.append(k)
                elif G.nodes[k]['state'] == State.Exposed_t:
                    if random.random() <= esparams.epsilon_t:
                        list_to_expose_r.append(k)
            if random.random() <= esparams.gamma_r:
                removeMyself = True
        elif G.nodes[n]['state'] == State.Spreader_t:
            for k in G.neighbors(n):
                if G.nodes[k]['state'] == State.Ignorant:
                    if random.random() <= esparams.alpha_t:
                        list_to_expose_t.append(k)
                elif G.nodes[k]['state'] == State.Exposed_r:
                    if random.random() <= esparams.epsilon_r:
                        list_to_expose_t.append(k)
            if random.random() <= esparams.gamma_t:
                removeMyself = True
        elif G.nodes[n]['state'] == State.Exposed_r:
            if random.random() <= esparams.beta_r:
                list_spreaders_r.append(n)
        elif G.nodes[n]['state'] == State.Exposed_t:
            if random.random() <= esparams.beta_t:
                list_spreaders_t.append(n)
        if random.random() <= esparams.death:
            fadeout = True
        return list_to_expose_r,list_to_expose_t,list_spreaders_r,list_spreaders_t,removeMyself,fadeout
    return spread_model


def spread_model_factory_SS(ssparams):
    """
    
    Creates and returns an instance of a spread model. This allows us 
    to create a number of models with different alpha, beta, and gamma parameters.
    Note that in a single time step an infected node can infect their neighbors
    and then be removed. 
    """
    def spread_model(n, G):
        list_to_expose_r = []
        list_spreaders_r = []
        list_to_expose_t = []
        list_spreaders_t = []
        removeMyself = False
        fadeout = False

        if G.nodes[n]['state'] == State.Spreader_t:
            for k in G.neighbors(n):
                if G.nodes[k]['state'] == State.Ignorant:
                    if random.random() <= ssparams.alpha_t:
                        list_to_expose_t.append(k)
                elif G.nodes[k]['state'] == State.Spreader_r:
                    if random.random() <= ssparams.delta_r:    
                        removeMyself = True
            if random.random() <= ssparams.gamma_r:
                removeMyself = True
        elif G.nodes[n]['state'] == State.Spreader_r:
            for k in G.neighbors(n):
                if G.nodes[k]['state'] == State.Ignorant:
                    if random.random() <= ssparams.alpha_r:
                        list_to_expose_r.append(k)
                elif G.nodes[k]['state'] == State.Spreader_t:
                    if random.random() <= ssparams.delta_t:
                        removeMyself = True
            if random.random() <= ssparams.gamma_t:
                removeMyself = True
        elif G.nodes[n]['state'] == State.Exposed_r:
            if random.random() <= ssparams.beta_r:
                list_spreaders_r.append(n)
        elif G.nodes[n]['state'] == State.Exposed_t:
            if random.random() <= ssparams.beta_t:
                list_spreaders_t.append(n)
        if random.random() <= ssparams.death:
            fadeout = True
        return list_to_expose_r,list_to_expose_t,list_spreaders_r,list_spreaders_t,removeMyself,fadeout
    return spread_model


def avoid_bias_exposed(G,etruth, erumor):
    list3 = set(etruth) & set(erumor)

    list3 = list(list3)

    etruth_new = []
    erumor_new = []
    sr = 0
    st = 0

    if list3:
        for i in etruth:
            if i not in list3:
                etruth_new.append(i)
        for i in erumor:
            if i not in list3:
                erumor_new.append(i)
        for n in etruth_new:
            G.nodes[n]['state'] = State.Exposed_t
        for n in erumor_new:
            G.nodes[n]['state'] = State.Exposed_r
        for n in list3:
            for k in G.neighbors(n):
                if G.nodes[k]['state'] == State.Spreader_t:
                    st+=1
                elif G.nodes[k]['state'] == State.Spreader_r:
                    sr+=1
            if sr > st:
                G.nodes[n]['state'] = State.Exposed_r
            elif st > sr:
                G.nodes[n]['state'] = State.Exposed_t
    else:
        for n in etruth:
            G.nodes[n]['state'] = State.Exposed_t
        for n in erumor:
            G.nodes[n]['state'] = State.Exposed_r


def apply_spread(G, 
                list_of_newly_exposed_r,
                list_of_newly_exposed_t, 
                list_of_new_spreaders_r, 
                list_of_new_spreaders_t, 
                list_of_newly_removed,
                list_out,
                birth):
    for i in range(birth):
        add_random_node_birth(G,2)

    avoid_bias_exposed(G, list_of_newly_exposed_t,list_of_newly_exposed_r)

    for n in list_of_new_spreaders_t:
        G.nodes[n]['state'] = State.Spreader_t
    for n in list_of_new_spreaders_r:
        G.nodes[n]['state'] = State.Spreader_r
    for n in list_of_newly_removed:
        G.nodes[n]['state'] = State.Removed
    for n in list_out:
        G.nodes[n]['state'] = State.Out

def get_last_node(G):
    return list(G.nodes)[-1]

def add_random_node_birth(G, max_edges):
    node_to_add = get_last_node(G) + 1
    leng = len(G.nodes)
    G.add_node(node_to_add)
    edges = random.randint(1, max_edges)
    for i in range(edges):
        new_edge = int(random.random()*(len(G.nodes)-1))
        trials = 0
        while (G.nodes[new_edge]['state'] == State.Out) and trials < 1000:
            new_edge = int(random.random()*(len(G.nodes)-1))
            trials+=1
        G.add_edge(node_to_add, new_edge)
    G.nodes[node_to_add]['state'] = State.Ignorant

    

def execute_one_step(G, model,birth):
    """
    :param G: the Graph on which to execute the infection model
    :param model: model used to infect nodes on G

    executes the infection model on all nodes in G
    """
    new_nodes_to_expose_r=[]
    new_nodes_to_expose_t=[] #nodes to infect after executing all nodes this time step  
    new_nodes_to_spreaders_r=[]
    new_nodes_to_spreaders_t=[] #nodes to turn into spreaders after this time step
    new_nodes_to_remove=[] #nodes to set to removed after this time step
    nodes_to_fadeout=[]

    # nodes_to_add = 0
    for n in G:
        exposed_n_r, exposed_n_t ,spreaders_n_r, spreaders_n_t,remove,fadeout = model(n, G)
        new_nodes_to_expose_r = new_nodes_to_expose_r + exposed_n_r
        new_nodes_to_expose_t = new_nodes_to_expose_t + exposed_n_t
        new_nodes_to_spreaders_r = new_nodes_to_spreaders_r + spreaders_n_r
        new_nodes_to_spreaders_t = new_nodes_to_spreaders_t + spreaders_n_t
        if remove:
            new_nodes_to_remove.append(n)
        if fadeout:
            nodes_to_fadeout.append(n)
    apply_spread(G,
                new_nodes_to_expose_r, 
                new_nodes_to_expose_t, 
                new_nodes_to_spreaders_r, 
                new_nodes_to_spreaders_t, 
                new_nodes_to_remove, 
                nodes_to_fadeout,
                birth)

def get_infection_stats(G):
    """
    :param G: the Graph on which to execute the infection model
    :returns: a tuple containing three lists of ignorant, exposed, spreader, and removed nodes.

    Creates lists of nodes in the graph G that are ignorant, exposed, spreader, and removed.
    """
    ignorant = []
    exposed_r = []
    exposed_t = []
    spreader_r = []
    spreader_t = []
    removed = []
    for n in G:
        if G.nodes[n]['state'] == State.Ignorant:
            ignorant.append(n)
        elif G.nodes[n]['state'] == State.Exposed_r:
            exposed_r.append(n)
        elif G.nodes[n]['state'] == State.Exposed_t:
            exposed_t.append(n)
        elif G.nodes[n]['state'] == State.Spreader_r:
            spreader_r.append(n)
        elif G.nodes[n]['state'] == State.Spreader_t:
            spreader_t.append(n)
        elif G.nodes[n]['state'] == State.Removed:
            removed.append(n)

    return ignorant, exposed_r, exposed_t, spreader_r, spreader_t, removed


def print_infection_stats(G):
    """
    :param G: the Graph on which to execute the infection model

    Prints the number of succeptible, infected and removed nodes in graph G.
    """
    i,er,et,sr,st,r = get_infection_stats(G)
    print("Ignorant: %d; Exposed to Rumor: %d; Exposed to Truth: %d; Spreader of Rumor: %d; Spreaders of Truth: %d; Removed %d"% (len(i),len(er),len(et),len(sr),len(st),len(r)))

def run_spread_simulation(G, model, params):

    i_results = []
    er_results = []
    et_results = []
    sr_results = []
    st_results = []
    r_results = []

    i,er,et,sr,st,r= get_infection_stats(G)
    stop = 0

    pos=nx.spring_layout(G,dim=2,k=1)

    while stop < 50:
        norm = float(len(i) + len(er) + len(et) + len(sr) + len(st) + len(r))
#         norm = 1
        i_results.append(len(i)/norm)
        er_results.append(len(er)/norm)
        et_results.append(len(et)/norm) 
        sr_results.append(len(sr)/norm)
        st_results.append(len(st)/norm) 
        r_results.append(len(r)/norm)
        execute_one_step(G, model, params.birth)
        i,er,et,sr,st,r = get_infection_stats(G)
#         draw_network_to_file(G,pos,stop) uncomment to save network stills to a file.
    
        stop += 1
    print('done.')
    return i_results,er_results,et_results,sr_results,st_results,r_results, stop #return our results for plotting



def plot_infection(I, Er, Et, Sr, St, R, G, params, fileName):
    """
    :param I: time-ordered list from simulation output indicating how ignorant count changes over time
    :param E: time-ordered list from simulation output indicating how exposed count changes over time
    :param S: time-ordered list from simulation output indicating how spreader count changes over time
    :param R: time-ordered list from simulation output indicating how removed count changes over time
    :param G: Graph/Network of statistic to plot
   
    Creates a plot of the I,E,S,R output of a spread simulation.
    """

    fig_size= [11,7]
    plt.rcParams.update({'font.size': 13, "figure.figsize": fig_size})
    xvalues = range(len(I))
    plt.plot(xvalues, I, color='deepskyblue', linestyle=':', label="I")
    plt.plot(xvalues, Er, color='b', linewidth=0.9, marker='o', label="E rumor")
    plt.plot(xvalues, Et, color='b', linestyle='--', label="E truth")
    plt.plot(xvalues, Sr, color='r', linewidth=0.4, marker='*', label="S rumor")
    plt.plot(xvalues, St, color='r', linewidth=0.7,linestyle='-', label="S truth")
    plt.plot(xvalues, R, color='coral', linestyle='-.', label="R")
    
    plt.legend(loc=1)
    plt.xlabel('Time (days)')
    plt.ylabel('Density Evolution')
    plt.margins(x=0,y=0)
    plt.yticks(np.arange(0, 1.2, 0.2))
    if params.modeltype == 'ES':
        plt.title(params.modeltype + ', N = 1000' + '\n' +
                    "α_t = " + str(params.alpha_t) +
                    ', α_r = ' + str(params.alpha_r) +
                    ', β_t = ' + str(params.beta_t) +
                    ', β_r = ' + str(params.beta_r) +
                    ', γ_t = ' + str(params.gamma_t) +
                    ', γ_r = ' + str(params.gamma_r) + '\n' +
                    'ε_t = ' + str(params.epsilon_t) +
                    ', ε_r = ' + str(params.epsilon_r) + 
                    ', Λ = ' + str(params.birth) +
                    ', μ = ' + str(params.death))
    elif params.modeltype =='SS':
        plt.title(params.modeltype + ', N = ' + str(G.order()) + '\n' +
                    "α_t = " + str(params.alpha_t) +
                    ', α_r = ' + str(params.alpha_r) +
                    ', β_t = ' + str(params.beta_t) +
                    ', β_r = ' + str(params.beta_r) +
                    ', γ_t = ' + str(params.gamma_t) +
                    ', γ_r = ' + str(params.gamma_r) + '\n' +
                    'δ_t = ' + str(params.delta_t) +
                    ', δ_r = ' + str(params.delta_r) +
                    ', Λ = ' + str(params.birth) +
                    ', μ = ' + str(params.death))
#    plt.show()
    plt.savefig("../outputs/"+str(fileName)+".png")
    plt.clf()
    
    
    
def draw_network_to_file(G,pos,t):
    """
    :param G: Graph to draw to png file
    :param pos: position defining how to layout graph
    :param t: current timestep of simualtion (used for filename distinction)
    :param initially_infected: list of initially infected nodes
   
    Draws the current state of the graph G, colouring nodes depending on their state.
    The image is saved to a png file in the images subdirectory.
    """
    # create the layout
    color_map = []
    for node in G:
            if G.nodes[node]['state'] == State.Ignorant:
                color_map.append('paleturquoise')
            elif G.nodes[node]['state'] == State.Exposed_r:
                color_map.append('blue')
            elif G.nodes[node]['state'] == State.Exposed_t:
                color_map.append('dodgerblue')
            elif G.nodes[node]['state'] == State.Spreader_r:
                color_map.append('crimson')
            elif G.nodes[node]['state'] == State.Spreader_t:
                color_map.append('salmon')
            elif G.nodes[node]['state'] == State.Removed:
                color_map.append('green')

    # draw the nodes and the edges (all)
    nx.draw(G,pos,node_color=color_map)
    # plt.annotate("alpha: " + str(ALPHA), xy=(-2,2.2), color="k")
    # plt.annotate("beta: " + str(BETA), xy=(-1.0,2.2), color="k")
    # plt.annotate("gamma: " + str(ALPHA), xy=(0.0,2.2), color="k")
    # plt.annotate("mu: " + str(MU), xy=(0.5,2.2), color="k")
    # plt.annotate("network size: " + str(n), xy=(-2.0,2), color="k")
    plt.savefig("video_of_spread/image"+str(t)+".png")
    plt.clf()

def saveResultsToFile(I, Er, Et, Sr, St, R, G, params, graphType, fileName):
    resultsFile = open('../outputs/' + str(graphType) + '_' + str(fileName) + '.json', 'w+')
    results = []
    results.append({
        "I": I,
        "Er": Er,
        "Et": Et,
        "Sr": Sr,
        "St": St,
        "R": R,
        "G": G,
        "params": params
    })
    json.dump(results, resultsFile)

N = 1000 #number of nodes

# uncomment the type of network to run simulation on
#nw = nx.barabasi_albert_graph(N, 3)
nw = nx.complete_graph(N)
#nw = nx.watts_strogatz_graph(N,2,0.4)
#nw = nx.erdos_renyi_graph(N, 0.6)

reset(nw) #initialize all nodes to ignorant

'''
Order of parameters: 
SS parameters(Lambda, mu,
            alpha+, beta+, gamma+
            alpha-, beta-, gamma-
            delta+, delta-
            S+_0, S-_0, E+_0, E_0)
            
ES parameters(Lambda, mu,
            alpha+, beta+, gamma+
            alpha-, beta-, gamma-
            epsilon+, epsilon-
            S+_0, S-_0, E+_0, E_0)'''

#uncomment the type of simulation to run, either SS or ES

params = SS_Params(4,0.15,
                   0.6,0.4,0.15,
                   0.6,0.4,0.15,
                   0.2,0.2,
                   50, 50, 50, 50)
m = spread_model_factory_SS(params)
#params = ES_Params(4,0.15,
#                  0.6,0.4,0.15,
#                  0.6,0.4,0.15,
#                  0.2,0.2,
#                  50, 50, 50, 50)
#m = spread_model_factory_ES(params)

#uncomment the method of initializing exposed, and spreaders
# initialise_rumor_random(nw, params.sr_init, params.st_init, params.er_init, params.et_init) #random selection of spreaders and exposed
initialise_truth_degree(nw, params.sr_init, params.st_init, params.er_init, params.et_init) #highest degree as spreaders of truth
# initialise_rumor_degree(nw, params.sr_init, params.st_init, params.er_init, params.et_init) #highest degree as spreaders of rumor
# initialise_truth_betweenness(nw, params.sr_init, params.st_init, params.er_init, params.et_init) #highest betweenness centrality as spreaders of truth
# initialise_rumor_betweenness(nw, params.sr_init, params.st_init, params.er_init, params.et_init) #highest betweenness centrality as spreaders of rumor

#run simulation and show graph
I, Er, Et, Sr, St, R, endtime = run_spread_simulation(nw, m, params)
plot_infection(I, Er, Et, Sr, St, R, nw, params, "SS_Truth_Degree")
saveResultsToFile(I, Er, Et, Sr, St, R, nw, params, "completeG", "SS_Truth_Degree")