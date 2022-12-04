############################################################
# Li Xi
# lxi@bu.edu
# Starter code for Bellman-Ford modification assignment
# November 2022
############################################################


import sys
import os
import heapq
import queue
import numpy as np
import simplegraphs as sg


############################################################
#
# HELPFUL CODE
#
############################################################


def bellmanFordSimple(G, s):
    # G is a dictionary with keys "n", "m", "adj" representing an *weighted* graph
    # G["adj"][u][v] is the cost (length / weight) of edge (u,v)
    # This algorithms finds least-costs paths to all vertices
    # Will not detect negative-cost cycles
    # Returns an dict of distances (path costs) and parents in the lightest-paths tree.
    #
    # This is basically the algorithm we covered in class (except it
    # finds paths from a source instead of to a desitnation).
    #
    n = G["n"]
    d = [{} for i in range(n + 1)]
    for u in G["adj"]:
        d[0][u] = np.inf
    d[0][s] = 0
    parent = {s: None}
    for i in range(1, n+1):
        changed = False
        for v in G["adj"]:
            d[i][v] = d[i-1][v]
        for u in G["adj"]:
            for v in G["adj"][u]:
                newlength = d[i-1][u] + G["adj"][u][v]
                if newlength < d[i][v]:
                    d[i][v] = newlength
                    parent[v] = u
                    changed = True
        # How can you decide whether it is ok to stop?
    if changed:
        print("Negative cycle reachable from source!")
    # If there are no negative-cost cycles, these are the correct distances.
    distances = d[n-1]
    return distances, parent


############################################################
#
# CODE FOR ASSIGNMENT
#
############################################################


def findClosestNodes(G):
    # G is a dictionary with keys "n", "m", "adj" representing an *weighted* graph
    # G["adj"][u][v] is the cost (length / weight) of edge (u,v)
    #

    #############################
    # Your code here
    # distances_to_closest[u] should be the length of the
    # lightest path of length at least 1 starting from u.
    # n = G["n"]
    # print("n:", G["n"])  # 3
    # print("m:", G["m"])  # 3
    # print("adj", G["adj"])  # {0: {1: 2}, 1: {2: 2}, 2: {0: -3}}
    # print(G["adj"][0])  # {1: 2}
    # print(G["adj"][1])  # {2: 2}
    # print(G["adj"][2])  # {0: -3}
    # print(G["adj"][0][1])  # edge length btn 0 and 1 -> 2
    # print(G["adj"][1][2])  # edge length btn 1 and 2 -> 2
    # print(G["adj"][2][0])  # edge length btn 2 and 3 -> -3
    distances_to_closest = {}
    # a is like the n+1 parameter in the bellmanFord algo.
    a = 40
    n = G["n"]
    d = [{} for i in range(a)]
    # print(d)
    for i in range(a):
        for u in G["adj"]:
            # initialize current best path to infinite large.
            d[i][u] = np.inf
            # print(d)
    for i in range(1, a):
        for u in G["adj"]:
            for v in G["adj"][u]:
                # compare the current best path length of u and u-v path length, choose the shorter one
                d[i][u] = min(d[i][u], G["adj"][u][v])
                # compare the current best path of u and u-v path + best path with 1 edge less to v.
                d[i][u] = min(d[i][u], d[i-1][v] + G["adj"][u][v])

    # If there are no negative-cost cycles, these are the correct distances.
    distances_to_closest = d[a-2]

    #############################
    return distances_to_closest


############################################################
#
# The remaining functions are for reading and writing outputs, and processing
# the command line arguments. You shouldn't have to modify them (but
# you can, for testing, if you want).
#
############################################################

def writeBFOutput(distances, parent, out_file_name):
    # Assumes parents and distances have the same keys.
    # Prints a file with one line per node, of the form "node, distance, parent"
    with open(out_file_name, 'w') as f:
        for u in distances:
            f.write("{}, {}, {}\n".format(u, distances[u], parent[u]))
    return


def writeCNDistances(distances_to_closest, out_file_name):
    # Prints a file with one line per node, of the form "node, distance to closest node"
    with open(out_file_name, 'w') as f:
        L = list(distances_to_closest.keys())
        L.sort()
        for u in L:
            f.write("{}, {}\n".format(u, distances_to_closest[u]))
    return


def main(args=[]):
    # Expects three to five command-line arguments:
    # 0) Tasks: either "shortestPaths" or "closestNodes"
    # 1) name of a file describing the graph
    # 2) name of a file where the output should be written
    # 3) For shortest paths, the source node s.
    if len(args) < 3:
        print("Too few arguments! There should be at least 3.")
        return
    task = args[0]
    graph_file_name = args[1]
    out_file_name = args[2]
    if task == "shortestPaths":
        if len(args) != 4:
            print("Problem! There were {} arguments instead of 4 for Shortest Paths.".format(
                len(args)))
            return
        G = sg.readGraph(graph_file_name)  # Read the graph from disk
        s = int(args[3])
        distances, parent = bellmanFordSimple(G, s)
        writeBFOutput(distances, parent, out_file_name)
    elif task == "closestNodes":
        if len(args) != 3:
            print("Problem! There were {} arguments instead of 3 for Shortest Paths.".format(
                len(args)))
        G = sg.readGraph(graph_file_name)  # Read the graph from disk
        distances_to_closest = findClosestNodes(G)
        # We ignore the names of closest nodes and the first step
        writeCNDistances(distances_to_closest, out_file_name)
    else:
        print("Problem! Task {} not recognized".format(task))
    return


if __name__ == "__main__":
    main(sys.argv[1:])
