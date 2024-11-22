"""
Algoritmo Constructivo Problema Vehicle
Routing Problem with Time Windows – VRPTW
Feasibility and Lower boundages
Juan Fernando Riascos Goyes
"""
import math
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

def is_feasible(route, new_node, capacity, times):
    # Check capacity feasibility
    total_demand = sum(node.demand for node in route) + new_node.demand
    if total_demand > capacity:  # Use '>' to strictly enforce the capacity constraint
        return False

  
    current_time = 0
    for i in range(1, len(route)):
       
        current_time += times[route[i-1].index][route[i].index]
        
      
        if current_time < route[i].time_window[0]:
            current_time = route[i].time_window[0]
        
        if current_time > route[i].time_window[1]:
            return False
        
        current_time += route[i].serv_time

    new_node_arrival_time = current_time + times[route[-1].index][new_node.index]

    
    if new_node_arrival_time < new_node.time_window[0]:
        new_node_arrival_time = new_node.time_window[0]
    if new_node_arrival_time > new_node.time_window[1]:
        return False  # Arrival too late for the new node

    return True  

def lower_bound_routes(customers, vehicle_capacity):
    total_demand = sum(customer.demand for customer in customers)
    return math.ceil(total_demand / vehicle_capacity)

def lower_bound_mst(depot, customers, distance_matrix):
    nodes = [depot] + customers
    n = len(nodes)

    # Create a full distance matrix for all nodes
    full_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            full_matrix[i, j] = distance_matrix[nodes[i].index][nodes[j].index]

    mst = minimum_spanning_tree(full_matrix).toarray()

    mst_distance = mst.sum()

    return mst_distance



