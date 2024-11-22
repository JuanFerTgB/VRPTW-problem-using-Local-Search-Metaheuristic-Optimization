## Juan Fernando Riascos Goyes
## ACO Heuristic Method for VRPTW problem 
## Libraries 
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from openpyxl import Workbook
from scipy.sparse.csgraph import minimum_spanning_tree
from Lecture import Nodo,save_to_excel,plot_routes,read_txt_file
from Feasibility_and_LB import lower_bound_mst,lower_bound_routes,is_feasible
import random
random.seed(2)  

def dist(node1, node2):
    return math.sqrt((node1.x_cord - node2.x_cord) ** 2 + (node1.y_cord - node2.y_cord) ** 2)

def calculate_travel_times(nodes):
    n = len(nodes)
    times = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            times[i][j] = dist(nodes[i], nodes[j])
    return times
 
def initialize_pheromones(num_nodes, times):
    
    pheromones = np.ones((num_nodes, num_nodes)) / (times + 1e-6)  # Feromonas inversamente proporcionales a la distancia
    return pheromones
def calculate_route_distance(route, times):
    distance = 0.0
    for i in range(len(route) - 1):
        distance += times[route[i].index][route[i + 1].index]
    return distance


def travel_times_matrix(nodes):
    n = len(nodes)
    travel_times = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            travel_times[i][j] = dist(nodes[i], nodes[j])
    return travel_times

def update_pheromones(pheromones, all_routes, Q, rho):
    """
    Actualiza las feromonas con evaporación y deposición.
    """
    pheromones *= (1 - rho)  # Evaporación
    for routes, distance in all_routes:
        for route in routes:
            for i in range(len(route) - 1):
                pheromones[route[i].index][route[i + 1].index] += Q / distance  

def calculate_total_distance(routes, times):
    return sum(calculate_route_distance(route, times) for route in routes)

# Parameters for ACO
aco_params = {
    'num_ants':50,
    'num_iterations': 100,
    'alpha': 1.5,
    'beta': 2,
    'rho': 0.7,
    'Q': 10.0
}

import copy

def opt_k(route, times, capacity, k=3):
    """
    Aplica k-opt utilizando la estrategia de mejor mejora para una ruta.
    
    route: Lista de nodos que conforman la ruta.
    times: Matriz de tiempos de viaje entre nodos.
    capacity: Capacidad máxima del vehículo.
    k: Número de intercambios a realizar en la ruta.
    
    Retorna:
        best_route: La mejor ruta encontrada después de aplicar k-opt.
        best_distance: La distancia total de la mejor ruta.
    """
    best_route = copy.deepcopy(route)
    best_distance = calculate_route_distance(best_route, times)
    improved = True

    while improved:
        improved = False
        best_local_route = None  # Aquí no se inicializa la mejor ruta local hasta encontrar una mejor
        best_local_distance = best_distance

        # Exploramos todas las combinaciones posibles de k-opt
        for i in range(1, len(route) - k):
            for j in range(i + 1, len(route) - (k - 1)):
                for m in range(j + 1, len(route) - (k - 2)):
                    # Generar una nueva ruta eliminando y reconectando k aristas
                    new_route = (
                        best_route[:i] +
                        best_route[i:j+1][::-1] +
                        best_route[j:m+1][::-1] +
                        best_route[m+1:]
                    )

                    # Verificar factibilidad de la nueva ruta
                    if is_feasible(new_route, new_route[-1], capacity, times):
                        new_distance = calculate_route_distance(new_route, times)

                        # Si la nueva distancia es mejor, guardamos la mejor solución
                        if new_distance < best_local_distance:
                            best_local_route = copy.deepcopy(new_route)
                            best_local_distance = new_distance

        # Si encontramos una mejora, actualizamos la ruta y continuamos buscando
        if best_local_distance < best_distance:
            best_route = best_local_route
            best_distance = best_local_distance
            improved = True  # Solo continuamos si encontramos una mejora

    return best_route, best_distance





def aco_vrptw(nodes, capacity, times, num_ants, num_iterations, alpha, beta, rho, Q):

    num_nodes = len(nodes)
    pheromones = initialize_pheromones(num_nodes, times)
    best_routes = None
    best_distance = float('inf')

    for iteration in range(num_iterations):
        all_routes = []
        for ant in range(num_ants):
            depot = nodes[0]
            customers = set(range(1, num_nodes))  # Índices de clientes
            routes = []

            while customers:
                route = [depot]
                current_load = 0

                while True:
                    # Clientes factibles para la hormiga
                    feasible_customers = [cust for cust in customers if is_feasible(route, nodes[cust], capacity, times)]
                    if not feasible_customers:
                        break

                    # Cálculo de probabilidades basado en feromonas y visibilidad
                    probabilities = []
                    for cust in feasible_customers:
                        pheromone = pheromones[route[-1].index][cust]
                        travel_time = times[route[-1].index][cust]
                        visibility = 1 / (travel_time if travel_time > 0 else 1e-6)
                        probabilities.append((pheromone ** alpha) * (visibility ** beta))

                    total_prob = sum(probabilities)
                    probabilities = np.array(probabilities) / total_prob if total_prob > 0 else np.ones(len(feasible_customers)) / len(feasible_customers)

                    # Seleccionar el próximo cliente basado en probabilidades
                    next_customer_index = np.random.choice(feasible_customers, p=probabilities)
                    next_customer = nodes[next_customer_index]

                    if current_load + next_customer.demand <= capacity:
                        route.append(next_customer)
                        current_load += next_customer.demand
                        customers.remove(next_customer_index)
                    else:
                        break

                route.append(depot)  # El vehículo regresa al depósito

                # Apply 3-opt local search to improve the route
                
                optimized_route, _ = opt_k(route, times, capacity)

                routes.append(optimized_route)

            total_distance = calculate_total_distance(routes, times)
            all_routes.append((routes, total_distance))

            # Actualizar la mejor solución si es mejor que la actual
            if total_distance < best_distance - 1e-6:
                best_distance = total_distance
                best_routes = routes

        # Actualizar feromonas
        update_pheromones(pheromones, all_routes, Q, rho)

    return best_routes, best_distance

def vrptw_solver(directory_path, output_filename):

    wb = Workbook()
    wb.remove(wb.active)

    execution_times = []  # Lista para guardar tiempos de ejecución
    output_folder = "ACO_images_kopt"  # Carpeta donde se guardarán las imágenes

    for i in range(1, 19):  # Procesar archivos VRPTW1 a VRPTW18
        filename = f'{directory_path}/VRPTW{i}.txt'
        file_start_time = time.time()  # Tiempo de inicio

        # Leer nodos y calcular la matriz de tiempos
        n, Q, nodes = read_txt_file(filename)
        times = travel_times_matrix(nodes)

        # Calcular las cotas inferiores (lower bounds)
        depot = nodes[0]
        customers = nodes[1:]
        lb_routes = lower_bound_routes(customers, Q)
        lb_distance = lower_bound_mst(depot, customers, times)  # Usar MST para la cota inferior

        # Aplicar ACO con k-opt
        routes, best_distance = aco_vrptw(nodes, Q, times, **aco_params)
        computation_time = (time.time() - file_start_time) * 1000  # Tiempo en milisegundos
        execution_times.append(computation_time)

        # Calcular el GAP para número de rutas y distancia total
        actual_routes = len(routes)
        gap_routes = max(((actual_routes - lb_routes) / lb_routes) * 100 if lb_routes > 0 else 0, 0)
        gap_distance = max(((best_distance - lb_distance) / lb_distance) * 100 if lb_distance > 0 else 0, 0)

        # Mostrar detalles de la solución
        base_filename = os.path.basename(filename)  # Extrae solo el nombre del archivo
        print(f"Solution for {base_filename}:")
        print(f"  - Total Distance = {best_distance}")
        print(f"  - Lower Bound Distance (MST) = {lb_distance:.2f}")
        print(f"  - GAP Distance = {gap_distance:.2f}")
        print(f"  - Actual Routes = {actual_routes}")
        print(f"  - Lower Bound Routes = {lb_routes}")
        print(f"  - GAP Routes = {gap_routes:.2f}")
        print(f"  - Execution Time = {computation_time:.0f} ms\n")

        # Guardar resultados en Excel
        sheet_name = f'VRPTW{i}'
        save_to_excel(wb, sheet_name, routes, best_distance, computation_time, times)

        # Llamar a plot_routes para graficar y guardar las rutas en la carpeta ACO_images
        plot_routes(routes, f"VRPTW{i}.txt", output_folder)

    # Guardar el archivo Excel
    wb.save(output_filename)

    total_elapsed_time = sum(execution_times)  # Tiempo total en milisegundos
    print(f"\nTotal execution time: {total_elapsed_time:.0f} ms")

# Ejemplo de uso
directory_path = "./Examples"  # Carpeta con los archivos de entrada
output_filename = "VRPTW_JuanFernando_ACO_k_opt.xlsx"
vrptw_solver(directory_path, output_filename)
