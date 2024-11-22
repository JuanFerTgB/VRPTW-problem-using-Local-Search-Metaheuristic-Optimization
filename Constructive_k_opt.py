## Juan Fernando Riascos Goyes
## Constructive Heuristic Method for VRPTW problem

## Libraries 
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import Workbook
import copy
import time
from scipy.sparse.csgraph import minimum_spanning_tree
from Lecture import Nodo, save_to_excel, plot_routes, read_txt_file
from Feasibility_and_LB import lower_bound_routes, is_feasible, lower_bound_mst
import itertools

## Time of travel (Define by Euclidean Distance)
def euclidean_distance(node1, node2):
    return math.sqrt((node1.x_cord - node2.x_cord) ** 2 + (node1.y_cord - node2.y_cord) ** 2)

## Function to calculate time travel (t_(i,j))
def calculate_travel_times(nodes):
    n = len(nodes)
    times = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            times[i][j] = euclidean_distance(nodes[i], nodes[j])
    return times

## Calculate the route distance for a route in 
def calculate_route_distance(route, times):
    distance = 0.0
    for i in range(len(route) - 1):
        distance += times[route[i].index][route[i + 1].index]
    return distance

## Sum of the distances calculated above 
def calculate_total_distance(routes, times):
    return sum(calculate_route_distance(route, times) for route in routes)



def route_selection(nodes, capacity, times):
    """
    Método constructivo que selecciona rutas factibles y luego aplica k-opt
    con una implementación estricta para optimizarlas.
    
    nodes: Lista de nodos (incluyendo el depósito y los clientes).
    capacity: Capacidad del vehículo.
    times: Matriz de tiempos de viaje entre nodos.
    
    Retorna:
        routes: Lista de rutas optimizadas con k-opt.
    """
    depot = nodes[0]
    customers = nodes[1:]
    routes = []

    # Paso constructivo: construir rutas factibles
    while customers:
        route = [depot]
        current_load = 0
        while True:
            feasible_customers = [cust for cust in customers if is_feasible(route, cust, capacity, times)]
            if not feasible_customers:
                break
            # Selección del cliente más cercano (heurística greedy)
            next_customer = min(feasible_customers, key=lambda x: times[route[-1].index][x.index])
            if current_load + next_customer.demand <= capacity:
                route.append(next_customer)
                current_load += next_customer.demand
                customers.remove(next_customer)
            else:
                break
        route.append(depot)  # Regresar al depósito
        routes.append(route)

    # Optimización con k-opt (estricta)
    optimized_routes = []
    for route in routes:
        optimized_route, _ = opt_k(route, times, capacity, k=3)  # Aplicar k-opt estricto
        optimized_routes.append(optimized_route)

    return optimized_routes


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




def vrptw_solver(output_filename):
    # Obtener la ruta del directorio donde se ejecuta el código
    directory_path = os.getcwd()  # Obtiene el directorio actual
    examples_path = os.path.join(directory_path, "Examples")  # Asume que hay una carpeta "Examples" en el directorio actual

    wb = Workbook()
    total_computation_time = 0  # Para calcular el tiempo total de ejecución

    output_folder = "constructive_images_kopt"  # Carpeta donde se guardarán las imágenes

    # Recorrer los archivos numerados en orden
    for i in range(1, 19):  # Ajustar el rango de acuerdo a tus archivos
        filename = f'VRPTW{i}.txt'
        file_path = os.path.join(examples_path, filename)
        
        if os.path.exists(file_path):  # Verifica que el archivo exista
            n, Q, nodes = read_txt_file(file_path)
            times = calculate_travel_times(nodes)

            # Calcular las cotas inferiores (lower bounds)
            depot = nodes[0]
            customers = nodes[1:]
            lb_routes = lower_bound_routes(customers, Q)
            lb_distance = lower_bound_mst(depot, customers, times)  # Usar MST para la cota inferior
           
            # Medir el tiempo de cómputo para cada archivo
            start_time = time.time()
            routes = route_selection(nodes, Q, times)  # Aplicar 2-opt en route_selection
            computation_time = time.time() - start_time
            total_computation_time += computation_time

            total_distance = calculate_total_distance(routes, times)

            # Calcular el GAP para número de rutas y distancia total
            actual_routes = len(routes)
            gap_routes = max(((actual_routes - lb_routes) / lb_routes) * 100 if lb_routes > 0 else 0, 0)
            gap_distance = max(((total_distance - lb_distance) / lb_distance) * 100 if lb_distance > 0 else 0, 0)

            # Mostrar detalles de la solución
            print(f"Solution for {filename}:")
            print(f"  - Total Distance = {total_distance}")
            print(f"  - Lower Bound Distance (MST) = {lb_distance:.2f}")
            print(f"  - GAP Distance = {gap_distance:.2f}")
            print(f"  - Actual Routes = {actual_routes}")
            print(f"  - Lower Bound Routes = {lb_routes}")
            print(f"  - GAP Routes = {gap_routes:.2f}")
            print(f"  - Execution Time = {computation_time * 1000:.0f} ms\n")

            # Guardar los resultados en una hoja de Excel con el nombre de la instancia
            sheet_name = f'VRPTW{i}'
            save_to_excel(wb, sheet_name, routes, total_distance, computation_time, times)

            plot_routes(routes, filename, output_folder)  # Pass output folder to save plots
        else:
            print(f"Archivo {filename} no encontrado.")
    
    # Guardar el archivo de Excel con todas las hojas
    wb.save(output_filename)
    
    # Imprimir el tiempo total de cómputo al final en milisegundos
    total_computation_time_ms = total_computation_time * 1000
    print(f"Total computation time for all files: {total_computation_time_ms:.4f} ms")

# Ejemplo de uso
output_filename = "VRPTW_JuanFernando_Constructivo_kopt.xlsx"
vrptw_solver(output_filename)
