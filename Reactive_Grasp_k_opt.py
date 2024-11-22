## Juan Fernando Riascos Goyes
## GRASP reactive Heuristic Method for VRPTW problem 
## Libraries 
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import Workbook
import random
import time 
import copy
from scipy.sparse.csgraph import minimum_spanning_tree
from Lecture import Nodo,save_to_excel,plot_routes,read_txt_file
from Feasibility_and_LB import lower_bound_mst,lower_bound_routes,is_feasible
random.seed(2)  


## Time of travel (Define by Euclidean Distance)
## Function given by teacher at [1]

def euclidean_distance(node1, node2):
    return round(math.sqrt((node1.x_cord - node2.x_cord) ** 2 + (node1.y_cord - node2.y_cord) ** 2), 3)

## Function to calculate time travel (t_(i,j))
## Function given by teacher at [1]
def calculate_travel_times(nodes):
    n = len(nodes)
    times = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            times[i][j] = euclidean_distance(nodes[i], nodes[j])
    return times
## FUNCTION FROM CONSTRUCTIVE METHOD 
## Calculate the route distance for a route in 
def calculate_route_distance(route, times):
    distance = 0.0
    for i in range(len(route) - 1):
        distance += times[route[i].index][route[i + 1].index]
    return distance

## Sum of the distances calculated above 
def calculate_total_distance(routes, times):
    return sum(calculate_route_distance(route, times) for route in routes)

## Math model for reactive grasp
def reactive_grasp_route_selection(nodes, capacity, times, alphas=[0.03, 0.05, 0.10, 0.11, 0.12], iterations=100):
    alpha_probs = {alpha: 1/len(alphas) for alpha in alphas}  # Probability of \alpha
    best_routes = None
    best_distance = float('inf')
    min_prob = 1e-6  # Umbral mínimo para probabilidades

    for _ in range(iterations):
        # \alpha selection from alpha probs
        alpha = random.choices(list(alpha_probs.keys()), weights=alpha_probs.values())[0]
        depot = nodes[0]
        customers = nodes[1:]
        routes = []   
        while customers:
            route = [depot]
            current_load = 0
            while True:
                feasible_customers = [cust for cust in customers if is_feasible(route, cust, capacity, times)]
                if not feasible_customers:
                    break
                # 
                feasible_customers.sort(key=lambda x: times[route[-1].index][x.index])
                # RCL LIST
                rcl_size = max(1, int(len(feasible_customers) * alpha))
                rcl = feasible_customers[:rcl_size]
                # Select customer on RCL
                next_customer = random.choice(rcl)
                if current_load + next_customer.demand <= capacity:
                    route.append(next_customer)
                    current_load += next_customer.demand
                    customers.remove(next_customer)
                else:
                    break
            route.append(depot)
            routes.append(route)

        # TOTAL DISTANCE
        total_distance = calculate_total_distance(routes, times)
        if total_distance < best_distance:
            best_distance = total_distance
            best_routes = routes

        # Check probs 
        for alpha_key in alpha_probs:
            if alpha_key == alpha:
                alpha_probs[alpha_key] += 1 / (1 + total_distance - best_distance)
            else:
                alpha_probs[alpha_key] = max(min_prob, alpha_probs[alpha_key] - 1 / (1 + total_distance - best_distance))
        
        # Normal
        total_prob = sum(alpha_probs.values())
        if total_prob == 0 or total_prob != total_prob:  
            alpha_probs = {alpha: 1/len(alphas) for alpha in alphas}  
        else:
            alpha_probs = {k: v / total_prob for k, v in alpha_probs.items()}

    return best_routes, best_distance

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

def vrptw_solver(directory_path, output_filename):
    wb = Workbook()
    wb.remove(wb.active)

    results = []
    total_computation_time = 0
    output_folder = "Grasp_reactive_kopt_images"
    for i in range(1, 19):  # Ajustar el rango de acuerdo a tus archivos
        filename = f'VRPTW{i}.txt'
        file_path = os.path.join(directory_path, filename)
        
        if os.path.exists(file_path): 
            file_path = os.path.join(directory_path, filename)
            n, Q, nodes = read_txt_file(file_path)
            
            times = calculate_travel_times(nodes)
            
            # Calcular las cotas inferiores (lower bounds)
            depot = nodes[0]
            customers = nodes[1:]
            lb_routes = lower_bound_routes(customers, Q)  # Cota inferior para las rutas
            lb_distance = lower_bound_mst(depot, customers, times)  # Usar MST para la cota inferior de la distancia
            
            # Medir el tiempo de cómputo para cada archivo
            start_time = time.time()
            routes, best_distance = reactive_grasp_route_selection(nodes, Q, times)

           # Aplicar 2-opt a cada ruta obtenida del GRASP
            optimized_routes = []
            for route in routes:
                opt_route, _ = opt_k(route, times, Q)  # Optimizar cada ruta con 2-opt
                optimized_routes.append(opt_route)

            # Calcular la nueva distancia después de aplicar 2-opt
            optimized_total_distance = calculate_total_distance(optimized_routes, times)

            computation_time = time.time() - start_time
            total_computation_time += computation_time

            # Calcular el GAP para número de rutas y distancia total
            actual_routes = len(optimized_routes)
            gap_routes = max(((actual_routes - lb_routes) / lb_routes) * 100 if lb_routes > 0 else 0, 0)
            gap_distance = max(((optimized_total_distance - lb_distance) / lb_distance) * 100 if lb_distance > 0 else 0, 0)

            # Mostrar detalles de la solución optimizada
            print(f"Solution for {filename}:")
            print(f"  - Optimized Total Distance = {optimized_total_distance}")
            print(f"  - Lower Bound Distance (MST) = {lb_distance:.2f}")
            print(f"  - GAP Distance = {gap_distance:.2f}")
            print(f"  - Actual Routes = {actual_routes}")
            print(f"  - Lower Bound Routes = {lb_routes}")
            print(f"  - GAP Routes = {gap_routes:.2f}")
            print(f"  - Execution Time = {computation_time * 1000:.0f} ms\n")

            # Guardar los resultados en una hoja de Excel con el nombre de la instancia
            sheet_name = f'VRPTW{i}'
            save_to_excel(wb, sheet_name, optimized_routes, optimized_total_distance, computation_time, times)

            # Guardar la imagen de las rutas
            plot_routes(optimized_routes, filename,output_folder)

        else:
            print(f"Archivo {filename} no encontrado.")


    # Imprimir el tiempo total de cómputo al final en milisegundos
    total_computation_time_ms = total_computation_time * 1000
    print(f"Total computation time for all files: {total_computation_time_ms:.4f} ms")
    
    wb.save(output_filename)
    
    # Ordenar los resultados por nombre de archivo para asegurarse que estén en orden
    results.sort(key=lambda x: int(x[0].split('VRPTW')[1].split('.txt')[0]))

# Ejemplo de uso
directory_path = "./Examples"
output_filename = "VRPTW_JuanFernando_Reactive_GRASP_kopt.xlsx"
vrptw_solver(directory_path, output_filename)
