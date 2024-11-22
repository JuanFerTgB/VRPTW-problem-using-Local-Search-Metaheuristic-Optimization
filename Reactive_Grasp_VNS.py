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
from scipy.sparse.csgraph import minimum_spanning_tree
from Lecture import Nodo,save_to_excel,plot_routes,read_txt_file
from Feasibility_and_LB import lower_bound_mst,lower_bound_routes,is_feasible
import copy
random.seed(20)  


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
    alpha_probs = {alpha: 1/len(alphas) for alpha in alphas}  # Probabilidades iniciales
    best_routes = None
    best_distance = float('inf')
    min_prob = 1e-6  # Umbral mínimo para probabilidades

    for _ in range(iterations):
        # Selección de alfa basada en las probabilidades
        alpha = random.choices(list(alpha_probs.keys()), weights=alpha_probs.values())[0]
        depot = nodes[0]
        customers = nodes[1:]
        routes = []
        remaining_customers = customers.copy()
        
        while remaining_customers:
            route = [depot]
            current_load = 0
            while True:
                feasible_customers = [cust for cust in remaining_customers if is_feasible(route, cust, capacity, times)]
                if not feasible_customers:
                    break
                # Ordenar los clientes factibles según la distancia
                feasible_customers.sort(key=lambda x: times[route[-1].index][x.index])
                # Generar RCL
                rcl_size = max(1, int(len(feasible_customers) * alpha))
                rcl = feasible_customers[:rcl_size]
                # Seleccionar un cliente de la RCL
                next_customer = random.choice(rcl)
                if current_load + next_customer.demand <= capacity:
                    route.append(next_customer)
                    current_load += next_customer.demand
                    remaining_customers.remove(next_customer)
                else:
                    break
            route.append(depot)
            routes.append(route)

        # **Aplicar VNS mejorado a todas las rutas**
        optimized_routes = vns_algorithm(routes, times, capacity)
        total_distance = calculate_total_distance(optimized_routes, times)
        if total_distance < best_distance:
            best_distance = total_distance
            best_routes = optimized_routes

        # Actualizar las probabilidades de alfa
        for alpha_key in alpha_probs:
            if alpha_key == alpha:
                alpha_probs[alpha_key] += 1 / (1 + total_distance - best_distance)
            else:
                alpha_probs[alpha_key] = max(min_prob, alpha_probs[alpha_key] - 1 / (1 + total_distance - best_distance))
        
        # Normalizar las probabilidades
        total_prob = sum(alpha_probs.values())
        if total_prob == 0 or total_prob != total_prob:  
            alpha_probs = {alpha: 1/len(alphas) for alpha in alphas}  
        else:
            alpha_probs = {k: v / total_prob for k, v in alpha_probs.items()}

    return best_routes, best_distance



def is_route_feasible(route, capacity, times):
    """
    Verifica si una ruta completa es factible utilizando la función is_feasible.
    """
    # Iniciamos con una ruta vacía que contiene solo el depósito
    feasible_route = [route[0]]  # Suponiendo que el depósito es el primer nodo
    for node in route[1:]:
        if is_feasible(feasible_route, node, capacity, times):
            feasible_route.append(node)
        else:
            return False  # La ruta no es factible al agregar este nodo
    return True


def swap_between_routes_best(routes, times, capacity):
    best_routes = [route.copy() for route in routes]
    best_distance = calculate_total_distance(best_routes, times)
    improved = False

    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            route1 = routes[i]
            route2 = routes[j]

            customers1 = route1[1:-1]
            customers2 = route2[1:-1]

            for idx1, cust1 in enumerate(customers1):
                for idx2, cust2 in enumerate(customers2):
                    temp_route1 = route1.copy()
                    temp_route2 = route2.copy()

                    temp_route1[idx1 + 1] = cust2
                    temp_route2[idx2 + 1] = cust1

                    if (is_route_feasible(temp_route1, capacity, times) and
                        is_route_feasible(temp_route2, capacity, times)):
                        temp_routes = routes.copy()
                        temp_routes[i] = temp_route1
                        temp_routes[j] = temp_route2
                        temp_distance = calculate_total_distance(temp_routes, times)
                        if temp_distance + 1e-6 < best_distance:
                            best_routes = temp_routes
                            best_distance = temp_distance
                            improved = True
    return best_routes, best_distance, improved


def relocate_between_routes_best(routes, times, capacity):
    best_routes = [route.copy() for route in routes]
    best_distance = calculate_total_distance(best_routes, times)
    improved = False

    for i in range(len(routes)):
        for j in range(len(routes)):
            if i == j:
                continue

            route_from = routes[i]
            route_to = routes[j]

            customers_from = route_from[1:-1]

            for idx_cust, cust in enumerate(customers_from):
                temp_route_from = route_from.copy()
                del temp_route_from[idx_cust + 1]

                for k in range(1, len(route_to)):
                    temp_route_to = route_to[:k] + [cust] + route_to[k:]

                    if (is_route_feasible(temp_route_from, capacity, times) and
                        is_route_feasible(temp_route_to, capacity, times)):
                        temp_routes = routes.copy()
                        temp_routes[i] = temp_route_from
                        temp_routes[j] = temp_route_to
                        temp_distance = calculate_total_distance(temp_routes, times)
                        if temp_distance + 1e-6 < best_distance:
                            best_routes = temp_routes
                            best_distance = temp_distance
                            improved = True
    return best_routes, best_distance, improved


def two_opt_within_route_single(route, times, capacity):
    best_route = route.copy()
    best_distance = calculate_route_distance(best_route, times)
    n = len(route)
    improved = False

    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
            # Verificar factibilidad utilizando is_feasible
            if is_route_feasible(new_route, capacity, times):
                new_distance = calculate_route_distance(new_route, times)
                if new_distance + 1e-6 < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    improved = True
    return best_route, best_distance

def or_opt_within_route_single(route, times, capacity):
    best_route = route.copy()
    best_distance = calculate_route_distance(best_route, times)
    n = len(route)
    improved = False

    # Tamaño de los segmentos a mover
    for segment_size in range(1, 4):  # Mover segmentos de 1 a 3 clientes
        for i in range(1, n - segment_size - 1):
            segment = route[i:i+segment_size]
            rest_route = route[:i] + route[i+segment_size:]

            for j in range(1, len(rest_route)):
                new_route = rest_route[:j] + segment + rest_route[j:]
                # Verificar factibilidad utilizando is_feasible
                if is_route_feasible(new_route, capacity, times):
                    new_distance = calculate_route_distance(new_route, times)
                    if new_distance + 1e-6 < best_distance:
                        best_route = new_route
                        best_distance = new_distance
                        improved = True
    return best_route, best_distance


def vns_algorithm(routes, times, capacity, max_iterations=50):
  
    best_routes = [route.copy() for route in routes]
    best_distance = calculate_total_distance(best_routes, times)
    iteration = 0

    # Definir las vecindades
    neighborhoods = [swap_between_routes_best, relocate_between_routes_best, two_opt_across_routes]

    while iteration < max_iterations:
        iteration += 1
        improved = False

        for neighborhood in neighborhoods:
            new_routes, new_distance, neighborhood_improved = neighborhood(best_routes, times, capacity)

            # Solo aceptamos soluciones que mejoran la distancia total
            if neighborhood_improved and new_distance + 1e-6 < best_distance:
                best_routes = [route.copy() for route in new_routes]
                best_distance = new_distance
                improved = True
                break  # Reiniciar desde la primera vecindad

        if not improved:
            break  # No se encontraron mejoras, terminar

    return best_routes

def two_opt_across_routes(routes, times, capacity):
    best_routes = [route.copy() for route in routes]
    best_distance = calculate_total_distance(best_routes, times)
    improved = False

    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            route1 = routes[i]
            route2 = routes[j]

            for idx1 in range(1, len(route1) - 1):
                for idx2 in range(1, len(route2) - 1):
                    # Crear nuevas rutas intercambiando segmentos
                    new_route1 = route1[:idx1] + route2[idx2:]
                    new_route2 = route2[:idx2] + route1[idx1:]

                    if (is_route_feasible(new_route1, capacity, times) and
                        is_route_feasible(new_route2, capacity, times)):
                        temp_routes = routes.copy()
                        temp_routes[i] = new_route1
                        temp_routes[j] = new_route2
                        temp_distance = calculate_total_distance(temp_routes, times)
                        if temp_distance + 1e-6 < best_distance:
                            best_routes = temp_routes
                            best_distance = temp_distance
                            improved = True
    return best_routes, best_distance, improved




def vrptw_solver(directory_path, output_filename):
    wb = Workbook()
    wb.remove(wb.active)

    total_computation_time = 0
    output_folder = "Grasp_reactive_VNS_images"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(1, 19):  # Ajustar el rango de acuerdo a tus archivos
        filename = f'VRPTW{i}.txt'
        file_path = os.path.join(directory_path, filename)
        
        if os.path.exists(file_path): 
            n, Q, nodes = read_txt_file(file_path)
            times = calculate_travel_times(nodes)
            
            # Calcular las cotas inferiores (lower bounds)
            depot = nodes[0]
            customers = nodes[1:]
            lb_routes = lower_bound_routes(customers, Q)
            lb_distance = lower_bound_mst(depot, customers, times)
            
            # Medir el tiempo de cómputo para cada archivo
            start_time = time.time()
            routes, best_distance = reactive_grasp_route_selection(nodes, Q, times)

            computation_time = time.time() - start_time
            total_computation_time += computation_time

            # Calcular el GAP para número de rutas y distancia total
            actual_routes = len(routes)
            gap_routes = max(((actual_routes - lb_routes) / lb_routes) * 100 if lb_routes > 0 else 0, 0)
            gap_distance = max(((best_distance - lb_distance) / lb_distance) * 100 if lb_distance > 0 else 0, 0)

            # Mostrar detalles de la solución optimizada
            print(f"Solution for {filename}:")
            print(f"  - Optimized Total Distance = {best_distance}")
            print(f"  - Lower Bound Distance (MST) = {lb_distance:.2f}")
            print(f"  - GAP Distance = {gap_distance:.2f}%")
            print(f"  - Actual Routes = {actual_routes}")
            print(f"  - Lower Bound Routes = {lb_routes}")
            print(f"  - GAP Routes = {gap_routes:.2f}%")
            print(f"  - Execution Time = {computation_time * 1000:.0f} ms\n")

            # Guardar los resultados en una hoja de Excel con el nombre de la instancia
            sheet_name = f'VRPTW{i}'
            save_to_excel(wb, sheet_name, routes, best_distance, computation_time, times)

            # Guardar la imagen de las rutas
            plot_routes(routes, filename, output_folder)

        else:
            print(f"Archivo {filename} no encontrado.")

    # Imprimir el tiempo total de cómputo al final en milisegundos
    total_computation_time_ms = total_computation_time * 1000
    print(f"Total computation time for all files: {total_computation_time_ms:.4f} ms")
    
    wb.save(output_filename)


# Ejemplo de uso
directory_path = "./Examples"
output_filename = "VRPTW_JuanFernando_Reactive_GRASP_VNS.xlsx"
vrptw_solver(directory_path, output_filename)
