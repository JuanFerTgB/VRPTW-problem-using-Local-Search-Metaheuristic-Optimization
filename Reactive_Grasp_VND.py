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

        # Calcular la distancia total sin aplicar VND
        total_distance = calculate_total_distance(routes, times)
        if total_distance < best_distance:
            best_distance = total_distance
            best_routes = routes.copy()

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
            if is_route_feasible(new_route, capacity, times):
                new_distance = calculate_route_distance(new_route, times)
                if new_distance + 1e-6 < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    improved = True
    return best_route, best_distance, improved  # Asegúrate de que se devuelven tres valores



def or_opt_within_route_single(route, times, capacity):
    best_route = route.copy()
    best_distance = calculate_route_distance(best_route, times)
    n = len(route)
    improved = False

    for segment_size in range(1, 4):
        for i in range(1, n - segment_size - 1):
            segment = route[i:i+segment_size]
            rest_route = route[:i] + route[i+segment_size:]

            for j in range(1, len(rest_route)):
                new_route = rest_route[:j] + segment + rest_route[j:]
                if is_route_feasible(new_route, capacity, times):
                    new_distance = calculate_route_distance(new_route, times)
                    if new_distance + 1e-6 < best_distance:
                        best_route = new_route
                        best_distance = new_distance
                        improved = True
    return best_route, best_distance, improved  # Asegúrate de que se devuelven tres valores



def merge_routes(routes, times, capacity):
    """
    Función mejorada para fusionar rutas de manera más agresiva.
    Intenta fusionar rutas considerando todos los pares posibles y las inversiones de rutas.
    """
    improved = True
    while improved:
        improved = False
        num_routes = len(routes)
        best_distance = calculate_total_distance(routes, times)
        best_routes = routes.copy()

        for i in range(num_routes):
            for j in range(num_routes):
                if i >= j:
                    continue

                route1 = routes[i]
                route2 = routes[j]

                # Intentar fusionar route1 y route2 directamente
                merged_route = route1[:-1] + route2[1:]
                if is_route_feasible(merged_route, capacity, times):
                    temp_routes = [routes[k] for k in range(num_routes) if k != i and k != j]
                    temp_routes.append(merged_route)
                    temp_distance = calculate_total_distance(temp_routes, times)
                    if temp_distance + 1e-6 < best_distance or len(temp_routes) < len(best_routes):
                        best_routes = temp_routes
                        best_distance = temp_distance
                        improved = True
                        break

                # Intentar fusionar route1 y la inversión de route2
                reversed_route2 = [route2[0]] + route2[1:-1][::-1] + [route2[-1]]
                merged_route = route1[:-1] + reversed_route2[1:]
                if is_route_feasible(merged_route, capacity, times):
                    temp_routes = [routes[k] for k in range(num_routes) if k != i and k != j]
                    temp_routes.append(merged_route)
                    temp_distance = calculate_total_distance(temp_routes, times)
                    if temp_distance + 1e-6 < best_distance or len(temp_routes) < len(best_routes):
                        best_routes = temp_routes
                        best_distance = temp_distance
                        improved = True
                        break

            if improved:
                routes = best_routes
                break  # Reiniciar búsqueda desde el principio

    return routes





def vnd_algorithm(routes, times, capacity, time_limit, start_time):
    """
    Algoritmo VND con criterio de parada basado en tiempo.
    """
    best_routes = [route.copy() for route in routes]
    best_distance = calculate_total_distance(best_routes, times)
    neighborhoods = [
        two_opt_within_route_single,
        or_opt_within_route_single,
        swap_between_routes_best,
        relocate_between_routes_best,
        two_opt_across_routes,
        merge_routes
    ]

    neighborhood_index = 0

    while neighborhood_index < len(neighborhoods):
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time >= time_limit:
            # Tiempo límite alcanzado, terminar
            break

        neighborhood = neighborhoods[neighborhood_index]
        improved = False

        if neighborhood in [two_opt_within_route_single, or_opt_within_route_single]:
            # Aplicar movimientos dentro de rutas individuales
            for idx, route in enumerate(best_routes):
                new_route, new_distance, route_improved = neighborhood(route, times, capacity)
                if route_improved and new_distance + 1e-6 < calculate_route_distance(best_routes[idx], times):
                    best_routes[idx] = new_route
                    best_distance = calculate_total_distance(best_routes, times)
                    improved = True

                    # Verificar el tiempo después de cada mejora
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    if elapsed_time >= time_limit:
                        break
            if elapsed_time >= time_limit:
                break
        else:
            if neighborhood == merge_routes:
                new_routes = neighborhood(best_routes, times, capacity)
                new_distance = calculate_total_distance(new_routes, times)
                neighborhood_improved = (len(new_routes) < len(best_routes) or new_distance + 1e-6 < best_distance)
            else:
                new_routes, new_distance, neighborhood_improved = neighborhood(best_routes, times, capacity)

            if neighborhood_improved and (new_distance + 1e-6 < best_distance or len(new_routes) < len(best_routes)):
                best_routes = [route.copy() for route in new_routes]
                best_distance = new_distance
                improved = True

                # Verificar el tiempo después de cada mejora
                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time >= time_limit:
                    break

        if improved:
            # Continuar con el mismo vecindario
            neighborhood_index = 0
        else:
            # Pasar al siguiente vecindario
            neighborhood_index += 1

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
    output_folder = "Grasp_reactive_VND_images"  # Cambiamos el nombre para reflejar VND
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Diccionario de tiempos límites por instancia
    time_limits = {
        'VRPTW1': 50,
        'VRPTW2': 50,
        'VRPTW3': 50,
        'VRPTW4': 50,
        'VRPTW5': 50,
        'VRPTW6': 50,
        'VRPTW7': 200,
        'VRPTW8': 200,
        'VRPTW9': 200,
        'VRPTW10': 200,
        'VRPTW11': 200,
        'VRPTW12': 200,
        'VRPTW13': 750,
        'VRPTW14': 750,
        'VRPTW15': 750,
        'VRPTW16': 750,
        'VRPTW17': 750,
        'VRPTW18': 750
    }

    for i in range(1, 19):  # Ajustar el rango de acuerdo a tus archivos
        filename = f'VRPTW{i}.txt'
        instance_name = f'VRPTW{i}'
        file_path = os.path.join(directory_path, filename)
        
        if os.path.exists(file_path): 
            n, Q, nodes = read_txt_file(file_path)
            times = calculate_travel_times(nodes)
            
            # Calcular las cotas inferiores (lower bounds)
            depot = nodes[0]
            customers = nodes[1:]
            lb_routes = lower_bound_routes(customers, Q)
            lb_distance = lower_bound_mst(depot, customers, times)
            
            # Obtener el tiempo límite para la instancia actual
            time_limit = time_limits.get(instance_name, 50)  # Por defecto 50 si no está especificado

            # Medir el tiempo de cómputo para cada archivo
            start_time = time.time()
            elapsed_time = 0

            # Generar la solución inicial con Reactive GRASP
            routes, best_distance = reactive_grasp_route_selection(nodes, Q, times)
            computation_time = time.time() - start_time
            elapsed_time = computation_time

            # Aplicar VND si queda tiempo
            if elapsed_time < time_limit:
                remaining_time = time_limit - elapsed_time
                # Actualizar start_time para VND
                start_time_vnd = time.time()
                routes = vnd_algorithm(routes, times, Q, remaining_time, start_time_vnd)
                best_distance = calculate_total_distance(routes, times)
                computation_time = time.time() - start_time  # Actualizar el tiempo total
                total_computation_time += computation_time
            else:
                total_computation_time += computation_time  # Si no hay tiempo, solo acumulamos el tiempo de construcción

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
output_filename = "VRPTW_JuanFernando_Reactive_GRASP_VND.xlsx"
vrptw_solver(directory_path, output_filename)
