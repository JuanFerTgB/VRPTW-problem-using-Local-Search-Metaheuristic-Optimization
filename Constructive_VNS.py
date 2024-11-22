## Juan Fernando Riascos Goyes
## Constructive Heuristic Method for VRPTW problem 
 
## Libraries 
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import Workbook
import time
from scipy.sparse.csgraph import minimum_spanning_tree
from Lecture import Nodo,save_to_excel,plot_routes,read_txt_file
from Feasibility_and_LB import lower_bound_routes,is_feasible,lower_bound_mst


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

## Constructive method to select the "optimal" route based on the above restrictions
def route_selection(nodes, capacity, times):
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
            next_customer = min(feasible_customers, key=lambda x: times[route[-1].index][x.index])
            if current_load + next_customer.demand <= capacity:
                route.append(next_customer)
                current_load += next_customer.demand
                customers.remove(next_customer)
            else:
                break
        route.append(depot)
        routes.append(route)
    return routes



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


def swap_between_routes(routes, times, capacity):
    import random
    new_routes = [route.copy() for route in routes]

    if len(new_routes) < 2:
        return new_routes  # No hay suficientes rutas para intercambiar

    route_indices = list(range(len(new_routes)))
    random.shuffle(route_indices)

    for i in route_indices:
        for j in route_indices:
            if i >= j:
                continue  # Evitar duplicados y misma ruta

            route1 = new_routes[i]
            route2 = new_routes[j]

            # Excluir los depósitos
            customers1 = route1[1:-1]
            customers2 = route2[1:-1]

            for idx1, cust1 in enumerate(customers1):
                for idx2, cust2 in enumerate(customers2):
                    # Crear copias de las rutas
                    temp_route1 = route1.copy()
                    temp_route2 = route2.copy()

                    # Intercambiar los clientes
                    temp_route1[idx1 + 1] = cust2  # +1 por el depósito al inicio
                    temp_route2[idx2 + 1] = cust1

                    # Verificar factibilidad de ambas rutas utilizando is_feasible
                    if (is_route_feasible(temp_route1, capacity, times) and
                        is_route_feasible(temp_route2, capacity, times)):
                        # Reemplazar las rutas originales
                        new_routes[i] = temp_route1
                        new_routes[j] = temp_route2
                        return new_routes  # Retornar después de la primera mejora
    return new_routes  # Si no se encontraron mejoras

def relocate_between_routes(routes, times, capacity):
    import random
    new_routes = [route.copy() for route in routes]

    if len(new_routes) < 2:
        return new_routes  # No hay suficientes rutas para reubicar

    route_indices = list(range(len(new_routes)))
    random.shuffle(route_indices)

    for i in route_indices:
        for j in route_indices:
            if i == j:
                continue  # No reubicar en la misma ruta

            route_from = new_routes[i]
            route_to = new_routes[j]

            # Excluir los depósitos
            customers_from = route_from[1:-1]

            for idx_cust, cust in enumerate(customers_from):
                # Crear copias de las rutas
                temp_route_from = route_from.copy()
                temp_route_to = route_to.copy()

                # Remover el cliente de la ruta origen
                del temp_route_from[idx_cust + 1]  # +1 por el depósito al inicio

                # Intentar insertar el cliente en todas las posiciones de la ruta destino
                for k in range(1, len(temp_route_to)):  # Evitar posición 0 (depósito)
                    temp_route_to_insert = temp_route_to[:k] + [cust] + temp_route_to[k:]

                    # Verificar factibilidad de ambas rutas utilizando is_feasible
                    if (is_route_feasible(temp_route_from, capacity, times) and
                        is_route_feasible(temp_route_to_insert, capacity, times)):
                        # Reemplazar las rutas originales
                        new_routes[i] = temp_route_from
                        new_routes[j] = temp_route_to_insert
                        return new_routes  # Retornar después de la primera mejora
    return new_routes  # Si no se encontraron mejoras

def two_opt_within_route(routes, times, capacity):
    new_routes = [route.copy() for route in routes]

    for idx, route in enumerate(new_routes):
        best_distance = calculate_route_distance(route, times)
        best_route = route.copy()
        improved = False

        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]

                # Verificar factibilidad utilizando is_feasible
                if is_route_feasible(new_route, capacity, times):
                    new_distance = calculate_route_distance(new_route, times)
                    if new_distance + 1e-6 < best_distance:
                        best_distance = new_distance
                        best_route = new_route
                        improved = True

        if improved:
            new_routes[idx] = best_route
            return new_routes  # Retornar después de la primera mejora

    return new_routes  # Si no se encontraron mejoras

def vns_algorithm(routes, times, capacity, max_iterations=100):
    """
    Implementa el algoritmo VNS para mejorar las rutas dadas.
    """
    best_routes = [route.copy() for route in routes]
    best_distance = calculate_total_distance(best_routes, times)
    iteration = 0

    # Definir las vecindades
    neighborhoods = [swap_between_routes, relocate_between_routes, two_opt_within_route]

    while iteration < max_iterations:
        iteration += 1
        improved = False

        for neighborhood in neighborhoods:
            new_routes = neighborhood(best_routes, times, capacity)
            new_distance = calculate_total_distance(new_routes, times)

            # Solo aceptamos soluciones que mejoran la distancia total
            if new_distance + 1e-6 < best_distance:
                best_routes = [route.copy() for route in new_routes]
                best_distance = new_distance
                improved = True
                break  # Reiniciar desde la primera vecindad

        if not improved:
            break  # No se encontraron mejoras, terminar

    return best_routes




def vrptw_solver(output_filename):
    # Obtener la ruta del directorio donde se ejecuta el código
    directory_path = os.getcwd()  # Obtiene el directorio actual
    examples_path = os.path.join(directory_path, "Examples")  # Asume que hay una carpeta "Examples" en el directorio actual

    wb = Workbook()
    total_computation_time = 0  # Para calcular el tiempo total de ejecución

    output_folder = "constructive_VNS_images"  # Carpeta donde se guardarán las imágenes
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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
            routes = route_selection(nodes, Q, times)

            # Aplicar VNS para mejorar las rutas
            routes = vns_algorithm(routes, times, Q)

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
            print(f"  - GAP Distance = {gap_distance:.2f}%")
            print(f"  - Actual Routes = {actual_routes}")
            print(f"  - Lower Bound Routes = {lb_routes}")
            print(f"  - GAP Routes = {gap_routes:.2f}%")
            print(f"  - Execution Time = {computation_time * 1000:.0f} ms\n")

            # Guardar los resultados en una hoja de Excel con el nombre de la instancia
            sheet_name = f'VRPTW{i}'
            save_to_excel(wb, sheet_name, routes, total_distance, computation_time, times)

            plot_routes(routes, filename, output_folder)
        else:
            print(f"Archivo {filename} no encontrado.")
    
    # Guardar el archivo de Excel con todas las hojas
    wb.save(output_filename)
    
    # Imprimir el tiempo total de cómputo al final en milisegundos
    total_computation_time_ms = total_computation_time * 1000
    print(f"Total computation time for all files: {total_computation_time_ms:.4f} ms")



# Ejemplo de uso
output_filename = "VRPTW_JuanFernando_Constructivo_VNS.xlsx"
vrptw_solver(output_filename)


