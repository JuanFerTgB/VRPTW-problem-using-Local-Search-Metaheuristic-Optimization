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
from Lecture import Nodo, save_to_excel, plot_routes, read_txt_file
from Feasibility_and_LB import lower_bound_routes, is_feasible, lower_bound_mst

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





def should_consider_route_pair(route1, route2, times, threshold=3):
    """
    Decide si dos rutas deben considerarse para intercambios, basado en la proximidad
    entre nodos iniciales o finales de las rutas y en la cercanía promedio de los nodos.
    
    Args:
    route1 -- Primera ruta.
    route2 -- Segunda ruta.
    times -- Matriz de tiempos de viaje.
    threshold -- Umbral de distancia para considerar rutas "vecinas".

    Returns:
    True si las rutas deben ser consideradas, False en caso contrario.
    """
    start1, end1 = route1[0], route1[-1]
    start2, end2 = route2[0], route2[-1]

    # Calculamos la cercanía media entre los nodos de las dos rutas
    avg_distance = (times[start1.index][start2.index] + times[end1.index][end2.index]) / 2

    # Solo consideramos las rutas si la cercanía promedio es menor que el umbral
    return avg_distance < threshold


def opt_2_star(routes, times, capacity, improvement_threshold=0.001):
    """
    Implementación robusta del algoritmo 2-opt* con criterios más estrictos para el intercambio.
    
    Args:
    routes -- Lista de rutas (cada ruta es una lista de objetos Nodo).
    times -- Matriz de tiempos de viaje entre nodos.
    capacity -- Capacidad máxima del vehículo.
    improvement_threshold -- Mínimo porcentaje de mejora para aplicar un intercambio.

    Returns:
    optimized_routes -- Las rutas optimizadas tras aplicar 2-opt*.
    """
    mejorado = True
    mejor_rutas = routes.copy()
    mejor_distancia_total = calculate_total_distance(mejor_rutas, times)

    while mejorado:
        mejorado = False
        # Iterar sobre combinaciones de pares de rutas cercanas
        for r1 in range(len(routes)):
            for r2 in range(r1 + 1, len(routes)):
                route1 = mejor_rutas[r1]
                route2 = mejor_rutas[r2]

                # Preselección de rutas "cercanas" con un criterio más estricto
                if not should_consider_route_pair(route1, route2, times):
                    continue

                # Probar intercambios de subsecciones entre rutas
                for i in range(1, len(route1) - 1):
                    for j in range(1, len(route2) - 1):
                        # Crear nuevas rutas intercambiando segmentos
                        new_route1 = route1[:i] + route2[j:]
                        new_route2 = route2[:j] + route1[i:]

                        # Verificación temprana de restricciones de capacidad y ventanas de tiempo
                        if not (is_feasible(new_route1[:-1], new_route1[-1], capacity, times) and 
                                is_feasible(new_route2[:-1], new_route2[-1], capacity, times)):
                            continue

                        # Calcular la nueva distancia total
                        nueva_distancia_total = calculate_total_distance([new_route1, new_route2], times)

                        # Aceptar el intercambio solo si la mejora es significativa
                        if (mejor_distancia_total - nueva_distancia_total) / mejor_distancia_total > improvement_threshold:
                            mejor_rutas[r1] = new_route1
                            mejor_rutas[r2] = new_route2
                            mejor_distancia_total = nueva_distancia_total
                            mejorado = True

    return mejor_rutas


## Función para ejecutar el solver con 2-opt*
def vrptw_solver_with_2opt_star(output_filename):
    directory_path = os.getcwd()  
    examples_path = os.path.join(directory_path, "Examples")

    wb = Workbook()
    total_computation_time = 0
    output_folder = "constructive_images_2opt_star"  # Carpeta donde se guardarán las imágenes


    for i in range(1, 19):
        filename = f'VRPTW{i}.txt'
        file_path = os.path.join(examples_path, filename)
        
        if os.path.exists(file_path):
            n, Q, nodes = read_txt_file(file_path)
            times = calculate_travel_times(nodes)

            # Calcular cotas inferiores
            depot = nodes[0]
            customers = nodes[1:]
            lb_routes = lower_bound_routes(customers, Q)
            lb_distance = lower_bound_mst(depot, customers, times)

            # Medir el tiempo de cómputo
            start_time = time.time()
            routes = route_selection(nodes, Q, times)  # Generar las rutas iniciales
            optimized_routes = opt_2_star(routes, times, Q)  # Aplicar 2-opt*
            optimized_total_distance = calculate_total_distance(optimized_routes, times)
            computation_time = time.time() - start_time
            total_computation_time += computation_time

            # Cálculo de GAPs
            actual_routes = len(optimized_routes)
            gap_routes = max(((actual_routes - lb_routes) / lb_routes) * 100 if lb_routes > 0 else 0, 0)
            gap_distance = max(((optimized_total_distance - lb_distance) / lb_distance) * 100 if lb_distance > 0 else 0, 0)

            # Mostrar resultados
            print(f"Solution for {filename}:")
            print(f"  - Total Distance = {optimized_total_distance}")
            print(f"  - Lower Bound Distance (MST) = {lb_distance:.2f}")
            print(f"  - GAP Distance = {gap_distance:.2f}")
            print(f"  - Actual Routes = {actual_routes}")
            print(f"  - Lower Bound Routes = {lb_routes}")
            print(f"  - GAP Routes = {gap_routes:.2f}")
            print(f"  - Execution Time = {computation_time * 1000:.0f} ms\n")

            # Guardar en Excel
            sheet_name = f'VRPTW{i}'
            save_to_excel(wb, sheet_name, optimized_routes, optimized_total_distance, computation_time, times)

            plot_routes(optimized_routes, filename,output_folder)
        else:
            print(f"Archivo {filename} no encontrado.")
    
    wb.save(output_filename)
    total_computation_time_ms = total_computation_time * 1000
    print(f"Total computation time for all files: {total_computation_time_ms:.4f} ms")


# Ejemplo de uso
output_filename = "VRPTW_JuanFernando_Constructivo_2opt_star.xlsx"
vrptw_solver_with_2opt_star(output_filename)




