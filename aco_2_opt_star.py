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
random.seed(4)

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
                routes.append(route)

            total_distance = sum(calculate_total_distance([route], times) for route in routes)
            all_routes.append((routes, total_distance))

            # Actualizar la mejor solución si es mejor que la actual
            if total_distance < best_distance:
                best_distance = total_distance
                best_routes = routes

        # Actualizar feromonas
        update_pheromones(pheromones, all_routes, Q, rho)

    return best_routes, best_distance


def vrptw_solver(directory_path, output_filename):
    wb = Workbook()
    wb.remove(wb.active)

    execution_times = []  # Lista para guardar tiempos de ejecución
    output_folder = "ACO_images_2_opt_star"  # Carpeta donde se guardarán las imágenes

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

        # Aplicar ACO para obtener las rutas iniciales
        routes, best_distance = aco_vrptw(nodes, Q, times, **aco_params)
        
        # Aplicar 2-opt* a las rutas encontradas
        optimized_routes = opt_2_star(routes, times, Q)  # Aplicar 2-opt* para mejorar las rutas
        best_distance = calculate_total_distance(optimized_routes, times)  # Recalcular la distancia

        computation_time = (time.time() - file_start_time) * 1000  # Tiempo en milisegundos
        execution_times.append(computation_time)

        # Calcular el GAP para número de rutas y distancia total
        actual_routes = len(optimized_routes)
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
        save_to_excel(wb, sheet_name, optimized_routes, best_distance, computation_time, times)

        # Llamar a plot_routes para graficar y guardar las rutas en la carpeta ACO_images
        plot_routes(optimized_routes, f"VRPTW{i}.txt", output_folder)

    # Guardar el archivo Excel
    wb.save(output_filename)

    total_elapsed_time = sum(execution_times)  # Tiempo total en milisegundos
    print(f"\nTotal execution time: {total_elapsed_time:.0f} ms")



# Ejemplo de uso
directory_path = "./Examples"  # Carpeta con los archivos de entrada
output_filename = "VRPTW_JuanFernando_ACO_2_opt__star.xlsx"
vrptw_solver(directory_path, output_filename)

