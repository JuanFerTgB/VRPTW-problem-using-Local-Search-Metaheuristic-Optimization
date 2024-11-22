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


def vrptw_solver(directory_path, output_filename):
    wb = Workbook()
    wb.remove(wb.active)

    results = []
    execution_times = []  # Lista para guardar tiempos de ejecución
    output_folder = "Grasp_reactive_2opt_star_images"
    for i in range(1, 19):  # Ajustar el rango de acuerdo a tus archivos
        filename = f'VRPTW{i}.txt'
        file_path = os.path.join(directory_path, filename)
        file_start_time = time.time()  # Tiempo de inicio

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
    
    
    # Ordenar los resultados por nombre de archivo para asegurarse que estén en orden
    results.sort(key=lambda x: int(x[0].split('VRPTW')[1].split('.txt')[0]))

# Ejemplo de uso
directory_path = "./Examples"
output_filename = "VRPTW_JuanFernando_Reactive_GRASP_2opt_star.xlsx"
vrptw_solver(directory_path, output_filename)
