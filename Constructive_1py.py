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

def vrptw_solver(output_filename):
    # Obtener la ruta del directorio donde se ejecuta el código
    directory_path = os.getcwd()  # Obtiene el directorio actual
    examples_path = os.path.join(directory_path, "Examples")  # Asume que hay una carpeta "Examples" en el directorio actual

    
    wb = Workbook()
    total_computation_time = 0  # Para calcular el tiempo total de ejecución

    output_folder = "constructive_images"  # Carpeta donde se guardarán las imágenes
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

            plot_routes(routes, filename,output_folder)
        else:
            print(f"Archivo {filename} no encontrado.")
    
    # Guardar el archivo de Excel con todas las hojas
    wb.save(output_filename)
    
    # Imprimir el tiempo total de cómputo al final en milisegundos
    total_computation_time_ms = total_computation_time * 1000
    print(f"Total computation time for all files: {total_computation_time_ms:.4f} ms")


# Ejemplo de uso
output_filename = "VRPTW_JuanFernando_Constructivo.xlsx"
vrptw_solver(output_filename)



