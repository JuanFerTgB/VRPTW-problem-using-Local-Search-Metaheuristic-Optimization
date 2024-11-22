"""
Algoritmo Constructivo Problema Vehicle
Routing Problem with Time Windows – VRPTW
Class
Juan Fernando Riascos Goyes
"""
## Class that represent a node for VRPTW problem (Vehicle Routing Problem with Time Windows).
## This node has this different variables:

## index (int): Index
## x_cord (int): X-coordinate of the node
## y_cord (int): Y-coordinate of the node
## demand (int): Demand of the node 
## inflim (int): Lower limit of the time window during which the node can be serviced
## suplim (int): Upper limit of the time window during which the node can be serviced
## serv (int): Service time 
import os
import matplotlib.pyplot as plt


class Nodo:
    def __init__(self, index, x_cord, y_cord, demand, inflim, suplim, serv):
        self.index = index
        self.x_cord = x_cord
        self.y_cord = y_cord
        self.demand = demand
        self.time_window = (inflim, suplim)
        self.serv_time = serv

    def __repr__(self):
        return f"Customer <{self.index}>"

## Function to read the 18 .txt files (example problems with different properties)
def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Read n an Q
        first_line = lines[0].strip().split()
        n = int(first_line[0])
        Q = int(first_line[1])

        nodes = []

        # Read the n+1 next lines for index (i), x and y coordinate (x_i, y_i),
        # demand (q_i), Lower and upper limit for time window (e_i),(l_i),
        # and time service (s_i)
        
        for line in lines[1:n+2]: 
            parts = list(map(int, line.strip().split()))
            node = Nodo(parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[6])
            nodes.append(node)
    return n, Q, nodes


def save_to_excel(workbook, sheet_name, routes, total_distance, computation_time, times):
    ws = workbook.create_sheet(title=sheet_name)

    # Primera fila con nú mero de vehículos, distancia total y tiempo de cómputo (convertido a milisegundos)
    num_vehicles = len(routes)
    computation_time_ms = round(computation_time * 1000, 0)  # Convertir a milisegundos
    ws.append([num_vehicles, round(total_distance, 3), computation_time_ms])

    # Filas siguientes con la información de cada vehículo
    for i, route in enumerate(routes, start=1):
        route_nodes = [0]  # Iniciar con el depósito
        arrival_times = []
        current_time = 0
        total_load = 0

        for j in range(1, len(route)):
            # Sumar el tiempo de viaje entre los nodos consecutivos
            current_time += times[route[j-1].index][route[j].index]

            # Si el vehículo llega antes de la ventana de tiempo, esperar hasta la hora mínima permitida
            if current_time < route[j].time_window[0]:
                current_time = route[j].time_window[0]  # Ajustar el tiempo de llegada

            # Registrar el tiempo de llegada (redondeado a 3 decimales)
            arrival_times.append(round(current_time, 3))

            # Sumar la demanda del nodo actual a la carga total del vehículo
            total_load += route[j].demand

            # Agregar el nodo actual a la lista de nodos de la ruta
            route_nodes.append(route[j].index)

            # Sumar el tiempo de servicio en el nodo actual
            current_time += route[j].serv_time

        # Al final de la ruta, el vehículo vuelve al depósito (nodo 0)
        route_nodes.append(0)

        # Calcular el número de clientes servidos en esta ruta (excluyendo el depósito)
        num_customers = len(route_nodes) - 3  # Restar los dos nodos del depósito (inicio y final)

        # Guardar el número de clientes, los nodos de la ruta, tiempos de llegada y la carga total
        ws.append([num_customers] + route_nodes + arrival_times + [total_load])

## Plot Solutions with node numeration and with different colors depending the route
def plot_routes(routes, filename, output_folder="constructive_images"):
    # Crear la carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Generar la figura de la ruta
    plt.figure(figsize=(10, 8))
    
    for route in routes:
        x_coords = [node.x_cord for node in route]
        y_coords = [node.y_cord for node in route]
        plt.plot(x_coords, y_coords, marker='o')
        for i, node in enumerate(route):
            plt.text(node.x_cord, node.y_cord, str(node.index), fontsize=12, ha='right')
    
    plt.title(f"VRPTW Solution: {filename}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)

    # Guardar la imagen en la carpeta especificada
    output_path = os.path.join(output_folder, filename.replace('.txt', '_solution.png'))
    plt.savefig(output_path)
    #plt.show()



