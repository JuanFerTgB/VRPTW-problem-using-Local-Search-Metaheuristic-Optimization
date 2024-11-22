def select_parents(population, fitness_values):
    """
    Selecciona a los padres utilizando el método de ruleta o selección por torneo.
    """
    total_fitness = sum(fitness_values)
    selection_probs = [fitness / total_fitness for fitness in fitness_values]
    parents = random.choices(population, weights=selection_probs, k=2)  # Selecciona 2 padres
    return parents
