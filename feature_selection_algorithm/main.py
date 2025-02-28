import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

# ------------------------ Constants ------------------------
POPULATION_SIZE = 100
GENOME_LENGTH = 101
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.6
GENERATIONS = 100

# ------------------------ Load Dataset ------------------------
def load_dataset(file_path):
    """Load dataset and split into features (X) and target (y)."""
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

# ------------------------ Fitness Calculation ------------------------
def evaluate_fitness(genome, X, y):
    """Calculate the RMSE for the selected features based on the genome."""
    selected_features = [i for i, bit in enumerate(genome) if bit == 1]
    if not selected_features:
        return float('inf')  # Penalize empty selection

    X_selected = X[:, selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    return rmse

# ------------------------ Population Generation ------------------------
def generate_population():
    """Creates an initial population of random genomes."""
    return [np.random.randint(0, 2, size=GENOME_LENGTH).tolist() for _ in range(POPULATION_SIZE)]

# ------------------------ Genetic Operators ------------------------
def single_point_crossover(parent1, parent2):
    """Perform single-point crossover on two parents and return two offspring."""
    if random.random() < CROSSOVER_RATE:
        crossover_point = random.randint(1, GENOME_LENGTH - 1)
        return (parent1[:crossover_point] + parent2[crossover_point:],
                parent2[:crossover_point] + parent1[crossover_point:])
    return parent1[:], parent2[:]

def mutate(genome):
    """Flip bits with a probability of MUTATION_RATE."""
    return [1 - bit if random.random() < MUTATION_RATE else bit for bit in genome]

# ------------------------ Selection Methods ------------------------
def tournament_selection(population, fitnesses, k=3):
    """Select the best individual from a random subset (tournament) of size k."""
    selected = random.sample(range(len(population)), k)
    best_index = min(selected, key=lambda idx: fitnesses[idx])
    return population[best_index]

# ------------------------ Survivor Selection: Crowding ------------------------
def hamming_distance(genome1, genome2):
    """Calculate Hamming distance between two genomes."""
    return sum(g1 != g2 for g1, g2 in zip(genome1, genome2))

def deterministic_crowding(parents, offspring, fitnesses):
    """Implement deterministic crowding survivor selection."""
    survivors = []
    for i in range(0, len(parents), 2):
        parent1, parent2 = parents[i], parents[i+1]
        child1, child2 = offspring[i], offspring[i+1]

        # Compare offspring to parents based on fitness and similarity
        if hamming_distance(parent1, child1) <= hamming_distance(parent1, child2):
            survivors.append(child1 if evaluate_fitness(child1, X, y) < fitnesses[i] else parent1)
            survivors.append(child2 if evaluate_fitness(child2, X, y) < fitnesses[i+1] else parent2)
        else:
            survivors.append(child2 if evaluate_fitness(child2, X, y) < fitnesses[i] else parent1)
            survivors.append(child1 if evaluate_fitness(child1, X, y) < fitnesses[i+1] else parent2)
    return survivors

# ------------------------ Genetic Algorithm ------------------------
def genetic_algorithm(X, y, use_crowding=False):
    """Runs the genetic algorithm for feature selection."""
    population = generate_population()
    best_rmse = float('inf')
    best_genome = None

    stats = []  # Collect stats for plotting
    log_dir = "logs/fit/" + tf.timestamp().__str__()
    summary_writer = tf.summary.create_file_writer(log_dir)

    for generation in range(GENERATIONS):
        fitnesses = [evaluate_fitness(genome, X, y) for genome in population]

        # Track the best solution
        generation_best_rmse = min(fitnesses)
        if generation_best_rmse < best_rmse:
            best_rmse = generation_best_rmse
            best_genome = population[fitnesses.index(best_rmse)]

        # Entropy calculation
        entropy = -sum((np.mean([genome[i] for genome in population]) *
                        np.log2(np.mean([genome[i] for genome in population])))
                       for i in range(GENOME_LENGTH) if np.mean([genome[i] for genome in population]) > 0)

        stats.append((generation, best_rmse, sum(fitnesses) / len(fitnesses), entropy))

        print(f"Generation {generation}: Best RMSE = {best_rmse:.4f}, Entropy = {entropy:.4f}")

        # TF Logging
        with summary_writer.as_default():
            tf.summary.scalar('Fitness/Best_RMSE', best_rmse, step=generation)
            tf.summary.scalar('Population/Entropy', entropy, step=generation)
            tf.summary.scalar('Fitness/Average_RMSE', sum(fitnesses) / len(fitnesses), step=generation)

        # Create offspring population
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            offspring1, offspring2 = single_point_crossover(parent1, parent2)
            new_population.append(mutate(offspring1))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate(offspring2))

        # Survivor selection
        if use_crowding:
            population = deterministic_crowding(population, new_population, fitnesses)
        else:
            population = new_population

    return best_genome, best_rmse, stats

# ------------------------ Plotting ------------------------
def plot_stats(stats, output_path="fitness_plot.png"):
    """Plots the minimum, average RMSE and entropy over generations."""
    generations = [s[0] for s in stats]
    min_rmse = [s[1] for s in stats]
    avg_rmse = [s[2] for s in stats]
    entropy = [s[3] for s in stats]

    plt.figure(figsize=(10, 6))
    plt.plot(generations, min_rmse, label='Min RMSE', color='blue')
    plt.plot(generations, avg_rmse, label='Avg RMSE', color='green')
    plt.plot(generations, entropy, label='Entropy', color='red')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.title('GA Statistics Over Generations')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()

# ------------------------ Main ------------------------
if __name__ == "__main__":
    dataset_path = "../Files/feature_selection/dataset.txt"
    X, y = load_dataset(dataset_path)

    # Run both SGA and crowding GA
    print("Running Simple Genetic Algorithm (SGA)...")
    best_genome_sga, best_rmse_sga, stats_sga = genetic_algorithm(X, y, use_crowding=False)

    print("Running Genetic Algorithm with Crowding...")
    best_genome_crowding, best_rmse_crowding, stats_crowding = genetic_algorithm(X, y, use_crowding=True)

    # Compare results
    print(f"SGA: Best RMSE = {best_rmse_sga:.4f}")
    print(f"Crowding: Best RMSE = {best_rmse_crowding:.4f}")
    
    # Plot results
    plot_stats(stats_sga, output_path="sga_stats.png")
    plot_stats(stats_crowding, output_path="crowding_stats.png")