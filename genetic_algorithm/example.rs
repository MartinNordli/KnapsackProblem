use rand::prelude::*;
use rand::Rng;

const POPULATION_SIZE: usize = 100;
const GENOME_LENGTH: usize = 20;
const MUTATION_RATE: f64 = 0.01;
const CROSSOVER_RATE: f64 = 0.7;
const GENERATIONS: usize = 200;

fn random_genome(length: usize) -> Vec<u8> {
    let mut rng = rand::thread_rng();
    (0..length).map(|_| rng.gen_range(0..=1)).collect()
}

fn init_population(population_size: usize, genome_length: usize) -> Vec<Vec<u8>> {
    (0..population_size)
        .map(|_| random_genome(genome_length))
        .collect()
}

fn fitness(genome: &[u8]) -> usize {
    genome.iter().map(|&gene| gene as usize).sum()
}

fn select_parent<'a>(population: &'a [Vec<u8>], fitness_values: &[usize]) -> &'a Vec<u8> {
    let total_fitness: usize = fitness_values.iter().sum();
    let mut rng = rand::thread_rng();
    let pick = rng.gen_range(0..total_fitness);
    let mut current = 0;

    for (individual, &fitness_value) in population.iter().zip(fitness_values.iter()) {
        current += fitness_value;
        if current > pick {
            return individual;
        }
    }
    &population[0] // Fallback (shouldn't happen)
}

fn crossover(parent1: &[u8], parent2: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let mut rng = rand::thread_rng();
    if rng.gen::<f64>() < CROSSOVER_RATE {
        let crossover_point = rng.gen_range(1..parent1.len());
        (
            parent1[..crossover_point]
                .iter()
                .chain(&parent2[crossover_point..])
                .cloned()
                .collect(),
            parent2[..crossover_point]
                .iter()
                .chain(&parent1[crossover_point..])
                .cloned()
                .collect(),
        )
    } else {
        (parent1.to_vec(), parent2.to_vec())
    }
}

fn mutate(genome: &mut Vec<u8>) {
    let mut rng = rand::thread_rng();
    for gene in genome.iter_mut() {
        if rng.gen::<f64>() < MUTATION_RATE {
            *gene = 1 - *gene; // Flip 0 to 1 or 1 to 0
        }
    }
}

fn genetic_algorithm() {
    let mut population = init_population(POPULATION_SIZE, GENOME_LENGTH);

    for generation in 0..GENERATIONS {
        let fitness_values: Vec<usize> = population.iter().map(|genome| fitness(genome)).collect();

        let mut new_population = Vec::new();
        for _ in 0..(POPULATION_SIZE / 2) {
            let parent1 = select_parent(&population, &fitness_values);
            let parent2 = select_parent(&population, &fitness_values);

            let (mut offspring1, mut offspring2) = crossover(parent1, parent2);

            mutate(&mut offspring1);
            mutate(&mut offspring2);

            new_population.push(offspring1);
            new_population.push(offspring2);
        }

        population = new_population;

        let best_fitness = *fitness_values.iter().max().unwrap();
        println!("Generation {}: Best fitness = {}", generation, best_fitness);
    }

    let fitness_values: Vec<usize> = population.iter().map(|genome| fitness(genome)).collect();
    let best_index = fitness_values
        .iter()
        .enumerate()
        .max_by_key(|&(_, fitness)| fitness)
        .unwrap()
        .0;
    let best_solution = &population[best_index];

    println!("Best solution: {:?}", best_solution);
    println!("Best fitness: {}", fitness(best_solution));
}

fn main() {
    genetic_algorithm();
}
