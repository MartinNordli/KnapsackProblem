use csv::ReaderBuilder;
use plotters::prelude::*;
use rand::prelude::*;
use rand::Rng;
use std::error::Error;
use std::fs::File;

// ---- Genetic algorithm constants ----
const CAPACITY: u32 = 280785;       // Maximum capacity of the knapsack
const POPULATION_SIZE: usize = 100; // Number of individuals in a population at a given time
const GENOME_LENGTH: usize = 500;   // Length of each individual's genome (number of possible items)

// ---- Hyperparameters ----
const MUTATION_RATE: f64 = 0.00001; // Probability of mutation per gene
const CROSSOVER_RATE: f64 = 0.6;    // Probability of crossover between parents
const CAPACITY_PENALTY: f64 = 1.0;  // Penalty for exceeding capacity
const GENERATIONS: usize = 5500;    // Number of generations to run

/// Represents an item that can be placed in the knapsack
#[derive(Debug)]
struct Item {
    weight: u32, // Weight of the item
    profit: u32, // Value/profit of the item
}

/// Loads items from a CSV file
/// 
/// The format of the CSV file is expected to be:
/// ID, profit, weight
fn load_items_from_csv(file_path: &str) -> Result<Vec<Item>, Box<dyn Error>> {
    let rdr = ReaderBuilder::new().from_path(file_path);
    let mut items = Vec::new();

    for result in rdr?.records() {
        let record = result?;
        let profit: u32 = record.get(1).unwrap().parse()?;
        let weight: u32 = record.get(2).unwrap().parse()?;
        items.push(Item { weight, profit });
    }

    Ok(items)
}

/// Generates a random initial population
/// 
/// Each individual is represented as a vector of 0s and 1s,
/// where 1 indicates that the item is included in the knapsack
fn generate_population() -> Vec<Vec<u8>> {
    let mut population = Vec::new();
    let mut rng = rand::thread_rng();

    for _ in 0..POPULATION_SIZE {
        let individual: Vec<u8> = (0..GENOME_LENGTH) // Creates 500 unique genes
            .map(|_| rng.gen_range(0..=1))           // Randomly generates 0 or 1
            .collect();
        population.push(individual);
    }

    population
}

/// Calculates the total weight of an individual's knapsack
fn total_weight(genome: &[u8], items: &[Item]) -> u32 {
    let mut sum = 0;
    for (gene, item) in genome.iter().zip(items.iter()) {
        if *gene == 1 {
            sum += item.weight;
        }
    }
    sum
}

/// Calculates fitness (quality) for an individual
/// 
/// Fitness is the total profit, with a penalty if the capacity is exceeded.
/// Higher fitness means a better solution.
fn fitness(genome: &Vec<u8>, items: &Vec<Item>) -> f64 {
    let mut total_profit = 0.0;
    let mut total_weight = 0.0;

    for (gene, item) in genome.iter().zip(items.iter()) {
        if *gene == 1 {
            total_profit += item.profit as f64;
            total_weight += item.weight as f64;
        }
    }

    if total_weight <= CAPACITY as f64 {
        total_profit // No penalty if weight doesn't exceed capacity
    } else {
        let over_capacity = total_weight - CAPACITY as f64;
        let penalty = over_capacity * CAPACITY_PENALTY; // The penalty is adjustable
        total_profit - penalty
    }
}

/// Evaluates fitness for each individual in the population
fn evaluate_population(population: &Vec<Vec<u8>>, items: &Vec<Item>) -> Vec<f64> {
    population
        .iter()
        .map(|genome| fitness(genome, items))
        .collect()
}

/// Selects an individual based on roulette wheel selection
/// 
/// Individuals with higher fitness have a greater probability of being selected.
/// The lifetime 'a specifies that the return value points to the same data as the input.
fn roulette_wheel_selection<'a>(
    population: &'a Vec<Vec<u8>>,
    fitnesses: &'a Vec<f64>,
) -> &'a Vec<u8> {
    use rand::Rng;

    let total_fitness: f64 = fitnesses.iter().sum();
    let mut rng = rand::thread_rng();
    let mut threshold = rng.gen_range(0.0..total_fitness);

    for (i, fit) in fitnesses.iter().enumerate() {
        threshold -= fit;
        if threshold <= 0.0 {
            return &population[i];
        }
    }

    &population[population.len() - 1] // Fallback if none is selected
}

/// Selects an individual based on tournament selection
/// 
/// Selects k individuals randomly and returns the one with the highest fitness.
/// This provides better selection pressure than roulette wheel selection.
fn tournament_selection<'a>(
    population: &'a Vec<Vec<u8>>,
    fitnesses: &'a Vec<f64>,
    k: usize,
) -> &'a Vec<u8> {
    use rand::seq::SliceRandom;

    let mut rng = rand::thread_rng();
    let tournament: Vec<usize> = (0..population.len())
        .collect::<Vec<usize>>()
        .choose_multiple(&mut rng, k)
        .cloned()
        .collect();

    // Find the index with the highest fitness
    let best_index = tournament
        .iter()
        .max_by(|&&i, &&j| fitnesses[i].partial_cmp(&fitnesses[j]).unwrap())
        .unwrap();

    &population[*best_index]
}

/// Performs single-point crossover between two parents to create two new offspring
/// 
/// Selects a random point in the genome and swaps the genes after this point
/// between the parents to create two new offspring.
fn single_point_crossover(parent1: &Vec<u8>, parent2: &Vec<u8>) -> (Vec<u8>, Vec<u8>) {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    let crossover_point = rng.gen_range(0..GENOME_LENGTH);

    let mut offspring1 = Vec::new();
    let mut offspring2 = Vec::new();

    offspring1.extend_from_slice(&parent1[..crossover_point]);
    offspring1.extend_from_slice(&parent2[crossover_point..]);

    offspring2.extend_from_slice(&parent2[..crossover_point]);
    offspring2.extend_from_slice(&parent1[crossover_point..]);

    (offspring1, offspring2)
}

/// Performs mutation on a genome
/// 
/// Iterates through each gene and flips it with probability equal to MUTATION_RATE.
/// This ensures genetic diversity and prevents the algorithm from getting stuck in local maxima.
fn mutate(genome: &mut Vec<u8>, mutation_rate: f64) {
    use rand::Rng;

    let mut rng = rand::thread_rng();
    for gene in genome.iter_mut() {
        if rng.gen_bool(mutation_rate) {
            *gene = if *gene == 0 { 1 } else { 0 };
        }
    }
}

/// Main function for the genetic algorithm
/// 
/// Implements the entire process with:
/// 1. Generation of initial population
/// 2. Fitness evaluation
/// 3. Selection
/// 4. Crossover
/// 5. Mutation
/// 6. Repetition over multiple generations
fn genetic_algorithm() -> Result<Vec<(usize, f64, f64, f64)>, Box<dyn std::error::Error>> {
    let items = load_items_from_csv("../Files/KP/knapPI_12_500_1000_82.csv")?;
    let mut population = generate_population();
    let mut fitnesses = evaluate_population(&population, &items);
    let mut rng = rand::thread_rng();

    // Store statistics: generation index, min, max, average
    let mut stats = Vec::new();

    for generation in 0..GENERATIONS {
        // Calculate min, max, average for the current generation
        let max_fitness = fitnesses.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min_fitness = fitnesses.iter().copied().fold(f64::INFINITY, f64::min);
        let avg_fitness = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;

        stats.push((generation, min_fitness, max_fitness, avg_fitness));

        println!(
            "Generation {}: Best = {:.2}, Worst = {:.2}, Avg = {:.2}",
            generation + 1,
            max_fitness,
            min_fitness,
            avg_fitness
        );

        // ---- Reproduction / create new population ----
        let mut new_population = Vec::new();
        while new_population.len() < POPULATION_SIZE {
            // We use tournament selection instead of roulette wheel selection
            // let parent1 = roulette_wheel_selection(&population, &fitnesses);
            // let parent2 = roulette_wheel_selection(&population, &fitnesses);

            let parent1 = tournament_selection(&population, &fitnesses, 3);
            let parent2 = tournament_selection(&population, &fitnesses, 3);

            // Perform crossover based on CROSSOVER_RATE
            let do_crossover = rng.gen_bool(CROSSOVER_RATE);
            let (mut offspring1, mut offspring2) = if do_crossover {
                single_point_crossover(parent1, parent2)
            } else {
                (parent1.clone(), parent2.clone())
            };

            // Perform mutation on offspring
            mutate(&mut offspring1, MUTATION_RATE);
            mutate(&mut offspring2, MUTATION_RATE);

            // Add offspring to the new population
            new_population.push(offspring1);
            if new_population.len() < POPULATION_SIZE {
                new_population.push(offspring2);
            }
        }

        // Replace old population with new one
        population = new_population;
        fitnesses = evaluate_population(&population, &items);
    }

    // Final statistics for the last generation
    let max_fitness = fitnesses.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_fitness = fitnesses.iter().copied().fold(f64::INFINITY, f64::min);
    let avg_fitness = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
    stats.push((GENERATIONS, min_fitness, max_fitness, avg_fitness)); // include the last generation stats

    // Find best solution
    let (best_index, best_fitness) = fitnesses
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    let best_individual = &population[best_index];
    let best_weight = total_weight(best_individual, &items);

    // Calculate difference between best weight and capacity
    let difference = best_weight as f64 - CAPACITY as f64;
    let difference_percent = (difference.abs() / CAPACITY as f64) * 100.0;

    if difference > 0.0 {
        println!(
            "Best Weight: {} (EXCEEDS knapsack capacity by {:.2}%)",
            best_weight, difference_percent
        );
    } else {
        println!(
            "Best Weight: {} (UNDER knapsack capacity by {:.2}%)",
            best_weight, difference_percent
        );
    }

    println!("Best Fitness: {:.2}", best_fitness);
    println!("Optimal solution: 296 835");
    println!("Best Individual: {:?}", best_individual);

    // Return collected statistics for plotting
    Ok(stats)
}

/// Plots statistics over generations (minimum, maximum, and average fitness)
/// 
/// Generates a PNG file with visualization of the algorithm's performance over time
fn plot_stats(
    stats: &[(usize, f64, f64, f64)],
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Extract separate vectors for generation, min, max, average
    let gen: Vec<usize> = stats.iter().map(|(g, _, _, _)| *g).collect();
    let min_vals: Vec<f64> = stats.iter().map(|(_, min_f, _, _)| *min_f).collect();
    let max_vals: Vec<f64> = stats.iter().map(|(_, _, max_f, _)| *max_f).collect();
    let avg_vals: Vec<f64> = stats.iter().map(|(_, _, _, avg_f)| *avg_f).collect();

    // Find overall min/max for chart bounds
    let overall_min = min_vals
        .iter()
        .copied()
        .chain(max_vals.iter().copied())
        .chain(avg_vals.iter().copied())
        .fold(f64::INFINITY, f64::min);
    let overall_max = min_vals
        .iter()
        .copied()
        .chain(max_vals.iter().copied())
        .chain(avg_vals.iter().copied())
        .fold(f64::NEG_INFINITY, f64::max);

    // Prepare drawing area
    let root = BitMapBackend::new(output_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("GA Fitness Over Generations", ("sans-serif", 20))
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(
            0..gen.last().copied().unwrap_or(0),
            (overall_min - 10.0)..(overall_max + 10.0),
        )?;

    chart.configure_mesh().draw()?;

    // Draw lines for min, max, average
    chart
        .draw_series(LineSeries::new(
            gen.iter().zip(min_vals.iter()).map(|(x, y)| (*x, *y)),
            &RED, // Thicker red line
        ))?
        .label("Min Fitness")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));

    chart
        .draw_series(LineSeries::new(
            gen.iter().zip(max_vals.iter()).map(|(x, y)| (*x, *y)),
            &BLUE,
        ))?
        .label("Max Fitness")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));

    chart
        .draw_series(LineSeries::new(
            gen.iter().zip(avg_vals.iter()).map(|(x, y)| (*x, *y)),
            &GREEN,
        ))?
        .label("Avg Fitness")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &GREEN));

    // Draw legend
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

/// Main function that starts the algorithm and plots the results
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Run by being in "genetic_algorithm" and typing "cargo run --release"
    let stats = genetic_algorithm()?;

    plot_stats(&stats, "fitness_plot.png")?;

    println!("Plot saved to fitness_plot.png");
    Ok(())
}
