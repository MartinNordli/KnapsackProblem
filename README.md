# Genetic Algorithms for Optimization

This project focuses on implementing and applying **genetic algorithms** to two different optimization problems:  
1. **Solving the Knapsack Problem** using a genetic algorithm.  
2. **Feature Selection for Machine Learning** using a genetic algorithm to improve model performance.  

Each part of the project is contained in separate directories:  
- **`genetic_algorithm/`** → Implements the genetic algorithm for solving the Knapsack Problem.  
- **`feature_selection_algorithm/`** → Uses a genetic algorithm for feature selection in a machine learning context.  

---

## 📌 Part 1: Genetic Algorithm for the Knapsack Problem

### Problem Description
The **Knapsack Problem** is a combinatorial optimization problem where items with different weights and profits must be selected to maximize total profit while respecting a weight capacity constraint. A **genetic algorithm (GA)** is used to find an optimal or near-optimal solution.

### Implementation Details
- **Fitness Function**: Implemented to evaluate solutions based on item weights and profits while penalizing infeasible solutions.
- **Genetic Operators**: Custom implementations of:
  - Population initialization
  - Parent selection
  - Crossover methods
  - Mutation operators
- **Survivor Selection**: Implemented two survivor selection methods to maintain diversity and improve convergence.
- **Algorithm Execution**:
  - The GA runs for multiple generations while tracking **maximum, minimum, and mean fitness** values.
  - The objective is to find a **feasible solution close to the known optimum** (≥ 29,000).

📂 **Code Location**: `genetic_algorithm/`

---

## 📌 Part 2: Feature Selection Using Genetic Algorithm

### Problem Description
**Feature selection** is a machine learning technique used to remove irrelevant and redundant features to improve model performance. In this project, a **genetic algorithm** is applied to **select the best subset of features** for a dataset, using **Root Mean Square Error (RMSE)** as the fitness function.

### Implementation Details
- **Fitness Function**: The GA evaluates feature subsets by feeding them into a **linear regression model**, which returns the RMSE. The goal is to **minimize** the RMSE.
- **Dataset**: Contains **1994 rows and 102 columns**, where the first 101 columns represent features, and the last column represents the target value.
- **Machine Learning Model**: A **linear regression model** is used for evaluation.
- **Survivor Selection with Crowding Techniques**:
  - Implemented a new **crowding-based selection function** to enhance diversity.
  - Compared results between **Simple Genetic Algorithm (SGA)** and **crowding-based selection**.
  - Plotted entropy changes over generations to analyze diversity.

### Results & Performance
- **RMSE Thresholds**:
  - **Without feature selection**: Baseline RMSE is computed.
  - **With GA-based feature selection**:
    - **For a 1-person team**: RMSE < **0.125**.
    - **For a 2+ person team**: RMSE < **0.124**.

📂 **Code Location**: `feature_selection_algorithm/`

---

## 📊 Results & Insights
- **Knapsack Problem**: The genetic algorithm successfully finds near-optimal solutions while balancing exploration and exploitation.
- **Feature Selection**: The GA significantly reduces the number of features while maintaining or improving model performance.
- **Crowding Selection**: Helps maintain genetic diversity and prevents premature convergence.