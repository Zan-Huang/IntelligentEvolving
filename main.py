# DISCLAIMER:
#   I apologize for any potentially redundant or unused code. Getting this to work was quite
#   the time committment. Being short on time, I had to get it working somehow
#   without accidentally breaking something through deletion.
#
#   Please commment out the bottom ofs simulate to run the main.py. It's not the end of the world
#   if you don't but it would force to wait through a while. If you want to run a visualization,
#   please run the complete simulate code uncommented. If you want to run a parameter sweep,
#   tun this file.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from simulate import Environment
import multiprocessing

simulation_steps = 45000
environment_size = 1000
max_food_centroid = 1000
num_food_centroids = 120
initial_agent_count = 20
agent_max_lifespan = 100
reproduction_cycle = agent_max_lifespan / 2
agent_size_range = (15, 15)
agent_flagella_range = (8, 8)
starting_energy = 50
energy_gain_food = 10
chance_to_skip_food = 0.0
energy_cost_reproduction = 0
num_offspring_range = (3, 5)
energy_decay_rate = 1

def run_simulation(mutation_rate, food_regeneration_rate, runs=3):
    print("new sim: mutation: ", mutation_rate, " food: ", food_regeneration_rate)
    generations = simulation_steps // agent_max_lifespan

    aggregate_populations = np.zeros(generations)
    aggregate_entropies = np.zeros(generations)

    for _ in range(runs):
        print("new sim: mutation: ", mutation_rate, " food: ", food_regeneration_rate, _) # just to check its working
        environment = Environment(food_regeneration_rate, num_food_centroids, agent_max_lifespan, reproduction_cycle,
                                  simulation_steps, environment_size, max_food_centroid, initial_agent_count, mutation_rate)
        environment.initialize_agents(mutation_rate, agent_size_range, agent_flagella_range)

        populations = []
        entropies = []

        for step in range(simulation_steps):
            environment.step()
            if step % agent_max_lifespan == 0:
                populations.append(len(environment.agents))
                entropies.append(environment.total_entropy / environment.max_entropy if environment.max_entropy != 0 else 0)

        populations = populations[:generations] + [populations[-1]] * (generations - len(populations))
        entropies = entropies[:generations] + [entropies[-1]] * (generations - len(entropies))

        aggregate_populations += np.array(populations)
        aggregate_entropies += np.array(entropies)

    avg_populations = aggregate_populations / runs
    avg_entropies = aggregate_entropies / runs

    return {
        'Mutation Rate': mutation_rate,
        'Food Regeneration Rate': food_regeneration_rate,
        'Average Population': avg_populations,
        'Average Entropy Ratio': avg_entropies
    }

def plot_from_csv():
    combined_df = pd.read_csv('combined_data.csv')

    unique_params = combined_df[['Mutation Rate', 'Food Regeneration Rate']].drop_duplicates()

    for _, row in unique_params.iterrows():
        mutation_rate = row['Mutation Rate']
        food_regeneration_rate = row['Food Regeneration Rate']
        subset = combined_df[(combined_df['Mutation Rate'] == mutation_rate) &
                             (combined_df['Food Regeneration Rate'] == food_regeneration_rate)]

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(subset['Generation'], subset['Average Population'], label='Population')
        plt.xlabel('Generation')
        plt.ylabel('Population')
        plt.title(f'Population over Time\nMutation: {mutation_rate}, Food Regen: {food_regeneration_rate}')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(subset['Generation'], subset['Average Entropy Ratio'], label='Entropy Ratio')
        plt.xlabel('Generation')
        plt.ylabel('Entropy Ratio')
        plt.title(f'Entropy Ratio over Time\nMutation: {mutation_rate}, Food Regen: {food_regeneration_rate}')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'plot_{mutation_rate}_{food_regeneration_rate}.png')
        plt.close()

if __name__ == '__main__':
    mutation_rates = [0.0, 0.05, 0.1, 0.25, 0.5]
    food_regeneration_rates = [0.25, 0.5, 0.75]

    params = [(mutation, food) for mutation in mutation_rates for food in food_regeneration_rates]

    results = []
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.starmap(run_simulation, params)

    valid_results = [res for res in results if res is not None]

    generations = int(simulation_steps / agent_max_lifespan)
    
    combined_data = {
        'Generation': np.tile(range(generations), len(valid_results)),
        'Mutation Rate': np.repeat([res['Mutation Rate'] for res in valid_results], generations),
        'Food Regeneration Rate': np.repeat([res['Food Regeneration Rate'] for res in valid_results], generations),
        'Average Population': np.concatenate([res['Average Population'] for res in valid_results]),
        'Average Entropy Ratio': np.concatenate([res['Average Entropy Ratio'] for res in valid_results])
    }

    combined_df = pd.DataFrame(combined_data)
    combined_df.to_csv('combined_data.csv', index=False)

    plot_from_csv()
