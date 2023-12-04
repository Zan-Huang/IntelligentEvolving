# DISCLAIMER:
#   I apologize for any potentially redundant or unused code. Getting this to work was quite
#   the time committment. Being short on time, I had to get it working somehow
#   without accidentally breaking something through deletion.
#
#   Run this file uncommented completely for visualization of the simulation. 


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import imageio
import os
import time
from sklearn import preprocessing as p 

food_regeneration_rate = 0.2

simulation_steps = 5000
environment_size = 1000
max_food_centroid = 500
num_food_centroids = 240
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
food_regeneration_rate = 0.4
energy_decay_rate = 1
save_gif = False # Set this to true if you want to generate a gif.
gif_duration = 54
mutation_rate = 0.05

class Environment:
    def __init__(self, food_regeneration_rate, num_food_centroids, agent_max_lifespan, reproduction_cycle, simulation_steps, environment_size, max_food_centroid, initial_agent_count, mutation_rate):
        self.food_regeneration_rate = food_regeneration_rate
        self.num_food_centroids = num_food_centroids
        self.agent_max_lifespan = agent_max_lifespan
        self.reproduction_cycle = reproduction_cycle
        self.simulation_steps = simulation_steps
        self.environment_size = environment_size
        self.max_food_centroid = max_food_centroid
        self.initial_agent_count = initial_agent_count
        self.time = 0
        self.agents = []
        self.food_centroids = np.random.rand(num_food_centroids, 2) * environment_size #np.array([[250, 250],[250, 750],[750, 250],[750, 750]])
        self.food_amounts = np.full(num_food_centroids, max_food_centroid)
        self.total_entropy = 0
        self.max_entropy = 0
        self.entropy_ratios = 0
        self.mutation_rate = mutation_rate

    def step(self):
        if self.time == 0 or self.time % self.reproduction_cycle == 0:
            self.max_entropy = self.calculate_max_entropy()
            self.total_entropy = self.calculate_total_entropy()
        self.time += 1
        self.move_food_centroids()
        self.regenerate_food()
        self.manage_agents()

    def move_food_centroids(self):
        movement = np.clip(np.random.normal(0, 1, (self.num_food_centroids, 2)) * 50, -4, 4)
        self.food_centroids = np.clip(self.food_centroids + movement, 0, self.environment_size)

    def regenerate_food(self):
        self.food_amounts = np.minimum(self.food_amounts + self.food_regeneration_rate, self.max_food_centroid)

    def manage_agents(self):
        new_agents = []
        for agent in self.agents:
            agent.act(self)
            if agent.energy <= 0:
                continue
            if self.time % self.reproduction_cycle == 0:
                new_agents.extend(agent.reproduce())
            if agent.age < self.agent_max_lifespan:
                new_agents.append(agent)
        self.agents = new_agents

    def visualize(self, ax):
        ax.clear()
        ax.set_xlim(0, self.environment_size)
        ax.set_ylim(0, self.environment_size)

        for i, centroid in enumerate(self.food_centroids):
            red_intensity = self.food_amounts[i] / self.max_food_centroid
            blue_intensity = 1 - red_intensity
            food_color = np.array(mcolors.to_rgb('red')) * red_intensity + np.array(mcolors.to_rgb('blue')) * blue_intensity
            food_color = np.clip(food_color, 0, 1)
            centroid_size = self.max_food_centroid
            ax.scatter(centroid[0], centroid[1], color=food_color, s=centroid_size, alpha=0.5)

        for agent in self.agents:
            ax.scatter(agent.position[0], agent.position[1], color='black', edgecolors='white', s=agent.size * 10)
            for flagellum in agent.flagella_positions:
                end_point = agent.position + np.array([np.cos(flagellum), np.sin(flagellum)]) * (agent.size / 4)
                ax.plot([agent.position[0], end_point[0]], [agent.position[1], end_point[1]], 'r-')

        generation = self.time // self.reproduction_cycle
        population = len(self.agents)
        ax.text(self.environment_size * 0.8, self.environment_size * 0.9, f"Time: {self.time}\nGeneration: {generation}\nPopulation: {population}", 
                bbox=dict(facecolor='white', alpha=0.5), fontsize=10)
        
        ax.text(self.environment_size * 0.8, self.environment_size * 0.8, f'Total Entropy: {self.total_entropy:.2f}',
                bbox=dict(facecolor='white', alpha=0.5), fontsize=10)

        ax.text(self.environment_size * 0.8, self.environment_size * 0.7, f'Max Entropy: {self.max_entropy :.2f}', 
                bbox=dict(facecolor='white', alpha=0.5), fontsize=10)
        if(self.max_entropy > 0):
            ax.text(self.environment_size * 0.8, self.environment_size * 0.6, f'Entropy Ratio: {(self.total_entropy / self.max_entropy):.2f}', 
                    bbox=dict(facecolor='white', alpha=0.5), fontsize=10)

    def initialize_agents(self, mutation_rate, agent_size_range, agent_flagella_range):
        for _ in range(self.initial_agent_count):
            size = random.uniform(*agent_size_range)
            num_flagella = random.randint(*agent_flagella_range)
            flagella_positions = np.random.uniform(0, 2 * np.pi, num_flagella)
            phenotype = np.concatenate(([size, num_flagella], flagella_positions))
            self.agents.append(Agent(phenotype, self.environment_size, mutation_rate, np.array([np.random.dirichlet(np.ones(8)) for _ in range(8)])))
    
    def calculate_total_entropy(self):
        if not self.agents:
            return 0

        total_entropy = 0
        for agent in self.agents:
            entropy = agent.calculate_entropy()
            total_entropy += entropy

        return total_entropy

    def calculate_max_entropy(self):
        if not self.agents:
            return 0
        max_entropy = 0
        for agent in self.agents:
            entropy = agent.calculate_max_entropy()
            max_entropy += entropy

        return max_entropy

class Agent:
    def __init__(self, phenotype, environment_size, mutation_rate, new_transition_matrix, inherited_q_table=None):
        self.phenotype = phenotype #legacy code from past examples, just ignore it for now
        self.size = phenotype[0]
        self.num_flagella = 8
        self.flagella_positions = self.initialize_flagella_positions()
        self.position = np.array([500, 500]) + np.random.normal(100)
        self.velocity = np.array([0.0, 0.0]) + np.random.normal(0)
        self.energy = starting_energy
        self.age = 0
        self.environment_size = environment_size
        self.energy_decay = energy_decay_rate
        self.mutation_rate = mutation_rate
        self.transition_matrix = new_transition_matrix

        self.q_table = inherited_q_table if inherited_q_table is not None else np.zeros((2048, 8))
        self.gamma = 0.9

        self.alpha = 0.1
        self.epsilon = 0.1
        self.is_on_right_side = False
    
    def move(self, environment):
        total_movement = np.zeros(2)

        current_state = self.get_state(environment)
        movement_index = self.choose_action(current_state)

        reward = self.calculate_reward(environment)
        next_state = self.get_state(environment)
        self.update_q_table(current_state, movement_index, reward, next_state)
        
        if np.isnan(self.transition_matrix).any():
            self.transition_matrix = np.nan_to_num(self.transition_matrix, nan=1.0 / self.transition_matrix.shape[1])

        use_vector = self.transition_matrix[movement_index]
        
        movement_choice = np.random.choice(range(8), p=use_vector)
        direction_angle = self.flagella_positions[movement_choice] + np.pi/4 * movement_index
        direction_vector = np.array([np.cos(direction_angle), np.sin(direction_angle)])

        flagella_effect_limit = self.size * 20
        movement = direction_vector * flagella_effect_limit / self.size # cancels out self.size but there for the physics illustration :)

        total_movement += movement 

        position = np.mod(self.position + total_movement, environment.environment_size)
        
        self.position = position

    def update_q_table(self, state, action, reward, next_state):
        max_future_q = np.max(self.q_table[next_state])
        current_q = self.q_table[state, action]
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * max_future_q)
        self.q_table[state, action] = new_q

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(8))
        return np.argmax(self.q_table[state])
   
    def food_direction(self, environment):
        directions = np.zeros(8, dtype=int)
        food_radius = self.size * 60

        for centroid in environment.food_centroids:
            if np.linalg.norm(centroid - self.position) < food_radius:
                delta_x = centroid[0] - self.position[0]
                delta_y = centroid[1] - self.position[1]
                direction_index = self.get_direction_index(delta_x, delta_y)
                directions[direction_index] = 1

        return directions

    def get_direction_index(self, delta_x, delta_y):
        angle = np.arctan2(delta_y, delta_x)
        index = int(np.round(np.degrees(angle) / 45)) % 8
        return index

    def get_state(self, environment):
        right_side = int(self.position[0] > environment.environment_size / 2)
        top_half = int(self.position[1] > environment.environment_size / 2)
        enough_energy = int(self.energy >= (starting_energy + 20))
        food_directions = self.food_direction(environment)

        food_direction_state = sum([bit << i for i, bit in enumerate(food_directions)])
        state = (right_side << 10) | (top_half << 9) | (enough_energy << 8) | food_direction_state
        return state
    

    def is_food_nearby(self, environment):
        for centroid in environment.food_centroids:
            distance = np.linalg.norm(self.position - centroid)
            if distance < self.size * 20:
                return True
        return False
    
    def calculate_reward(self, environment):
        reward = 0

        food_consumed_reward = 1
        if self.is_food_nearby(environment):
            initial_energy = self.energy
            self.consume_food(environment)
            if self.energy > initial_energy:
                reward += food_consumed_reward

        reproduction_reward = 50
        reproduction_energy_threshold = starting_energy + 20
        if self.energy >= reproduction_energy_threshold:
            reward += reproduction_reward

        return reward
    
    def act(self, environment):
        direction = self.move(environment)
        self.consume_food(environment)
        self.age += 1
        self.energy -= self.energy_decay

    def consume_food(self, environment):
        if np.random.rand() < chance_to_skip_food:
            return
        for i, centroid in enumerate(environment.food_centroids):
            centroid_size = np.sqrt(environment.max_food_centroid / np.pi)
            distance = np.linalg.norm(self.position - centroid)
            if distance < centroid_size and environment.food_amounts[i] > 0:
                self.energy += energy_gain_food
                environment.food_amounts[i] -= energy_gain_food

    def mutate_q_table(self):
        mutation = np.random.normal(0, self.mutation_rate, self.q_table.shape)
        self.q_table += mutation
        self.q_table = np.clip(self.q_table, -2, 2)
        
    def reproduce(self):
        offspring = []
        if self.energy >= (starting_energy + 20) and self.position[0] > self.environment_size / 2:
            num_offspring = random.randint(num_offspring_range[0], num_offspring_range[1])
            for _ in range(num_offspring):
                new_phenotype = self.phenotype
                transition_mutation = np.random.uniform(-self.mutation_rate, self.mutation_rate, self.transition_matrix.shape)
                new_transition_matrix = np.clip(self.transition_matrix + transition_mutation, 0, None)

                row_sums = new_transition_matrix.sum(axis=1, keepdims=True)
                new_transition_matrix = new_transition_matrix / row_sums

                self.mutate_q_table()
                offspring.append(Agent(new_phenotype, self.environment_size, self.mutation_rate, new_transition_matrix, np.copy(self.q_table)))
            self.energy -= energy_cost_reproduction
        return offspring
    
    def calculate_entropy(self):
        normalized_matrix = self.transition_matrix / (self.transition_matrix.sum(axis=1, keepdims=True) + 1e-9)
        entropy_per_row = -np.sum(normalized_matrix * np.log(normalized_matrix + 1e-9), axis=1)
        total_entropy = np.sum(entropy_per_row)
        return total_entropy

    def calculate_max_entropy(self):
        num_states = self.transition_matrix.shape[1]
        max_entropy = np.log(num_states)
        return max_entropy * self.transition_matrix.shape[0]

    def initialize_flagella_positions(self):
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        return angles        

# WARNING: comment the below out if you want to run my parameter sweep in main.
environment = Environment(food_regeneration_rate, num_food_centroids, agent_max_lifespan, reproduction_cycle,
                          simulation_steps, environment_size, max_food_centroid, initial_agent_count, mutation_rate)
environment.initialize_agents(mutation_rate, agent_size_range, agent_flagella_range)

fig1, ax1 = plt.subplots(figsize=(8, 8))
frames = []

frame_pause_duration = 0.01

for step in range(simulation_steps):
    environment.step()
    if step % 1 == 0:
        environment.visualize(ax1)
        plt.pause(frame_pause_duration)

        if save_gif:
            frame_filename1 = f'frame_env_{step}.png'
            fig1.savefig(frame_filename1)
            frames.append(frame_filename1)

if save_gif:
    with imageio.get_writer('test.gif', mode='I', duration=gif_duration) as writer:
        for filename in frames:
            frame = imageio.imread(filename)
            writer.append_data(frame)
            os.remove(filename)
    print("'test.gif' created")