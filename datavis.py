# DISCLAIMER:
#   I apologize for any redundant or unused code. Getting this to work was quite
#   the time committment. Being short on time, I had to get it working somehow
#   without accidentally breaking soemthing through deletion.
#
#   Run this to generate charts from the csv data.

import pandas as pd
import matplotlib.pyplot as plt

csv_file_path = 'combined_data.csv'
df = pd.read_csv(csv_file_path)

food_regen_rates = [0.25, 0.5, 0.75]
filtered_df = df[df['Food Regeneration Rate'].isin(food_regen_rates)]

mutation_rates = filtered_df['Mutation Rate'].unique()

for food_regen_rate in food_regen_rates:
    plt.figure(figsize=(10, 6))

    for mutation_rate in mutation_rates:
        subset = filtered_df[(filtered_df['Mutation Rate'] == mutation_rate) & 
                             (filtered_df['Food Regeneration Rate'] == food_regen_rate)]

        plt.plot(subset['Generation'], subset['Average Population'], label=f'Mutation Rate: {mutation_rate}')

    plt.xlabel('Generation')
    plt.ylabel('Population')
    plt.title(f'Population over Generations for Food Regeneration Rate {food_regen_rate}')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'population_food_regen_{food_regen_rate}.png')
    #plt.show() 

plt.close('all')
