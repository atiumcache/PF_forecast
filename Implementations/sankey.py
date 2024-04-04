import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualize_particles(num_of_particles, resampling_indices):
    plt.figure(figsize=(15, 10))

    def unique_ranks(arr):
        # Sort the array and get indices
        sort_order = np.argsort(arr)
        sorted_arr = arr[sort_order]
        
        # Initialize the rank array with zeros
        ranks = np.zeros_like(arr)
        
        # The first element in the sorted array gets the first rank
        current_rank = 0
        ranks[sort_order[0]] = current_rank
        
        # Iterate over the sorted array to assign ranks
        for i in range(1, len(arr)):
            # Increase the rank only if the current value is different from the previous
            if sorted_arr[i] != sorted_arr[i - 1]:
                current_rank += 1
            # Assign the next rank, making sure it's unique by adding the difference from 'i'
            ranks[sort_order[i]] = current_rank + (i - current_rank)
            
        return ranks


    # Print the first few elements of first few arrays for verification/debugging
    #for i, arr in enumerate(resampling_indices):
    #    print(f"Array {i+1}: {arr[:5]} ... {arr[num_of_particles-5:]}")


    # Create a positions array to sort the elements based on which previous particle
    # that they are resampled from.
    # This cleans up the visualization and prevents lines from crossing. 
        
    # We don't want to outright "sort" the index arrays themselves,
    # as this will provide false information as to where each particle came from. 
    print("Debugging Info:")
    print("Indices Length: ", len(resampling_indices))
    print("Particle Count: ", num_of_particles)

    positions = np.zeros_like(resampling_indices, dtype=int)
    positions[0] = np.arange(num_of_particles)  # Initial positions are just the particle indices

    for step in range(1, len(resampling_indices)):
        positions[step] = unique_ranks(resampling_indices[step])
    
    time_steps = 12
    for step in range(1, time_steps):
        for p in range(num_of_particles):
            plt.plot([step-1, step], [resampling_indices[step][p], positions[step, p]], linestyle='solid', color='blue', markersize=0.1, linewidth=10/num_of_particles)

    # TODO: Trace the ending particles back through the chart in red?

    plt.xlabel('Time Step')
    plt.ylabel('Particles')
    plt.title('Particle Filtering Visualization')
    plt.grid(True)
    plt.xticks(np.arange(time_steps))
    plt.show()