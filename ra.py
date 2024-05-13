import numpy as np
import matplotlib.pyplot as plt
from midi_to_array import midi_to_array
# from query_by_humming import QBH

def horizontal_shift(vector, h):
    n = len(vector)
    shifted_vector = np.full_like(vector, np.nan)  # Create an array of nans with the same length as the input vector
    if h < 0:  # Shift left
        shifted_vector[:n+h] = vector[-h:]
    elif h > 0:  # Shift right
        shifted_vector[h:] = vector[:-h]
    else:  # No shift
        shifted_vector = vector.copy()
    return shifted_vector

def scale_vector(vector, ratio):
    new_length = round(len(vector) * ratio)
    old_length = len(vector)
    indices = np.linspace(0, old_length - 1, new_length)
    scaled_vector = np.interp(indices, np.arange(old_length), vector)
    diff = old_length - new_length
    if diff > 0:
        scaled_vector = np.concatenate((scaled_vector, (np.empty((diff)) * np.nan)))
    else:
        scaled_vector = scaled_vector[:old_length]
    return scaled_vector

def dynamic_time_warping(query, song, plot=False):
    # find optimal combination of horizontal shift (h), vertical shift(v), and scale factor (s) to minimize eulclidean distance

    @np.vectorize
    def compute_distance(h, s, v):
        # for a given h, v, s, the modified query vector is as follows
        mod_query = scale_vector(horizontal_shift(query + v, h), s)
        indices = ~np.isnan(mod_query)
        return np.linalg.norm(mod_query[indices] - song[indices]) - 10*(len(indices)/len(mod_query))

    h_range = range(-25, 25, 10)
    s_range = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    v_range = [-3, -2, -1, 0, 1, 2, 3]

    H, S, V = np.meshgrid(h_range, s_range, v_range, indexing='ij')

    results = compute_distance(H, S, V)
    best_dist = np.min(results)
    min_indices = np.unravel_index(np.argmin(results), results.shape)

    # Retrieve the corresponding parameter values using the indices
    min_h = h_range[min_indices[0]]
    min_s = s_range[min_indices[1]]
    min_v = v_range[min_indices[2]]

    if plot:

        scaled_query = scale_vector(horizontal_shift(query + min_v, min_h), min_s)

        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot the original query
        ax.plot(query, label='Original Query', marker='o')
        
        # Plot the song vector
        ax.plot(song, label='Song Vector', marker='x')
        
        # Plot the optimally scaled query
        ax.plot(scaled_query, label=f'Scaled Query', marker='s')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Pitch')
        ax.set_title('Dynamic Time Warping Alignment')
        ax.legend()
        
        plt.show()
    
    return best_dist, min_h, min_s, min_v

def optimize_split_scaling(query, song, min_h, min_s, min_v):
    # Apply initial transformations
    transformed_query = scale_vector(horizontal_shift(query + min_v, min_h), min_s)
    
    # Split the query into two halves
    mid_point = len(transformed_query) // 2
    first_half = transformed_query[:mid_point]
    second_half = transformed_query[mid_point:]
    
    # Define shift and scale ranges for testing each half
    h_range = np.arange(-5, 5, 5)  # example range for shifts
    s_range = [0.8, 0.9, 1.0, 1.1, 1.2]  # example scale factors
    
    # Initialize the best distance and parameters
    best_distance = np.inf
    best_params = (None, None, None, None)  # (first_half_h, first_half_s, second_half_h, second_half_s)
    
    # Test all combinations of shifts and scales for the two halves
    for h1 in h_range:
        for s1 in s_range:
            for h2 in h_range:
                for s2 in s_range:
                    # Scale and shift both halves
                    scaled_first_half = scale_vector(horizontal_shift(first_half, h1), s1)
                    scaled_second_half = scale_vector(horizontal_shift(second_half, h2), s2)
                    
                    # Combine the halves
                    combined_query = np.concatenate((scaled_first_half, scaled_second_half))
                    
                    # Calculate the distance
                    indices = ~np.isnan(combined_query)
                    distance = np.linalg.norm(combined_query[indices] - song[indices])
                    
                    # Update the best scales, shifts, and distance
                    if distance < best_distance:
                        best_distance = distance
                        best_params = (h1, s1, h2, s2)
    
    # Return the best parameters and distance
    best_first_half = scale_vector(horizontal_shift(first_half, best_params[0]), best_params[1])
    best_second_half = scale_vector(horizontal_shift(second_half, best_params[2]), best_params[3])
    best_scaled_query = np.concatenate((best_first_half, best_second_half))
    
    return best_distance, best_scaled_query

def dtw_with_split_scaling(query, song, plot=False):
    # Perform initial DTW
    initial_dist, min_h, min_s, min_v = dynamic_time_warping(query, song)
    initial_scaled_query = scale_vector(horizontal_shift(query + min_v, min_h), min_s)
    
    # Perform split and optimize scaling
    optimized_dist, optimized_query = optimize_split_scaling(query, song, min_h, min_s, min_v)
    
    # Plotting
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(song, label='Song Vector', color='black', marker='x', linewidth=2)
        plt.plot(query, label='Original Query', color='gray', linestyle='-', marker='o', alpha=0.7)
        plt.plot(initial_scaled_query, label='Initial DTW Scaled Query', color='blue', linestyle='--')
        plt.plot(optimized_query, label='Optimally Split Scaled Query', color='red', linestyle='-.')
        plt.title('Comparison of Original, Initial DTW, and Optimized Split Scaled Queries with Song Vector')
        plt.xlabel('Time')
        plt.ylabel('Pitch')
        plt.legend()
        plt.show()
    
    return optimized_dist


# QBH Test
sz = 50
w = round(6 * 1000 / sz)
d = round(3 * 1000 / sz)
b = 50
r = 50
# midi_qbh = QBH(w, d, b, r, sz, "MIR-QBSH/midiFile")

# # song_array = midi_qbh.extract_midi_notes_nn("MIR-QBSH/midimp3s/00035.mp3", step_size=sz, plot=True)
# song_array = midi_to_array("MIR-QBSH/midiFile/00020.mid")
# song_vectors = midi_qbh.extract_pitch_vector(song_array)
# # song_array2 = midi_qbh.extract_midi_notes_nn("MIR-QBSH/midimp3s/00032.mp3", step_size=sz)
# # song_vectors2 = midi_qbh.extract_pitch_vector(song_array2)
# query_array = midi_qbh.extract_midi_notes_nn("MIR-QBSH/waveFile/year2003/person00002/00020.wav", step_size=sz)
# query_vectors = midi_qbh.extract_pitch_vector(query_array)

# dist = dtw_with_split_scaling(query_vectors[0], song_vectors[0], plot=True)
# print("Final distance for song 1: " + str(dist))

# dist = dtw_with_split_scaling(query_vectors[0], song_vectors[1], plot=True)
# print("Final distance for song 2: " + str(dist))

# dist = dtw_with_split_scaling(query_vectors[0], song_vectors2[0], plot=True)
# print("Final distance for song 3: " + str(dist))

# dist = dtw_with_split_scaling(query_vectors[0], song_vectors2[1], plot=True)
# print("Final distance for song 4: " + str(dist))

# # fig, ax = plt.subplots(figsize=(12, 6))
        
# # # Plot the song vector
# # ax.plot(song_array, label='Song Vector', marker='x')
# # ax.plot(song_array2, label='Song Vector', marker='x')

# # plt.show()
        