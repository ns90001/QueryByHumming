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
    s_range = [0.8, 0.9, 1.0, 1.1, 1.2]
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
    
    return best_dist

# QBH Test
sz = 50
w = round(6 * 1000 / sz)
d = round(3 * 1000 / sz)
b = 50
r = 50
midi_qbh = QBH(w, d, b, r, sz, "MIR-QBSH/midiFile")

# song_array = midi_qbh.extract_midi_notes_nn("MIR-QBSH/midimp3s/00035.mp3", step_size=sz, plot=True)
song_array = midi_to_array("MIR-QBSH/midiFile/00045.mid")
song_vectors = midi_qbh.extract_pitch_vector(song_array)
song_array2 = midi_qbh.extract_midi_notes_nn("MIR-QBSH/midimp3s/00032.mp3", step_size=sz)
song_vectors2 = midi_qbh.extract_pitch_vector(song_array2)
query_array = midi_qbh.extract_midi_notes_nn("MIR-QBSH/waveFile/year2003/person00004/00035.wav", step_size=sz)
query_vectors = midi_qbh.extract_pitch_vector(query_array)

dist = dynamic_time_warping(query_vectors[0], song_vectors[0], plot=True)
print("final distance = " + str(dist))
dist = dynamic_time_warping(query_vectors[0], song_vectors[1], plot=True)
print("final distance = " + str(dist))
dist = dynamic_time_warping(query_vectors[0], song_vectors2[0], plot=True)
print("final distance = " + str(dist))
dist = dynamic_time_warping(query_vectors[0], song_vectors2[1], plot=True)
print("final distance = " + str(dist))

# fig, ax = plt.subplots(figsize=(12, 6))
        
# # Plot the song vector
# ax.plot(song_array, label='Song Vector', marker='x')
# ax.plot(song_array2, label='Song Vector', marker='x')

# plt.show()
        