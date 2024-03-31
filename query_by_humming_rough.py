import mido
import numpy as np
import string
import matplotlib.pyplot as plt
import extract_midi_melody as extract_midi
import glob
from tqdm import tqdm
import os
import librosa
import soundfile as sf
import vocal_remover
from vocal_remover import inference, lib
import time
import crepe
import math
from librosa.display import specshow

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

# mid = mido.MidiFile('midi/wii-sports-resort-main-theme-piano-solo.midi', clip=True)

def msg2dict(msg):
    result = dict()
    if 'note_on' in msg:
        on_ = True
    elif 'note_off' in msg:
        on_ = False
    else:
        on_ = None
    result['time'] = int(msg[msg.rfind('time'):].split(' ')[0].split('=')[1].translate(
        str.maketrans({a: None for a in string.punctuation})))

    if on_ is not None:
        for k in ['note', 'velocity']:
            result[k] = int(msg[msg.rfind(k):].split(' ')[0].split('=')[1].translate(
                str.maketrans({a: None for a in string.punctuation})))
    return [result, on_]

def switch_note(last_state, note, velocity, on_=True):
    # piano has 88 notes, corresponding to note id 21 to 108, any note out of this range will be ignored
    result = [0] * 88 if last_state is None else last_state.copy()
    if 21 <= note <= 108:
        result[note-21] = velocity if on_ else 0
    return result

def get_new_state(new_msg, last_state):
    new_msg, on_ = msg2dict(str(new_msg))
    new_state = switch_note(last_state, note=new_msg['note'], velocity=new_msg['velocity'], on_=on_) if on_ is not None else last_state
    return [new_state, new_msg['time']]
def track2seq(track):
    # piano has 88 notes, corresponding to note id 21 to 108, any note out of the id range will be ignored
    result = []
    last_state, last_time = get_new_state(str(track[0]), [0]*88)
    for i in range(1, len(track)):
        new_state, new_time = get_new_state(track[i], last_state)
        if new_time > 0:
            result += [last_state]*new_time
        last_state, last_time = new_state, new_time
    return result

def mid2arry(mid, min_msg_pct=0.1):
    tracks_len = [len(tr) for tr in mid.tracks]
    min_n_msg = max(tracks_len) * min_msg_pct
    # convert each track to nested list
    all_arys = []
    for i in range(len(mid.tracks)):
        if len(mid.tracks[i]) > min_n_msg:
            ary_i = track2seq(mid.tracks[i])
            all_arys.append(ary_i)
    # make all nested list the same length
    max_len = max([len(ary) for ary in all_arys])
    for i in range(len(all_arys)):
        if len(all_arys[i]) < max_len:
            all_arys[i] += [[0] * 88] * (max_len - len(all_arys[i]))
    all_arys = np.array(all_arys)
    all_arys = all_arys.max(axis=0)
    # trim: remove consecutive 0s in the beginning and at the end
    sums = all_arys.sum(axis=1)
    ends = np.where(sums > 0)[0]
    return all_arys[min(ends): max(ends)]

# result_array = mid2arry(mid)

def convert_to_one_pitch(melody):
    for i in range(len(melody)):
        t = melody[i]
        if sum(t) > 0:
            indices = t.nonzero()[0]
            new_t = np.zeros(t.shape)
            # get highest velocity note
            idx = indices[np.argmax(t[indices][::-1])]
            new_t[idx] = t[idx]
            melody[i] = new_t      
    return melody

def remove_short_notes(melody, threshold):
    last_note = (0, 0, 0)
    duration = 0
    for i in range(len(melody)):
        t = melody[i]
        if len(t.nonzero()[0]) > 0:
            note_idx = (t.nonzero()[0])[0]
            if note_idx != last_note[1] and abs(i - last_note[0]) > 1:
                if duration < threshold:
                    melody[last_note[0]:i] = np.zeros((i-last_note[0], 88))
                duration = 0
                last_note = (i, note_idx, melody[i, note_idx])
            else:
                duration += 1
    return melody

def remove_rests(melody):
    last_note = (0, 0, 0)
    start_index = None
    for i in range(len(melody)):
        t = melody[i]
        if np.sum(t) != 0:
            # note:
            if start_index is not None:
                # Found end of rest segment
                end_index = i - 1
                melody[start_index:end_index + 1, note_idx] = note_vel
                start_index = None
            note_idx = (t.nonzero()[0])
            last_note = (i, note_idx, melody[i, note_idx])
        elif start_index is None:
            # start of rest segment
            start_index = i
            j, note_idx, note_vel = last_note
    
    # Handle the case where the rest segment extends till the end of the melody
    if start_index is not None:
        end_index = len(melody) - 1
        melody[start_index:end_index + 1, note_idx] = note_vel
        
    return melody


# def extract_pitch_vector(melody, w, d, mod_tempo = False):
#     # preprocess data
#     # melody_cleaned = remove_short_notes(convert_to_one_pitch(melody), 100)
#     # melody_cleaned = remove_rests(convert_to_one_pitch(melody))
#     melody_cleaned = melody

#     tempo_modifiers = [1.0]

#     if mod_tempo:
#         tempo_modifiers = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]

#     pitch_vectors = []

#     for t in tempo_modifiers:

#         if t == 1.0: m = melody_cleaned
#         else: m = stretch_or_squeeze_melody(melody_cleaned, t)

#         for i in range(0, len(m) - w + 1, d):
#             window = m[i:i+w]
#             p_i = np.zeros((w))
#             for j in range(w):
#                 nonzero = window[j].nonzero()[0]
#                 if len(nonzero) == 0:
#                     p_i[j] = 0
#                 else:
#                     p_i[j] = nonzero[0]
#             # normalize pitch vector
#             print(p_i)
#             p_i -= round(np.mean(p_i))
#             print(p_i)
#             pitch_vectors.append(p_i)

#     print(pitch_vectors)

#     return np.array(pitch_vectors)

def extract_pitch_vector(melody, w, d, mod_tempo = False):
    # preprocess data
    # melody_cleaned = remove_short_notes(convert_to_one_pitch(melody), 100)
    # melody_cleaned = remove_rests(convert_to_one_pitch(melody))
    melody_cleaned = np.copy(melody)

    tempo_modifiers = [1.0]

    if mod_tempo:
        tempo_modifiers = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]

    pitch_vectors = []

    for t in tempo_modifiers:

        if t == 1.0: m = melody_cleaned
        else: m = stretch_or_squeeze_melody(melody_cleaned, t)

        for i in range(0, len(m) - w + 1, d):
            window = np.copy(m[i:i+w])
            window -= round(np.mean(window))
            pitch_vectors.append(window)

    return np.array(pitch_vectors)

# pitch_vectors = extract_pitch_vector(result_array, 500, 250)

# IMPLEMENT NN SEARCH WITH LOCALITY SENSITIVE HASHING

# fig, ax = plt.subplots(4)
# ax[0].plot(range(500), pitch_vectors[0], marker='.', markersize=1, linestyle='')
# ax[1].plot(range(500), pitch_vectors[1], marker='.', markersize=1, linestyle='')
# ax[2].plot(range(500), pitch_vectors[2], marker='.', markersize=1, linestyle='')
# ax[3].plot(range(500), pitch_vectors[3], marker='.', markersize=1, linestyle='')
# plt.show()

# result_array = result_array
# fig, ax = plt.subplots(4)
# ax[0].plot(range(result_array.shape[0]), np.multiply(np.where(result_array>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
# result_array_one = convert_to_one_pitch(result_array)
# ax[1].plot(range(result_array_one.shape[0]), np.multiply(np.where(result_array_one>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
# result_array_cleaned = remove_short_notes(result_array_one, 100)
# ax[2].plot(range(result_array_cleaned.shape[0]), np.multiply(np.where(result_array_cleaned>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
# result_array_trimmed = remove_rests(result_array_cleaned)
# ax[3].plot(range(result_array_trimmed.shape[0]), np.multiply(np.where(result_array_trimmed>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
# plt.show()
    
# result_array = result_array[0:10000]
# fig, ax = plt.subplots(4)
# ax[0].plot(range(result_array.shape[0]), np.multiply(np.where(result_array>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
# result_array_cleaned = remove_short_notes(result_array, 100)
# ax[1].plot(range(result_array_cleaned.shape[0]), np.multiply(np.where(result_array_cleaned>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
# result_array_trimmed = remove_rests(result_array_cleaned)
# ax[2].plot(range(result_array_trimmed.shape[0]), np.multiply(np.where(result_array_trimmed>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
# result_array_one = convert_to_one_pitch(result_array_trimmed)
# ax[3].plot(range(result_array_one.shape[0]), np.multiply(np.where(result_array_one>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
# plt.show()

# model_output, midi_data, note_events = predict('vocals/divenire.mp3')
# # save output (for testing purposes)
# predict_and_save(
#     'vocals/divenire.mp3',
#     'midi',
#     True,
#     True,
#     False,
#     False)
#

# Love Story Testing:
# midi_data1 = mido.MidiFile("Love Story (Taylor's Version) - Taylor Swift.mid", clip=True)
# midi_data2 = mido.MidiFile("midi_database/Love Story - Taylor Swift.mid", clip=True)
# my_midi_array = mid2arry(midi_data1)
# my_midi_array_2 = mid2arry(midi_data2)
# my_pitch_vectors_1 = extract_pitch_vector(my_midi_array, 100000, 50000)
# my_pitch_vectors_2 = np.load("saved/pitch_vectors/Love Story - Taylor Swift.npy")

# fig, ax = plt.subplots(4)
# ax[0].plot(range(my_midi_array.shape[0]), np.multiply(np.where(my_midi_array>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
# ax[1].plot(range(my_midi_array_2.shape[0]), np.multiply(np.where(my_midi_array_2>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
# ax[2].plot(range(len(my_pitch_vectors_1[0])), my_pitch_vectors_1[0], marker='.', markersize=1, linestyle='')
# ax[3].plot(range(len(my_pitch_vectors_2[0])), my_pitch_vectors_2[0], marker='.', markersize=1, linestyle='')
# plt.show()

def extract_midi_notes_nn(audio_file, step_size=10, plot=False):

    y, sr = librosa.load(audio_file)
    time, notes, confidence, _ = crepe.predict(y, sr, viterbi=True, step_size=step_size, model_capacity="medium")

    if plot:   
        specshow(librosa.amplitude_to_db(np.abs(librosa.cqt(y, hop_length=1024, sr=sr))**2))
        notes = librosa.hz_to_midi(notes + np.finfo(float).eps).round()
        plt.step(np.arange(len(notes)), 
                notes - librosa.note_to_midi('C1'), 
                marker='|',
                label='Melody')
        plt.title('CQT spectrogram with melody overlay')
        plt.legend()
        plt.show()
    
    notes = librosa.hz_to_midi(notes)
    return notes

def extract_vocals(input_file, output_filepath):
    inference.main(input_file, output_filepath)

def build_database(audio_dir, output_dir, w, d, sz, remove_vocals=True):

    print("extracting melodies...")

    notes_and_names = []
    song_names = []

    for infile in glob.iglob(audio_dir + '/*'):
        song_name = (infile.split('/')[-1]).split('.')[0]
        vfile = output_dir + '/' + str(song_name) + ".wav"
        outfile = output_dir + '/' + str(song_name) + ".mid"
        # # extract midi melody and add to output_dir directory
        if remove_vocals and song_name != "Let It Be Piano 1":
            extract_vocals(infile, vfile)
        else:
            vfile = infile
        # extract_midi.audio_to_midi_melodia(infile, outfile, 100)
        notes = extract_midi_notes_nn(vfile, step_size=sz)
        notes_and_names.append((notes, song_name))
        song_names.append(song_name)
        if remove_vocals and song_name != 'Let It Be Piano 1': os.remove(vfile)
    
    # build array of pitch vectors and song names
        
    pitch_vectors = []

    print("generating pitch vectors...")
    for i in range(len(notes_and_names)):
        print(notes_and_names[i][1])
        notes = notes_and_names[i][0]
        pitch_vec = extract_pitch_vector(notes, w, d)
        for j in range(len(pitch_vec)):
            pitch_vectors.append([pitch_vec[j], song_names[i], j*d])

    print("saving pitch vectors...")
    np.save('saved/pitch_vectors.npy', np.array(pitch_vectors))
    np.save('saved/song_name_database.npy', np.array(song_names))
    print("complete!")

def stretch_or_squeeze_melody(melody, ratio):    
    
    # determine distirbtion
    distribution = [] # (note, duration)

    prev = melody[0]
    cnt = 0
    for p in melody:
        if (p != prev).any():
            distribution.append((prev, cnt))
            cnt = 1
        else:
            cnt += 1
        prev = p
    distribution.append((prev, cnt))

    # recreate pitch vector
    new_pitch_vector = []

    for note, dur in distribution:
        new_duration = round(dur * ratio)
        for _ in range(new_duration):
            new_pitch_vector.append(note)

    return new_pitch_vector

def query(hum_audio, w, d, sz, plot=False):
    # takes in a hummed audio and returns the top 10 closest songs by pitch vector similarity

    song_name_database = np.load('saved/song_name_database.npy')
    pitch_vector_info = np.load('saved/pitch_vectors.npy', allow_pickle=True)

    notes = extract_midi_notes_nn(hum_audio, step_size=sz)
    query_pitch_vectors = extract_pitch_vector(notes, w, d, mod_tempo=False)

    closest_songs = {}

    k = 5

    for i in range(len(query_pitch_vectors)):
        query_vector = query_pitch_vectors[i]
        closest = []

        spread = max(query_vector) - min(query_vector)

        for song_vector_info in pitch_vector_info:
            song_vector = np.array(song_vector_info[0])
            song_name = song_vector_info[1]
            song_index = song_vector_info[2]

            distance = np.linalg.norm(query_vector - song_vector) / spread
            closest.append((song_name, song_index, distance, song_vector))

        closest.sort(key=lambda x: x[2])
        closest_songs[tuple(query_vector)] = (closest[:k], i)

    final_scores = {}

    for query_vector, closest_info in closest_songs.items():
        closest, i = closest_info
        for song_info in closest:
            if song_info[0] not in final_scores:
                final_scores[song_info[0]] = [0, 0]
            final_scores[song_info[0]][0] += song_info[2]
            final_scores[song_info[0]][1] += 1

    # for query_vector, closest_info in closest_songs.items():
    #     closest, i = closest_info
    #     if plot: print("For query pitch vector:", i)
    #     for song_info in closest:
    #         if plot: print(f"Song: {song_info[0]}, Index: {song_info[1]}, Distance: {song_info[2]}")
    #         closest_to_query.append((distance, song_info, i))
    #         if plot:
    #             fig, ax = plt.subplots(2)
    #             ax[0].plot(range(len(query_vector)), query_vector, marker='.', markersize=1, linestyle='')
    #             ax[1].plot(range(len(song_info[3])), song_info[3], marker='.', markersize=1, linestyle='')
    #             plt.show();
    #     if plot: print()

    
    # closest_to_query = sorted(closest_to_query, key=lambda x: x[0])

    # for k in range(10):
    #     _, song_info, i = closest_to_query[k]
    #     if song_info[0] not in final_scores:
    #         final_scores[song_info[0]] = [0, 0]
    #     final_scores[song_info[0]][0] += song_info[2]
    #     final_scores[song_info[0]][1] += 1

    max_i = 0
    max_d = 0
    for k, v in final_scores.items():
        d, i = v
        max_i = max(i, max_i)
        max_d = max(d, max_d)

    w1 = 0.90
    w2 = 0.10

    for k, v in final_scores.items():
        d, i = v
        d = (d/i) / max_d
        i = i / max_i
        final_scores[k] = (w1 * d - w2 * i + w2) * 10

    final_scores = sorted(final_scores.items(), key=lambda x:x[1])

    print("FINAL RANKINGS:")
    for i in range(len(final_scores)):
        song, dist = final_scores[i]
        print(str(i+1) + ": " + str(song) + ", Distance: " + str(dist))

    # # convert to midi audio
    # song_name = (hum_audio.split('/')[-1]).split('.')[0]
    # # extract_midi.audio_to_midi_melodia(hum_audio, 'queries/' + str(song_name) + ".mid", 100)
    # # midi_data = mido.MidiFile('queries/' + str(song_name) + ".mid", clip=True)
    # # midi_array = mid2arry(midi_data)
    # notes = extract_midi_notes_nn(hum_audio)
    # query_pitch_vectors = extract_pitch_vector(notes, w, d)

    # # compute similarity:
    # min_distances = []
    # for p in pitch_vector_database:
    #     d = []
    #     for q in query_pitch_vectors:
    #         dist = np.linalg.norm(p - q, axis=1)
    #         num_to_compare = round(len(query_pitch_vectors) * 0.75)
    #         closest_indices = np.argsort(dist)[:num_to_compare]
    #         d.append(np.mean(dist))
    #     d.sort()
    #     min_distances.append(np.mean(d[:-2]))
    
    # num_to_return = min(10, len(song_name_database))
    # closest_indices = np.argsort(min_distances)[:num_to_return]
    # closest_songs = song_name_database[closest_indices]

    # # print results
    # for i in range(len(closest_songs)):
    #     print(str(i + 1) + ": " + str(closest_songs[i]) + ", distance = " + str(min_distances[closest_indices[i]]))
    # print("----------")

    # best_candidates = []

def plot_pitch_vectors(audio_file, num_to_print):
    n = extract_midi_notes_nn(audio_file, plot=True)
    p = extract_pitch_vector(n, w, d)
    _, ax = plt.subplots(num_to_print + 1)
    for i in range(num_to_print):
        ax[i].plot(range(len(p[i])), p[i], marker='.', markersize=1, linestyle='')
    ax[-1].plot(range(len(n)), n, marker='.', markersize=1, linestyle='')
    plt.show()

    # for k in range(len(query_pitch_vectors)):
    #     q = query_pitch_vectors[k]
    #     d = []
    #     candidates = [[]]
    #     for i in range(len(pitch_vector_database)):
    #         p = pitch_vector_database[i]
    #         song = song_name_database[i]
    #         dist = np.linalg.norm(p - q, axis=1)
    #         closest_indices = np.argsort(dist)[:5]
    #         d = np.concatenate((d, dist[closest_indices]))
    #         print(len(d))
    #         c = p[closest_indices]
    #         c_with_label = []
    #         for j in range(len(c)):
    #             c_with_label.append((c[j], song, q, closest_indices[j], k))
    #         if i == 0:
    #             candidates = c_with_label
    #         else:
    #             candidates = np.concatenate((candidates, c_with_label))
    #     i = np.argsort(d)
    #     candidates = candidates[i]
    #     best_candidates.append((d[0], candidates[0]))
    #     best_candidates.append((d[1], candidates[1]))
    #     best_candidates.append((d[2], candidates[2]))
    #     best_candidates.append((d[3], candidates[3]))
    #     best_candidates.append((d[4], candidates[4]))
        
    # best_candidates.sort(key=lambda x: x[0])

    # print results
    # for i in range(len(best_candidates)):
    #     d, cand_info = best_candidates[i]
    #     print(str(i + 1) + ": " + str(cand_info[1]) + ", distance = " + str(d) + ", p = " + str(cand_info[0]) + ", q = " + str(cand_info[2]) + ", i = " + str(cand_info[3]) + ", k = " + str(cand_info[4]))
    # print("----------")

    # fig, ax = plt.subplots(4)
    # ax[0].plot(range(len(candidates[0][2])), candidates[0][2], marker='.', markersize=1, linestyle='')
    # ax[1].plot(range(len(candidates[0][0])), candidates[0][0], marker='.', markersize=1, linestyle='')
    # ax[2].plot(range(len(candidates[1][2])), candidates[1][2], marker='.', markersize=1, linestyle='')
    # ax[3].plot(range(len(candidates[1][0])), candidates[1][0], marker='.', markersize=1, linestyle='')
    # plt.show()


    # return min_distances[closest_indices], closest_songs

# TEST!


# w_vals = [1000, 2500, 5000, 10000]
# d_mults = [0.75, 0.5, 0.25, 0.1]

# for w in w_vals:
#     for d_m in d_mults:
# d = int(w*d_m)
    
# print("NEW TEST =========================")

sz = 50

w = round(6 * 1000 / sz)
d = round(1.25 * 1000 / sz)

print("w: " + str(w) + ", d: " + str(d))
# build_database('song_database_real', 'midi_database', w, d, sz, remove_vocals=True)
# query("HumScale.mp3", w, d)
# query("TwinkleHum.mp3", w, d)
# query("New Recording 23.mp3", w, d, plot=True)
# plot_pitch_vectors("New Recording 23.mp3", 3)
# print("LET IT BE TESTS:")
# query("Let It Be Hum 1.mp3", w, d, sz)
# query("Let It Be Hum 2.mp3", w, d, sz)
# query("song_database_real/Let It Be Piano 1.mp3", w, d, sz, plot=False)
# query("Let It Be Piano 2.mp3", w, d, sz)
# print("LOVE STORY TESTS:")
# query("Love Story Hum 1.mp3", w, d, sz, plot=False)
# query("Love Story Hum 2.mp3", w, d, sz)
# query("Love Story Piano 1.mp3", w, d, sz)
# print("LOLA TEST:")
query('New Recording 15.mp3', w, d, sz, plot=True)

# empty query data after use
# files = glob.glob('queries/*')
# for f in files:
#     os.remove(f)

# extract_vocals('Let It Be - The Beatles.mp3', '/Users/naveen/cs1952q/query-by-humming/vocal_tests/Let It Be - The Beatles [v].wav')

# compute similarity:
# song_name_database = np.load('saved/song_name_database.npy')

# pitch_vector_database = []

# for song in song_name_database:
#     pitch_vector_database.append(np.load('saved/pitch_vectors/' + str(song) + '.npy', allow_pickle=True))
# min_distances = []
# for p in pitch_vector_database:
#     d = []
#     for q in pitch_vector_database[1]:
#         dist = np.linalg.norm(p - q, axis=1)
#         # closest_indices = np.argsort(dist)[:5]
#         d.append(min(dist))
    
#     min_distances.append(np.mean(d))
    
# num_to_return = min(10, len(song_name_database))
# closest_indices = np.argsort(min_distances)[:num_to_return]
# closest_songs = song_name_database[closest_indices]

# # print results
# for i in range(len(closest_songs)):
#     print(str(i + 1) + ": " + str(closest_songs[i]) + ", distance = " + str(min_distances[closest_indices[i]]))
# print("----------")

# melody = mido.MidiFile('midi_database/Let It Be - The Beatles.mid', clip=True)
# melody = mido.MidiFile('midi_database/Love Story - Taylor Swift.mid', clip=True)
# mel2 = mido.MidiFile('queries/Let It Be Piano 1.mid', clip=True)
# m = mid2arry(melody)
# m2 = mid2arry(mel2)
# p = extract_pitch_vector(m, 10000, 2500)
# p3 = extract_pitch_vector(m2, 10000, 2500)
# p2 = extract_pitch_vector(m, 300000, 10000)

# print(np.linalg.norm(p3 - p[89]))
# print(np.linalg.norm(p3))

# fig, ax = plt.subplots(5)
# ax[0].plot(range(len(p[0])), p[0], marker='.', markersize=1, linestyle='')
# ax[1].plot(range(len(p2[0])), p2[0], marker='.', markersize=1, linestyle='')
# ax[2].plot(range(len(m)), np.multiply(np.where(m>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
# ax[3].plot(range(len(m2)), np.multiply(np.where(m2>0, 1, 0), range(1, 89)), marker='.', markersize=1, linestyle='')
# ax[4].plot(range(len(p3[13])), p3[13], marker='.', markersize=1, linestyle='')
# plt.show()
