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


def extract_pitch_vector(melody, w, d):
    # preprocess data
    melody_cleaned = remove_rests(convert_to_one_pitch(melody))

    pitch_vectors = []

    for i in range(0, len(melody_cleaned) - w + 1, d):
        window = melody_cleaned[i:i+w]
        p_i = np.zeros((w))
        for j in range(w):
            nonzero = window[j].nonzero()[0]
            if len(nonzero) == 0:
                p_i[j] = 0
            else:
                p_i[j] = nonzero[0]
        # normalize pitch vector
        p_i -= round(np.mean(p_i))
        pitch_vectors.append(p_i)

    return np.array(pitch_vectors)

# pitch_vectors = extract_pitch_vector(result_array, 500, 250)

def melodic_similarity(p1, p2):
    dist = np.linalg.norm(p1 - p2)
    return dist

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

def extract_vocals(input_file, output_filepath):
    inference.main(input_file, output_filepath)

def build_database(audio_dir, output_dir, w, d):

    print("extracting melodies...")
    for infile in glob.iglob(audio_dir + '/*'):
        song_name = (infile.split('/')[-1]).split('.')[0]
        vfile = output_dir + '/' + str(song_name) + ".wav"
        outfile = output_dir + '/' + str(song_name) + ".mid"
        # extract midi melody and add to output_dir directory
        extract_vocals(infile, vfile)
        extract_midi.audio_to_midi_melodia(vfile, outfile, 100)
        os.remove(vfile)
    
    # build array of pitch vectors and song names
        
    pitch_vectors = []
    song_names = []

    print("generating pitch vectors...")
    for m in glob.iglob(output_dir + '/*'):
        song_name = (m.split('/')[-1]).split('.')[0]
        print(song_name)
        midi_data = mido.MidiFile(m, clip=True)
        midi_array = mid2arry(midi_data)
        pitch_vectors.append(extract_pitch_vector(midi_array, w, d))
        song_names.append(song_name)

    print("saving pitch vectors...")
    for i in range(len(pitch_vectors)):
        song_name = song_names[i]
        np.save('saved/pitch_vectors/' + str(song_name) + '.npy', np.array(pitch_vectors[i]))
    np.save('saved/song_name_database.npy', np.array(song_names))
    print("complete!")

def query(hum_audio, w, d):
    # takes in a hummed audio and returns the top 10 closest songs by pitch vector similarity

    song_name_database = np.load('saved/song_name_database.npy')

    pitch_vector_database = []

    for song in song_name_database:
        pitch_vector_database.append(np.load('saved/pitch_vectors/' + str(song) + '.npy', allow_pickle=True))

    # convert to midi audio
    song_name = (hum_audio.split('/')[-1]).split('.')[0]
    print(song_name)
    extract_midi.audio_to_midi_melodia(hum_audio, 'queries/' + str(song_name) + ".mid", 100)
    midi_data = mido.MidiFile('queries/' + str(song_name) + ".mid", clip=True)
    midi_array = mid2arry(midi_data)
    query_pitch_vectors = extract_pitch_vector(midi_array, w, d)

    # compute similarity:
    min_distances = []
    for p in pitch_vector_database:
        d = []
        for q in query_pitch_vectors:
            dist = np.linalg.norm(p - q, axis=1)
            # closest_indices = np.argsort(dist)[:5]
            d.append(min(dist))
        min_distances.append(np.mean(d))
        
    num_to_return = min(10, len(song_name_database))
    closest_indices = np.argsort(min_distances)[:num_to_return]
    closest_songs = song_name_database[closest_indices]

    # print results
    for i in range(len(closest_songs)):
        print(str(i + 1) + ": " + str(closest_songs[i]) + ", distance = " + str(min_distances[closest_indices[i]]))
    print("----------")

    # return min_distances[closest_indices], closest_songs

# TEST!

w, d = 5000, 2500
# build_database('song_database', 'midi_database', w, d)
# query("song_database/Let It Be - The Beatles.mp3", w, d)
# query("song_database/Titi Me Pregunto - Bad Bunny.mp3", w, d)
# query("song_database/Love Story - Taylor Swift.mp3", w, d)
# query("Love Story (Taylor's Version) - Taylor Swift.mp3", w, d)
# print("LET IT BE TESTS:")
# query("Let It Be Hum 1.mp3", w, d)
# query("Let It Be Hum 2.mp3", w, d)
# query("Let It Be Piano 1.mp3", w, d)
# query("Let It Be Piano 2.mp3", w, d)

# print("LOVE STORY TESTS:")
# query("Love Story Hum 1.mp3", w, d)
# query("Love Story Hum 2.mp3", w, d)
# query("Love Story Piano 1.mp3", w, d)

query('New Recording 15.mp3', w, d)


# empty query data after use
# files = glob.glob('queries/*')
# for f in files:
#     os.remove(f)

# extract_vocals('Let It Be - The Beatles.mp3', '/Users/naveen/cs1952q/query-by-humming/vocal_tests/Let It Be - The Beatles [v].wav')