import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import librosa
from vocal_remover import inference
import crepe
from librosa.display import specshow
import warnings
from e2lsh import E2LSH

def extract_pitch_vector(melody, w, d, mod_tempo = False):

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

def extract_midi_notes_nn(audio_file, step_size=10, plot=False):

    y, sr = librosa.load(audio_file)
    _, notes, _, _ = crepe.predict(y, sr, viterbi=True, step_size=step_size, model_capacity="medium")

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

    for infile in glob.iglob(audio_dir + '/*'):
        song_name = (infile.split('/')[-1]).split('.')[0]
        vfile = output_dir + '/' + str(song_name) + ".wav"
        # extract midi melody and add to output_dir directory
        if remove_vocals and song_name != "Let It Be Piano 1":
            extract_vocals(infile, vfile)
        else:
            vfile = infile
        # extract_midi.audio_to_midi_melodia(infile, outfile, 100)
        notes = extract_midi_notes_nn(vfile, step_size=sz)
        notes_and_names.append((notes, song_name))
        if remove_vocals and song_name != 'Let It Be Piano 1': os.remove(vfile)
    
    # build array of pitch vectors and song names
        
    pitch_vectors = []

    print("generating pitch vectors...")
    for i in range(len(notes_and_names)):
        print(notes_and_names[i][1])
        notes = notes_and_names[i][0]
        pitch_vec = extract_pitch_vector(notes, w, d)
        for j in range(len(pitch_vec)):
            pitch_vectors.append([pitch_vec[j], notes_and_names[i][1], j*d])

    print("saving pitch vectors...")
    np.save('saved/pitch_vectors.npy', np.array(pitch_vectors))
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

def plot_pitch_vectors(audio_file, num_to_print, w, d):
    n = extract_midi_notes_nn(audio_file, plot=True)
    p = extract_pitch_vector(n, w, d)
    _, ax = plt.subplots(num_to_print + 1)
    for i in range(num_to_print):
        ax[i].plot(range(len(p[i])), p[i], marker='.', markersize=1, linestyle='')
    ax[-1].plot(range(len(n)), n, marker='.', markersize=1, linestyle='')
    plt.show()

def main():
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

    sz = 50
    w = round(6 * 1000 / sz)
    d = round(1.25 * 1000 / sz)

    print("w: " + str(w) + ", d: " + str(d))
    # build_database('song_database_real', 'midi_database', w, d, sz, remove_vocals=True)
    print("LET IT BE TESTS:")
    query("Let It Be Hum 1.mp3", w, d, sz)
    query("Let It Be Hum 2.mp3", w, d, sz)
    query("song_database_real/Let It Be Piano 1.mp3", w, d, sz, plot=False)
    query("Let It Be Piano 2.mp3", w, d, sz)
    print("LOVE STORY TESTS:")
    query("Love Story Hum 1.mp3", w, d, sz, plot=False)
    query("Love Story Hum 2.mp3", w, d, sz)
    query("Love Story Piano 1.mp3", w, d, sz)
    print("LOLA TEST:")
    query('New Recording 15.mp3', w, d, sz, plot=True)

if __name__ == "__main__":
    main()