import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import librosa
from vocal_remover import inference
import crepe
from librosa.display import specshow
import warnings
from mlsh import MLSH
from tqdm import tqdm

class QBH:
    def __init__(self, w, d, b, r, sz, song_database):
        self.w = w
        self.d = d
        self.b = b
        self.r = r
        self.sz = sz
        self.song_database = song_database
        pitch_vector_info = np.load('saved/pitch_vectors.npy', allow_pickle=True)
        self.mlsh = MLSH(self.w, self.b, self.r, pitch_vector_info)

    def extract_pitch_vector(self, melody, mod_tempo = False):

        melody_cleaned = np.copy(melody)
        tempo_modifiers = [1.0]

        if mod_tempo:
            tempo_modifiers = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]

        pitch_vectors = []

        for t in tempo_modifiers:

            if t == 1.0: m = melody_cleaned
            else: m = self.stretch_or_squeeze_melody(melody_cleaned, t)

            for i in range(0, len(m) - self.w + 1, self.d):
                window = np.copy(m[i:i+self.w])
                window -= round(np.mean(window))
                pitch_vectors.append(window)

        return np.array(pitch_vectors)

    def extract_midi_notes_nn(self, audio_file, step_size=10, plot=False):
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

    def extract_vocals(self, input_file, output_filepath):
        inference.main(input_file, output_filepath)

    def build_database(self, remove_vocals=True):

        print("extracting melodies...")

        notes_and_names = []

        for infile in glob.iglob(self.song_database + '/*'):
            song_name = (infile.split('/')[-1]).split('.')[0]
            vfile = self.song_database + '/' + str(song_name) + ".wav"
            # extract midi melody and add to output_dir directory
            if remove_vocals and song_name != "Let It Be Piano 1":
                self.extract_vocals(infile, vfile)
            else:
                vfile = infile
            # extract_midi.audio_to_midi_melodia(infile, outfile, 100)
            notes = self.extract_midi_notes_nn(vfile, step_size=self.sz)
            notes_and_names.append((notes, song_name))
            if remove_vocals and song_name != 'Let It Be Piano 1': os.remove(vfile)
        
        # build array of pitch vectors and song names
            
        pitch_vectors = []

        print("generating pitch vectors...")
        for i in range(len(notes_and_names)):
            print(notes_and_names[i][1])
            notes = notes_and_names[i][0]
            pitch_vec = self.extract_pitch_vector(notes)
            for j in range(len(pitch_vec)):
                pitch_vectors.append([pitch_vec[j], notes_and_names[i][1], j*self.d])

        print("saving pitch vectors...")
        np.save('saved/pitch_vectors.npy', np.array(pitch_vectors))
        print("complete!")

        print("setting up MLSH module")
        self.mlsh = MLSH(self.d, self.b, self.r, np.array(pitch_vectors))

    def stretch_or_squeeze_melody(self, melody, ratio):    
        
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

    def query(self, hum_audio, plot=False, use_mlsh=True):
        # takes in a hummed audio and returns the top 10 closest songs by pitch vector similarity

        pitch_vector_info = np.load('saved/pitch_vectors.npy', allow_pickle=True)

        notes = self.extract_midi_notes_nn(hum_audio, step_size=self.sz)
        query_pitch_vectors = self.extract_pitch_vector(notes, mod_tempo=False)

        closest_songs = {}

        k = 5

        for i in range(len(query_pitch_vectors)):
            query_vector = query_pitch_vectors[i]

            closest = []

            if use_mlsh:
                query_mlsh = [query_vector, "query", 0]
                closest = np.array(self.mlsh.query(query_mlsh, k))
                closest_songs[tuple(query_vector)] = (closest[:,1:], i)
            else:
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
        ranked_songs = []
        for i in range(len(final_scores)):
            song, dist = final_scores[i]
            ranked_songs.append(song)
            print(str(i+1) + ": " + str(song) + ", Distance: " + str(dist))

        return ranked_songs

    def plot_pitch_vectors(self, audio_file, num_to_print):
        n = self.extract_midi_notes_nn(audio_file, plot=True)
        p = self.extract_pitch_vector(n)
        _, ax = plt.subplots(num_to_print + 1)
        for i in range(num_to_print):
            ax[i].plot(range(len(p[i])), p[i], marker='.', markersize=1, linestyle='')
        ax[-1].plot(range(len(n)), n, marker='.', markersize=1, linestyle='')
        plt.show()

    def test(self, test_data, labels, use_mlsh=True):
        top1 = 0
        top3 = 0
        top5 = 0
        n = len(test_data)
        for i in range(n):
            test_audio = test_data[i]
            label = labels[i]

            rankings = self.query(test_audio, use_mlsh=use_mlsh)

            try:
                idx = rankings.index(label)
                if idx == 0:
                    top1 += 1
                if idx < 3:
                    top3 += 1
                if idx < 5:
                    top5 += 1
            except:
                pass
        
        return top1/n, top3/n, top5/n

def main():
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

    sz = 50
    w = round(6 * 1000 / sz)
    d = round(1.25 * 1000 / sz)
    b = 50
    r = 50

    qbh = QBH(w, d, b, r, sz, 'song_database')
    # qbh.build_database()
    # # build_database('song_database', 'midi_database', w, d, sz, remove_vocals=True)
    # print("LET IT BE TESTS:")
    # qbh.query("Let It Be Hum 1.mp3")
    # qbh.query("Let It Be Hum 2.mp3")
    # qbh.query("Let It Be Piano 2.mp3")
    # print("LOVE STORY TESTS:")
    # qbh.query("Love Story Hum 1.mp3")
    # qbh.query("Love Story Hum 2.mp3")
    # qbh.query("Love Story Piano 1.mp3")
    # print("LOLA TEST:")
    # qbh.query('New Recording 15.mp3')

    test_data = ["Let It Be Hum 1.mp3", "Let It Be Hum 2.mp3", "Let It Be Piano 2.mp3", "Love Story Hum 1.mp3", "Love Story Hum 2.mp3", "Love Story Piano 1.mp3", "New Recording 15.mp3"]
    labels = ["Let It Be - The Beatles", "Let It Be - The Beatles", "Let It Be - The Beatles", "Love Story - Taylor Swift", "Love Story - Taylor Swift", "Love Story - Taylor Swift", "Titi Me Pregunto - Bad Bunny"]

    top1, top3, top5 = qbh.test(test_data, labels)
    top1b, top3b, top5b = qbh.test(test_data, labels, use_mlsh=False)

    print("BRUTE FORCE:")
    print("TOP 1 ACCURACY: " + str(top1b * 100) + '%')
    print("TOP 3 ACCURACY: " + str(top3b * 100) + '%')
    print("TOP 5 ACCURACY: " + str(top5b * 100) + '%')
    print("MLSH:")
    print("TOP 1 ACCURACY: " + str(top1 * 100) + '%')
    print("TOP 3 ACCURACY: " + str(top3 * 100) + '%')
    print("TOP 5 ACCURACY: " + str(top5 * 100) + '%')

    # notes = extract_midi_notes_nn("Let It Be Hum 1.mp3", step_size=sz)
    # query_pitch_vector = extract_pitch_vector(notes, w, d, mod_tempo=False)[0]
    # pitch_vector_info = np.load('saved/pitch_vectors.npy', allow_pickle=True)


    # # BRUTE FORCE

    # print("BRUTE FORCE:")
    # closest_brute = []

    # spread = max(query_pitch_vector) - min(query_pitch_vector)

    # for song_vector_info in pitch_vector_info:
    #     song_vector = np.array(song_vector_info[0])
    #     song_name = song_vector_info[1]
    #     song_index = song_vector_info[2]

    #     distance = np.linalg.norm(query_pitch_vector - song_vector) / spread
    #     closest_brute.append((song_name, song_index, distance, song_vector))

    # closest_brute.sort(key=lambda x: x[2])

    # for i in range(10):
    #     c = closest_brute[i]
    #     print(str(i+1) + ": " + str(c[0]) + ", " + str(c[1]) + ", " + str(c[2]))

    # # MLSH

    # print("MLSH:")

    # query_pitch_vector = [query_pitch_vector, "query", 0]
    # mlsh = MLSH(w, 50, 50, pitch_vector_info)

    # closest_mlsh = mlsh.query(query_pitch_vector, 10)

    # for i in range(len(closest_mlsh)):
    #     c = closest_mlsh[i]
    #     print(str(i+1) + ": " + str(c[1]) + ", " + str(c[2]) + ", " + str(c[3]))

if __name__ == "__main__":
    main()