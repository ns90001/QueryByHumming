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
import tkinter as tk
from tkinter import ttk
from threading import Thread

from tkinter import messagebox
from tkinter import scrolledtext

import pyaudio
import wave
import pygame
from pygame import mixer

class QBH:
    def __init__(self, w, d, b, r, sz, song_database):
        self.w = w
        self.d = d
        self.b = b
        self.r = r
        self.sz = sz
        self.song_database = song_database
        
        pitch_vector_info = np.load('saved/pitch_vectors.npy', allow_pickle=True)
        # self.mlsh = MLSH(self.w, self.b, self.r, pitch_vector_info, pre_loaded_buckets=None)

    def extract_pitch_vector(self, melody, mod_tempo = False):

        melody_cleaned = np.copy(melody)
        tempo_modifiers = [1.0]

        if mod_tempo:
            tempo_modifiers = [0.9, 1.0, 1.1, 1.2]

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
        self.mlsh = MLSH(self.w, self.b, self.r, np.array(pitch_vectors))

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
        query_pitch_vectors = self.extract_pitch_vector(notes, mod_tempo=True)

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
                if plot:
                    print(f"Song: {song_info[0]}, Index: {song_info[1]}, Distance: {song_info[2]}")
                    fig, ax = plt.subplots(2)
                    ax[0].plot(range(len(query_vector)), query_vector, marker='.', markersize=1, linestyle='')
                    ax[1].plot(range(len(song_info[3])), song_info[3], marker='.', markersize=1, linestyle='')
                    plt.show();

        max_i = 0
        max_d = 0
        for k, v in final_scores.items():
            d, i = v
            max_i = max(i, max_i)
            max_d = max(d, max_d)

        w1 = 0.80
        w2 = 0.20

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
        top10 = 0
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
                if idx < 10:
                    top10 += 1
            except:
                pass
        
        return top1/n, top3/n, top5/n, top10/n

class AudioRecorder:
    def __init__(self, filename, qbh, duration=15, sample_rate=44100, chunk=1024):
        self.filename = filename
        self.duration = duration
        self.sample_rate = sample_rate
        self.chunk = chunk
        self.qbh = qbh

        self.audio = pyaudio.PyAudio()
        self.frames = []

        self.root = tk.Tk()
        self.root.title("Audio Recorder")
        self.root.geometry("600x400")
        self.root.configure(bg="#f0f0f0")

        self.label = tk.Label(self.root, text="Press Record to start recording.", bg="#f0f0f0", fg="black", font=("Helvetica", 12))
        self.label.pack(pady=10)

        self.record_button = ttk.Button(self.root, text="Record", command=self.record, style='Record.TButton')
        self.record_button.pack(pady=5)

        self.rerecord_button = ttk.Button(self.root, text="Re-record", command=self.rerecord, state=tk.DISABLED, style='Button.TButton')
        self.rerecord_button.pack(pady=5)

        self.play_button = ttk.Button(self.root, text="Play", command=self.play, state=tk.DISABLED, style='Button.TButton')
        self.play_button.pack(pady=5)

        self.stop_button = ttk.Button(self.root, text="Stop", command=self.stop, state=tk.DISABLED, style='Button.TButton')
        self.stop_button.pack(pady=5)

        self.submit_button = ttk.Button(self.root, text="Submit", command=self.submit, state=tk.DISABLED, style='Button.TButton')
        self.submit_button.pack(pady=5)

        self.result_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=60, height=10)
        self.result_text.pack(pady=10)

        self.loading_label = tk.Label(self.root, text="", bg="#f0f0f0", fg="black", font=("Helvetica", 10))
        self.loading_label.pack(pady=5)

        self.style = ttk.Style()
        self.style.configure('Record.TButton', foreground='black', background='red', font=("Helvetica", 12))
        self.style.configure('Button.TButton', foreground='black', background='#f0f0f0', font=("Helvetica", 12))

        self.root.mainloop()

    def record(self):
        self.record_button.config(state=tk.DISABLED)
        self.rerecord_button.config(state=tk.DISABLED)
        self.play_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.submit_button.config(state=tk.DISABLED)

        thread = Thread(target=self.record_audio)
        thread.start()

    def record_audio(self):
        try:
            self.stream = self.audio.open(format=pyaudio.paInt16, channels=1,
                                          rate=self.sample_rate, input=True,
                                          frames_per_buffer=self.chunk)

            self.label.config(text="Recording...", fg="red")

            for _ in range(0, int(self.sample_rate / self.chunk * self.duration)):
                data = self.stream.read(self.chunk)
                self.frames.append(data)

            self.label.config(text="Recording saved.", fg="green")
            self.record_button.config(state=tk.DISABLED)
            self.rerecord_button.config(state=tk.NORMAL)
            self.play_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.submit_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def rerecord(self):
        self.frames = []
        self.label.config(text="Press Record to start recording.", fg="black")
        self.record_button.config(state=tk.NORMAL)
        self.rerecord_button.config(state=tk.DISABLED)
        self.play_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.submit_button.config(state=tk.DISABLED)
        self.result_text.delete(1.0, tk.END)

    def play(self):
        self.label.config(text="Playing...", fg="blue")
        self.play_button.config(state=tk.DISABLED)

        thread = Thread(target=self.play_audio)
        thread.start()

    def play_audio(self):
        try:
            self.audio_output = pyaudio.PyAudio()
            self.play_stream = self.audio_output.open(format=self.audio.get_format_from_width(self.audio.get_sample_size(pyaudio.paInt16)),
                                                       channels=1,
                                                       rate=self.sample_rate,
                                                       output=True)
            for frame in self.frames:
                self.play_stream.write(frame)

            self.label.config(text="Playback complete.", fg="green")
            self.play_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def stop(self):
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()

            self.label.config(text="Recording saved.", fg="green")
            self.record_button.config(state=tk.DISABLED)
            self.rerecord_button.config(state=tk.NORMAL)
            self.play_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.submit_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def submit(self):
        self.loading_label.config(text="Processing...", fg="blue")
        self.record_button.config(state=tk.DISABLED)
        self.rerecord_button.config(state=tk.DISABLED)
        self.play_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        self.submit_button.config(state=tk.DISABLED)

        # Query the QBH class with the recorded audio
        closest_songs = self.query_qbh()

        # Clear existing items from the treeview
        for item in self.treeview.get_children():
            self.treeview.delete(item)

        # Display the closest song matches
        for idx, song in enumerate(closest_songs[:10], 1):
            # Insert song name and play/pause button into the treeview
            self.treeview.insert("", "end", values=(song, "Play"), tags=("play_button",))

        self.loading_label.config(text="", fg="black")

    def toggle_play_pause(self, event):
        item = self.treeview.selection()[0]
        song_name = self.treeview.item(item, "values")[0]
        button_text = self.treeview.item(item, "values")[1]

        # Get the corresponding MP3 file name
        mp3_file = f"{song_name}.mp3"

        if button_text == "Play":
            try:
                mixer.music.load(mp3_file)
                mixer.music.play()
                self.treeview.item(item, values=(song_name, "Pause"))
            except pygame.error as e:
                messagebox.showerror("Error", str(e))
        else:
            mixer.music.pause()
            self.treeview.item(item, values=(song_name, "Play"))

    def query_qbh(self):

        # Save the recorded audio to a WAV file
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        # Query the QBH class with the recorded audio
        closest_songs = self.qbh.query(self.filename, use_mlsh=False)

        # Delete the recorded audio file after use
        os.remove(self.filename)

        return closest_songs

def record_audio(filename, duration=15, sample_rate=44100, chunk=1024):
    """Record audio from microphone and save it to a WAV file."""
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=sample_rate, input=True,
                        frames_per_buffer=chunk)

    frames = []

    print("Recording...")
    for _ in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def live_test_qbh(qbh):
    # Set the path to store the recorded audio
    filename = "hum_audio.wav"

    # Record audio from microphone
    AudioRecorder(filename)

    # Query the QBH class with the recorded audio
    closest_songs = qbh.query(filename, use_mlsh=False)

    # Delete the recorded audio file after use
    os.remove(filename)

def main():
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

    sz = 50
    w = round(6 * 1000 / sz)
    d = round(1.25 * 1000 / sz)

    b_vals = [5, 10, 20, 50, 70]
    r_vals = [5, 10, 20, 50, 70]

    b = 50
    r = 50

    # for b in b_vals:
    #     for r in r_vals:
    #         print("b = " + str(b) + ", r = " + str(r))

    # pitch_vector_info = np.load('saved/pitch_vectors.npy', allow_pickle=True)
    # mlsh = MLSH(w, b, r, pitch_vector_info)
    # buckets = mlsh.get_buckets()
    # print(buckets)
    # np.save('saved/buckets.npy', buckets, allow_pickle=True)
    # buckets = np.load('saved/buckets.npy', allow_pickle=True)
    # print(buckets)
    qbh = QBH(w, d, b, r, sz, 'song_database_real')

    # qbh.build_database()
    # print(" == database succesfully built! == ")

    # test_data = [
    #     "Let It Be Hum 1.mp3", 
    #     "Let It Be Hum 2.mp3", 
    #     "Love Story Hum 1.mp3", 
    #     "Love Story Hum 2.mp3", 
    #     "Love Story Piano 1.mp3",
    #     "Let Her Go Hum 1.mp3",
    #     "Girls Like You Hum.mp3",
    #     "Just The Way You Are Hum.mp3",
    #     "All of Me Hum.mp3"]
    
    # labels = ["Let It Be - The Beatles", 
    #           "Let It Be - The Beatles", 
    #           "Love Story - Taylor Swift", 
    #           "Love Story - Taylor Swift", 
    #           "Love Story - Taylor Swift",
    #           " Let Her Go",
    #           " Girls Like You",
    #           " Just the Way You Are",
    #           " All of Me"]

    # # top1, top3, top5, top10 = qbh.test(test_data, labels)
    # top1b, top3b, top5b, top10b = qbh.test(test_data, labels, use_mlsh=False)

    # print("BRUTE FORCE:")
    # print("TOP 1 ACCURACY: " + str(top1b * 100) + '%')
    # print("TOP 3 ACCURACY: " + str(top3b * 100) + '%')
    # print("TOP 5 ACCURACY: " + str(top5b * 100) + '%')
    # print("TOP 10 ACCURACY: " + str(top10b * 100) + '%')
    # print("MLSH:")
    # print("TOP 1 ACCURACY: " + str(top1 * 100) + '%')
    # print("TOP 3 ACCURACY: " + str(top3 * 100) + '%')
    # print("TOP 5 ACCURACY: " + str(top5 * 100) + '%')
    # print("TOP 10 ACCURACY: " + str(top10 * 100) + '%')

    # qbh.query('Girls Like You Hum.mp3')
    # qbh.query('All of Me Hum.mp3')
    # qbh.query('7 Years Hum.mp3')
    # qbh.query('Just The Way You Are Hum.mp3')

    filename = 'hum_audio.wav'
    # pygame.init()
    mixer.init()
    AudioRecorder(filename, qbh)

if __name__ == "__main__":
    main()