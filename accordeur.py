import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

# Paramètres audio
Fs = 44100
N = 2048

notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def freq_vers_note(f):
    if f < 20:
        return "Silence", 0, 0
    h = round(12 * np.log2(f / 440.0))
    midi = h + 69
    nom_note = notes[midi % 12]
    octave = (midi // 12) - 1
    f_exacte = 440.0 * (2.0 ** ((midi - 69) / 12.0))
    cents = 1200 * np.log2(f / f_exacte) if f_exacte > 0 else 0
    return f"{nom_note}{octave}", f_exacte, cents

# Configuration du graphique en direct
plt.ion()
fig, ax = plt.subplots(figsize=(9, 4))
freqs = np.fft.rfftfreq(N, 1 / Fs)
ligne, = ax.plot(freqs, np.zeros(len(freqs)), color='purple')

ax.set_xlim(0, 2000)
ax.set_ylim(0, 400)
ax.set_title("Spectre Fréquentiel - TFD")
ax.set_xlabel("Fréquence (Hz)")
ax.set_ylabel("Amplitude")
ax.grid(True)

def callback_audio(indata, frames, time, status):
    if status:
        return
    
    signal = indata[:, 0]
    # Application de la fenêtre de Hamming
    signal_fenetre = signal * np.hamming(len(signal))
    
    # Calcul de la FFT
    fft_res = np.fft.rfft(signal_fenetre)
    mags = np.abs(fft_res)
    
    # Mise à jour du graphe
    ligne.set_ydata(mags)
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    # Détection du pic principal
    idx_max = np.argmax(mags)
    f_dom = freqs[idx_max]
    
    if mags[idx_max] > 20:
        note, f_ex, cents = freq_vers_note(f_dom)
        print(f"Freq : {f_dom:6.2f} Hz | Note : {note:3s} | Ecart : {cents:+5.1f} cents", end="\r")

print("Démarrage de l'accordeur... (Ctrl+C pour quitter)")

try:
    with sd.InputStream(channels=1, samplerate=Fs, blocksize=N, callback=callback_audio):
        plt.show(block=True)
except KeyboardInterrupt:
    print("\nArrêt du programme.")
