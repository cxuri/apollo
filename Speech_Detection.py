import sounddevice as sd
import numpy as np
import scipy.signal
import librosa
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import collections
import queue
import webrtcvad

# --- Hardware Simulation Configuration ---
SAMPLE_RATE = 16000
CHUNK_SIZE = 480  
N_MELS = 40
NUM_FRAMES = 100 

# 1. High-Pass Filter (Removes low-frequency thuds)
SOS_FILTER = scipy.signal.butter(N=2, Wn=300, btype='highpass', fs=SAMPLE_RATE, output='sos')

# 2. Vowel Gate Thresholds
VOWEL_ZCR_MAX = 0.15  # Vowels rarely cross 15% ZCR. Claps easily exceed 25%.

BUFFER_CHUNKS = 50 
audio_ring_buffer = collections.deque(maxlen=BUFFER_CHUNKS)

# Track ONLY confirmed vowels in the last 450ms
vad_history = collections.deque(maxlen=15)
vad_history.extend([False] * 15) 
hangover_timer = 0
KWS_HOLD_FRAMES = 50  

vad = webrtcvad.Vad(3)
mel_filterbank = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=512, n_mels=N_MELS)

processing_queue = queue.Queue()
mel_buffer = np.zeros((N_MELS, NUM_FRAMES))

def core_1_audio_callback(indata, frames, time, status):
    if status:
        print(status)
    chunk = indata[:, 0].copy()
    audio_ring_buffer.append(chunk) 
    processing_queue.put(chunk)

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#1e1e1e')
ax.set_facecolor('#1e1e1e')
ax.tick_params(colors='white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')

img = ax.imshow(mel_buffer, aspect='auto', origin='lower', cmap='inferno', vmin=-80, vmax=0)
ax.set_ylabel("Mel Filter Banks (0-40)")
ax.set_xlabel("Time (Frames)")
cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
cbar.set_label("Energy (dB)", color='white')
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

def core_0_intelligence(frame):
    global mel_buffer, hangover_timer
    updated = False
    is_sustained_speech = False
    debug_status = "QUIET | Listening..."
    status_color = '#a9a9a9'

    while not processing_queue.empty():
        audio_chunk = processing_queue.get()
        
        # 1. Strip low-frequency thuds
        filtered_chunk = scipy.signal.sosfilt(SOS_FILTER, audio_chunk)
        
        # 2. Calculate Zero Crossing Rate (ZCR)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(filtered_chunk)))) / 2
        zcr = zero_crossings / CHUNK_SIZE
        
        audio_chunk_int16 = np.int16(filtered_chunk * 32767.0)
        raw_bytes = audio_chunk_int16.tobytes()
        
        # 3. WebRTC Gate
        is_webrtc_speech = vad.is_speech(raw_bytes, SAMPLE_RATE)
        
        # 4. THE VOWEL GATE (Anti-Clap Logic)
        is_vowel = False
        if is_webrtc_speech:
            if zcr < VOWEL_ZCR_MAX:
                is_vowel = True
                debug_status = f"VOWEL DETECTED | ZCR: {zcr:.2f}"
            else:
                # WebRTC was fooled, but ZCR proves it's a clap/hiss
                debug_status = f"REJECTED CLAP/NOISE | High ZCR: {zcr:.2f}"
                status_color = '#ff8c00' # Orange warning

        vad_history.append(is_vowel)
        
        # 5. Trigger: Require 4 frames (~120ms) of vowels in the last 450ms
        if sum(vad_history) >= 4:
            hangover_timer = KWS_HOLD_FRAMES

        if hangover_timer > 0:
            is_sustained_speech = True
            hangover_timer -= 1
        else:
            is_sustained_speech = False

        if is_sustained_speech:
            windowed_audio = filtered_chunk * np.hanning(CHUNK_SIZE)
            padded_audio = np.pad(windowed_audio, (0, 32), mode='constant')
            fft_result = np.fft.rfft(padded_audio, n=512)
            power_spectrum = np.abs(fft_result) ** 2
            mel_spectrogram = np.dot(mel_filterbank, power_spectrum)
            log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max, top_db=80)
            
            debug_status = f"ACTIVE | Human Speech Verified | Holding: {hangover_timer}"
            status_color = '#00ff00'
        else:
            log_mel = np.full(N_MELS, -80.0)

        mel_buffer = np.roll(mel_buffer, -1, axis=1)
        mel_buffer[:, -1] = log_mel
        updated = True

    if updated:
        img.set_array(mel_buffer)
        ax.set_title(debug_status, color=status_color, fontweight='bold')
            
    return [img]

print("Simulating Vowel-Gated VAD Architecture... Close window to stop.")
stream = sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE, dtype='float32', callback=core_1_audio_callback)
ani = animation.FuncAnimation(fig, core_0_intelligence, interval=30, blit=False)

with stream:
    plt.show()
