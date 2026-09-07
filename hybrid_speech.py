
import sounddevice as sd
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import collections
import queue
import webrtcvad


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
CHANNELS = 1

# 30 ms frame
CHUNK_SIZE = 480

# 49 KB = 50,176 bytes
# 16-bit = 2 bytes/sample
# 50,176 / 2 = 25,088 samples
# 25,088 / 16,000 = 1.568 seconds
BUFFER_KB = 49
BUFFER_SAMPLES = (BUFFER_KB * 1024) // 2

# Rolling raw audio buffer
audio_ring_buffer = collections.deque(
    maxlen=BUFFER_SAMPLES
)


# ============================================================
# FILTERS
# ============================================================

# Instead of the old 300 Hz HPF, retain the useful speech
# fundamental region.
#
# 85 Hz HPF removes rumble/thuds but preserves low speech
# fundamentals.

SOS_HPF = scipy.signal.butter(
    N=2,
    Wn=85,
    btype="highpass",
    fs=SAMPLE_RATE,
    output="sos"
)

# Speech-related bands
SOS_VOICE = scipy.signal.butter(
    N=2,
    Wn=[85, 255],
    btype="bandpass",
    fs=SAMPLE_RATE,
    output="sos"
)

SOS_SPEECH = scipy.signal.butter(
    N=2,
    Wn=[255, 2000],
    btype="bandpass",
    fs=SAMPLE_RATE,
    output="sos"
)

SOS_HIGH = scipy.signal.butter(
    N=2,
    Wn=[2000, 4000],
    btype="bandpass",
    fs=SAMPLE_RATE,
    output="sos"
)


# ============================================================
# WEBRTC VAD
# ============================================================

# 3 = most aggressive WebRTC setting
vad = webrtcvad.Vad(3)


# ============================================================
# FILTER STATES
# ============================================================

# IMPORTANT:
# Preserve filter state between frames.
# Otherwise every 30 ms block creates artificial transients.

zi_hpf = scipy.signal.sosfilt_zi(SOS_HPF)
zi_voice = scipy.signal.sosfilt_zi(SOS_VOICE)
zi_speech = scipy.signal.sosfilt_zi(SOS_SPEECH)
zi_high = scipy.signal.sosfilt_zi(SOS_HIGH)


# ============================================================
# TEMPORAL SPEECH GATE
# ============================================================

# 3 frames = 90 ms
CONFIRM_FRAMES = 3

# Require at least 2 good frames out of 3
MIN_POSITIVE_FRAMES = 2

# Once speech is active, allow 1 bad frame
MAX_GAP_FRAMES = 1

# 3 bad frames = 90 ms to release
RELEASE_FRAMES = 3

speech_history = collections.deque(
    maxlen=CONFIRM_FRAMES
)

speech_state = False

gap_count = 0
negative_count = 0


# ============================================================
# THRESHOLDS
# ============================================================

# WebRTC must normally agree.
# Strong DSP can override WebRTC for fast response.
DSP_THRESHOLD = 0.60
STRONG_DSP_THRESHOLD = 0.82

# ZCR anti-clap gate
VOWEL_ZCR_MAX = 0.15

# Minimum signal level
MIN_DB = -55.0


# ============================================================
# ADAPTIVE NOISE FLOOR
# ============================================================

noise_floor_db = -55.0

NOISE_UPDATE_RATE = 0.02


# ============================================================
# PROCESSING QUEUE
# ============================================================

processing_queue = queue.Queue(
    maxsize=8
)


# ============================================================
# AUDIO CALLBACK
# ============================================================

def audio_callback(
    indata,
    frames,
    callback_time,
    status
):

    if status:
        print(status)

    chunk = indata[:, 0].copy()

    # Keep raw float32 history for visualization/debugging
    audio_ring_buffer.extend(
        chunk.tolist()
    )

    try:
        processing_queue.put_nowait(
            chunk
        )
    except queue.Full:
        # Drop old processing work rather than allowing
        # latency to accumulate.
        pass


# ============================================================
# BASIC FEATURES
# ============================================================

def rms(x):

    return np.sqrt(
        np.mean(x * x) + 1e-12
    )


def zero_crossing_rate(x):

    if len(x) < 2:
        return 0.0

    crossings = np.sum(
        x[:-1] * x[1:] < 0
    )

    return crossings / (len(x) - 1)


def teager_energy(x):

    if len(x) < 3:
        return 0.0

    teo = (
        x[1:-1] ** 2
        - x[:-2] * x[2:]
    )

    return np.mean(
        np.abs(teo)
    )


# ============================================================
# SPECTRAL FEATURES
# ============================================================

def spectral_features(x):

    window = np.hanning(
        len(x)
    )

    spectrum = np.abs(
        np.fft.rfft(
            x * window
        )
    )

    power = spectrum ** 2

    freqs = np.fft.rfftfreq(
        len(x),
        1.0 / SAMPLE_RATE
    )

    total = (
        np.sum(power) + 1e-12
    )

    # Spectral centroid
    centroid = (
        np.sum(freqs * power)
        / total
    )

    # Spectral flatness
    geometric_mean = np.exp(
        np.mean(
            np.log(
                power + 1e-12
            )
        )
    )

    arithmetic_mean = (
        np.mean(power) + 1e-12
    )

    flatness = (
        geometric_mean
        / arithmetic_mean
    )

    # 85% rolloff
    cumulative = np.cumsum(power)

    target = (
        0.85 * cumulative[-1]
    )

    index = np.searchsorted(
        cumulative,
        target
    )

    index = min(
        index,
        len(freqs) - 1
    )

    rolloff = freqs[index]

    return (
        centroid,
        flatness,
        rolloff
    )


# ============================================================
# EXTRACT DSP FEATURES
# ============================================================

def extract_features(frame):

    global zi_hpf
    global zi_voice
    global zi_speech
    global zi_high

    # Input from sounddevice is float32 [-1, +1]
    x = frame.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Persistent filters
    # --------------------------------------------------------

    filtered, zi_hpf = scipy.signal.sosfilt(
        SOS_HPF,
        x,
        zi=zi_hpf
    )

    voice, zi_voice = scipy.signal.sosfilt(
        SOS_VOICE,
        x,
        zi=zi_voice
    )

    speech_band, zi_speech = scipy.signal.sosfilt(
        SOS_SPEECH,
        x,
        zi=zi_speech
    )

    high_band, zi_high = scipy.signal.sosfilt(
        SOS_HIGH,
        x,
        zi=zi_high
    )

    # --------------------------------------------------------
    # Energies
    # --------------------------------------------------------

    total_energy = np.mean(
        filtered ** 2
    ) + 1e-12

    voice_energy = np.mean(
        voice ** 2
    ) + 1e-12

    speech_energy = np.mean(
        speech_band ** 2
    ) + 1e-12

    high_energy = np.mean(
        high_band ** 2
    ) + 1e-12

    # --------------------------------------------------------
    # Ratios
    # --------------------------------------------------------

    voice_ratio = (
        voice_energy
        / total_energy
    )

    speech_ratio = (
        speech_energy
        / total_energy
    )

    high_ratio = (
        high_energy
        / total_energy
    )

    # --------------------------------------------------------
    # Level
    # --------------------------------------------------------

    frame_rms = np.sqrt(
        total_energy
    )

    db = (
        20.0
        * np.log10(
            frame_rms + 1e-8
        )
    )

    # --------------------------------------------------------
    # ZCR
    # --------------------------------------------------------

    zcr = zero_crossing_rate(
        filtered
    )

    # --------------------------------------------------------
    # Teager
    # --------------------------------------------------------

    teo = teager_energy(
        filtered
    )

    teo_ratio = (
        teo
        / total_energy
    )

    # --------------------------------------------------------
    # Spectrum
    # --------------------------------------------------------

    centroid, flatness, rolloff = (
        spectral_features(
            filtered
        )
    )

    return {
        "filtered": filtered,
        "db": db,
        "voice_ratio": voice_ratio,
        "speech_ratio": speech_ratio,
        "high_ratio": high_ratio,
        "zcr": zcr,
        "teager": teo_ratio,
        "centroid": centroid,
        "flatness": flatness,
        "rolloff": rolloff
    }


# ============================================================
# DSP SPEECH SCORE
# ============================================================

def calculate_dsp_score(f):

    # --------------------------------------------------------
    # Energy
    # --------------------------------------------------------

    if f["db"] <= MIN_DB:

        energy_score = 0.0

    elif f["db"] >= -25:

        energy_score = 1.0

    else:

        energy_score = (
            (f["db"] - MIN_DB)
            / (-25.0 - MIN_DB)
        )

    # --------------------------------------------------------
    # Voice band
    # --------------------------------------------------------

    voice_score = np.clip(
        f["voice_ratio"] / 0.35,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Main speech band
    # --------------------------------------------------------

    speech_score = np.clip(
        f["speech_ratio"] / 0.45,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # High-frequency speech
    # --------------------------------------------------------

    high_score = np.clip(
        f["high_ratio"] / 0.15,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # ZCR
    #
    # Very low ZCR:
    #     likely hum / mechanical noise
    #
    # Very high ZCR:
    #     likely hiss / clap / broadband noise
    # --------------------------------------------------------

    z = f["zcr"]

    if z < 0.015:

        zcr_score = 0.0

    elif z > 0.25:

        zcr_score = 0.1

    else:

        zcr_score = (
            (z - 0.015)
            / (0.25 - 0.015)
        )

    zcr_score = np.clip(
        zcr_score,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Spectral centroid
    # --------------------------------------------------------

    c = f["centroid"]

    if c < 200:

        centroid_score = 0.0

    elif c <= 1500:

        centroid_score = (
            (c - 200)
            / (1500 - 200)
        )

    elif c <= 3500:

        centroid_score = (
            1.0
            - (c - 1500)
            / (3500 - 1500)
        )

    else:

        centroid_score = 0.0

    centroid_score = np.clip(
        centroid_score,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Flatness
    # --------------------------------------------------------

    flatness_score = np.clip(
        1.0 - f["flatness"],
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Teager
    # --------------------------------------------------------

    teager_score = np.clip(
        f["teager"] * 200.0,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Combined DSP score
    # --------------------------------------------------------

    score = (

        0.20 * energy_score +

        0.20 * voice_score +

        0.20 * speech_score +

        0.10 * high_score +

        0.10 * zcr_score +

        0.08 * centroid_score +

        0.07 * flatness_score +

        0.05 * teager_score
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )


# ============================================================
# TRANSIENT / SPIKE REJECTION
# ============================================================

previous_db = -55.0
previous_score = 0.0


def detect_transient(
    features,
    dsp_score
):

    global previous_db
    global previous_score

    current_db = features["db"]

    db_jump = (
        current_db
        - previous_db
    )

    score_jump = (
        dsp_score
        - previous_score
    )

    # A sudden huge energy increase combined with a sudden
    # score increase is treated as a transient candidate.
    #
    # The temporal gate will also prevent isolated events.

    transient = (

        db_jump > 18.0

        and score_jump > 0.35

        and dsp_score > 0.75
    )

    previous_db = current_db
    previous_score = dsp_score

    return transient


# ============================================================
# HYBRID DECISION
# ============================================================

def hybrid_candidate(
    features,
    dsp_score,
    webrtc_speech
):

    # --------------------------------------------------------
    # Basic signal presence
    # --------------------------------------------------------

    if features["db"] < MIN_DB:

        return False

    # --------------------------------------------------------
    # Old anti-clap vowel gate
    #
    # Very high ZCR is unlikely to be a voiced vowel.
    # --------------------------------------------------------

    vowel_like = (
        features["zcr"]
        < VOWEL_ZCR_MAX
    )

    # --------------------------------------------------------
    # Normal case:
    #
    # WebRTC + DSP + vowel characteristics
    # --------------------------------------------------------

    normal_candidate = (

        webrtc_speech

        and vowel_like

        and dsp_score >= DSP_THRESHOLD
    )

    # --------------------------------------------------------
    # Strong DSP case.
    #
    # Allows fast detection if WebRTC happens to miss a
    # frame, but still requires reasonable ZCR.
    # --------------------------------------------------------

    strong_candidate = (

        dsp_score >= STRONG_DSP_THRESHOLD

        and vowel_like

        and features["speech_ratio"] > 0.08
    )

    return (
        normal_candidate
        or strong_candidate
    )


# ============================================================
# TEMPORAL GATE
# ============================================================

def update_temporal_gate(
    candidate,
    transient
):

    global speech_state
    global gap_count
    global negative_count

    # --------------------------------------------------------
    # Transient never counts as a positive speech frame.
    # --------------------------------------------------------

    if transient:

        candidate = False

    # --------------------------------------------------------
    # Store current frame
    # --------------------------------------------------------

    speech_history.append(
        candidate
    )

    # ========================================================
    # CURRENTLY NOT SPEECH
    # ========================================================

    if not speech_state:

        positive_frames = sum(
            speech_history
        )

        # Require 2 positive frames within 3 frames.
        #
        # This prevents:
        #
        # 0 0 1 0 0
        #
        # from triggering.
        #
        # But allows:
        #
        # 0 1 1
        #
        # to trigger in ~60 ms.
        #

        if (
            positive_frames
            >= MIN_POSITIVE_FRAMES
        ):

            speech_state = True

            gap_count = 0
            negative_count = 0

    # ========================================================
    # CURRENTLY SPEECH
    # ========================================================

    else:

        if candidate:

            gap_count = 0
            negative_count = 0

        else:

            gap_count += 1
            negative_count += 1

            # Allow a single bad frame inside speech
            if gap_count <= MAX_GAP_FRAMES:

                return speech_state

            # Release after several consecutive bad frames
            if (
                negative_count
                >= RELEASE_FRAMES
            ):

                speech_state = False

                speech_history.clear()

                gap_count = 0
                negative_count = 0

    return speech_state


# ============================================================
# VISUALIZATION
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 6)
)

score_history = collections.deque(
    maxlen=100
)

webrtc_history = collections.deque(
    maxlen=100
)

speech_history_plot = collections.deque(
    maxlen=100
)

time_history = collections.deque(
    maxlen=100
)

start_time = None


def update_plot():

    if not score_history:
        return

    ax.clear()

    t = np.array(
        time_history
    )

    scores = np.array(
        score_history
    )

    webrtc = np.array(
        webrtc_history
    )

    speech = np.array(
        speech_history_plot
    )

    ax.plot(
        t,
        scores,
        label="DSP speech score"
    )

    ax.plot(
        t,
        webrtc * 0.9,
        linestyle="--",
        label="WebRTC"
    )

    ax.plot(
        t,
        speech,
        linewidth=3,
        label="Final speech"
    )

    ax.axhline(
        DSP_THRESHOLD,
        linestyle=":",
        label="DSP threshold"
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.set_xlabel(
        "Time (seconds)"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Hybrid Human Speech Detector"
    )

    ax.grid(
        True,
        alpha=0.2
    )

    ax.legend(
        loc="upper right"
    )

    plt.tight_layout()

    plt.pause(
        0.001
    )


# ============================================================
# INTELLIGENCE LOOP
# ============================================================

def core_intelligence(frame):

    global start_time
    global noise_floor_db

    updated = False

    while not processing_queue.empty():

        try:

            audio_chunk = (
                processing_queue.get_nowait()
            )

        except queue.Empty:

            break

        if len(audio_chunk) != CHUNK_SIZE:
            continue

        if start_time is None:
            start_time = __import__(
                "time"
            ).time()

        # ----------------------------------------------------
        # DSP
        # ----------------------------------------------------

        features = extract_features(
            audio_chunk
        )

        # ----------------------------------------------------
        # Adaptive noise floor
        # ----------------------------------------------------

        current_db = features["db"]

        if not speech_state:

            if (
                current_db
                < noise_floor_db + 10
            ):

                noise_floor_db = (

                    (1.0 - NOISE_UPDATE_RATE)
                    * noise_floor_db

                    +

                    NOISE_UPDATE_RATE
                    * current_db
                )

        # ----------------------------------------------------
        # DSP score
        # ----------------------------------------------------

        dsp_score = calculate_dsp_score(
            features
        )

        # ----------------------------------------------------
        # Noise suppression
        # ----------------------------------------------------

        if (
            current_db
            < noise_floor_db + 8
        ):

            dsp_score *= 0.35

        # ----------------------------------------------------
        # Convert filtered audio to int16 for WebRTC
        #
        # WebRTC expects PCM16.
        # ----------------------------------------------------

        filtered = features[
            "filtered"
        ]

        filtered = np.clip(
            filtered,
            -1.0,
            1.0
        )

        audio_int16 = (
            filtered * 32767.0
        ).astype(
            np.int16
        )

        raw_bytes = (
            audio_int16.tobytes()
        )

        # ----------------------------------------------------
        # WebRTC
        # ----------------------------------------------------

        try:

            webrtc_speech = vad.is_speech(
                raw_bytes,
                SAMPLE_RATE
            )

        except Exception:

            webrtc_speech = False

        # ----------------------------------------------------
        # Detect sudden spike
        # ----------------------------------------------------

        transient = detect_transient(
            features,
            dsp_score
        )

        # ----------------------------------------------------
        # Hybrid candidate
        # ----------------------------------------------------

        candidate = hybrid_candidate(
            features,
            dsp_score,
            webrtc_speech
        )

        # ----------------------------------------------------
        # Final temporal decision
        # ----------------------------------------------------

        final_speech = (
            update_temporal_gate(
                candidate,
                transient
            )
        )

        # ----------------------------------------------------
        # Plot
        # ----------------------------------------------------

        now = (
            __import__("time").time()
            - start_time
        )

        time_history.append(
            now
        )

        score_history.append(
            dsp_score
        )

        webrtc_history.append(
            1.0
            if webrtc_speech
            else 0.0
        )

        speech_history_plot.append(
            1.0
            if final_speech
            else 0.0
        )

        # ----------------------------------------------------
        # Console
        # ----------------------------------------------------

        if transient:

            status = (
                "SPIKE REJECTED"
            )

        elif final_speech:

            status = (
                "HUMAN SPEECH"
            )

        elif candidate:

            status = (
                "SPEECH CANDIDATE"
            )

        else:

            status = (
                "NO SPEECH"
            )

        print(
            f"\r"
            f"{status:<18} "
            f"DSP={dsp_score:.2f} "
            f"WebRTC={int(webrtc_speech)} "
            f"ZCR={features['zcr']:.3f} "
            f"dB={current_db:6.1f}",
            end=""
        )

        updated = True

    if updated:
        update_plot()

    return []


# ============================================================
# START
# ============================================================

print()
print("=" * 65)
print("HYBRID HUMAN SPEECH DETECTOR")
print("=" * 65)
print()
print("16 kHz / 16-bit PCM")
print(f"Frame size       : {CHUNK_SIZE} samples")
print(f"Frame duration   : {CHUNK_SIZE / SAMPLE_RATE * 1000:.0f} ms")
print(f"Audio buffer     : {BUFFER_KB} KB")
print(
    f"Buffer duration  : "
    f"{BUFFER_SAMPLES / SAMPLE_RATE:.3f} seconds"
)
print()
print("WebRTC VAD       : ENABLED")
print("DSP features     : ENABLED")
print("ZCR vowel gate   : ENABLED")
print("Spike rejection  : ENABLED")
print("Temporal gate    : ENABLED")
print()
print("Speak normally.")
print("Clicks/claps/spikes should not trigger speech.")
print("Close the window or press Ctrl+C to stop.")
print()


stream = sd.InputStream(
    channels=CHANNELS,
    samplerate=SAMPLE_RATE,
    blocksize=CHUNK_SIZE,
    dtype="float32",
    callback=audio_callback
)

ani = animation.FuncAnimation(
    fig,
    core_intelligence,
    interval=30,
    blit=False,
    cache_frame_data=False
)

try:

    with stream:
        plt.show()

except KeyboardInterrupt:

    print("\nStopped.")

