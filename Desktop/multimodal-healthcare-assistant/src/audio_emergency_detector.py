import queue
import numpy as np
import sounddevice as sd
import whisper

SAMPLE_RATE = 16000
CHUNK_DURATION = 5  # seconds
VAD_THRESHOLD = 3

KEYWORDS = ["help", "doctor", "pain", "emergency", "can't breathe"]

audio_queue = queue.Queue()

# Load Whisper model
whisper_model = whisper.load_model("base")


def rms_energy(signal):
    return np.sqrt(np.mean(signal ** 2))


def record_audio_chunk(duration=CHUNK_DURATION):
    recording = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    return recording.flatten()


def detect_emergency(text):
    text = text.lower()
    return [kw for kw in KEYWORDS if kw in text]


while True:
    # Record audio
    audio_chunk = record_audio_chunk()

    # Energy check (simple VAD)
    energy = rms_energy(audio_chunk)
    median_energy = np.median(np.abs(audio_chunk))

    if energy > VAD_THRESHOLD * median_energy:
        try:
            # Speech-to-text
            result = whisper_model.transcribe(audio_chunk)
            text = result["text"].lower()

            # Keyword detection
            detected_keywords = detect_emergency(text)

            if detected_keywords:
                print(f"[CRITICAL AUDIO ALERT] {detected_keywords}")

        except Exception as e:
            print("Transcription error:", e)