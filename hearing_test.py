import time
import numpy as np
from faster_whisper import WhisperModel

from reachy_mini import ReachyMini

stt = WhisperModel("base", device="cpu", compute_type="int8")


def listen(mini, seconds=3.0):
    sr = mini.media.get_input_audio_samplerate()
    mini.media.start_recording()
    chunks, t0 = [], time.time()
    try:
        while time.time() - t0 < seconds:
            s = mini.media.get_audio_sample()
            if s is not None:
                chunks.append(s)
            else:
                time.sleep(0.01)
    finally:
        mini.media.stop_recording()
    if not chunks:
        return np.zeros(0, np.float32), sr
    audio = np.concatenate(chunks).astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def transcribe(audio, sr):
    if audio.size == 0:
        return "", 0.0
    if sr != 16000:
        import math
        n = int(math.floor(audio.size * 16000 / sr))
        audio = np.interp(np.linspace(0, audio.size, n, endpoint=False),
                          np.arange(audio.size), audio).astype(np.float32)
    segments, _ = stt.transcribe(audio, language="en")
    segments = list(segments)
    text = " ".join(seg.text for seg in segments).strip()
    # avg_logprob is Whisper's confidence (closer to 0 = more confident)
    conf = np.mean([seg.avg_logprob for seg in segments]) if segments else 0.0
    return text, conf


with ReachyMini() as mini:
    print("Say something after the prompt. Ctrl+C to quit.")
    try:
        while True:
            input("Press Enter, then speak for 3 seconds… ")
            audio, sr = listen(mini, seconds=3.0)
            level = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
            text, conf = transcribe(audio, sr)
            print(f"  mic level : {level:.4f}  (near 0 means no sound captured)")
            print(f"  heard     : {text!r}")
            print(f"  confidence: {conf:.2f}  (closer to 0 is better)\n")
    except KeyboardInterrupt:
        print("\nDone.")
