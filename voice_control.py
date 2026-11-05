import time
import numpy as np
from faster_whisper import WhisperModel

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

stt = WhisperModel("base", device="cpu", compute_type="int8")  # offline, small

def listen(mini, seconds=3.0):
    """Record ~`seconds` of mic audio, return it as one mono float32 array + samplerate."""
    sr = mini.media.get_input_audio_samplerate()
    mini.media.start_recording()
    chunks, t0 = [], time.time()
    try:
        while time.time() - t0 < seconds:
            s = mini.media.get_audio_sample()
            if s is not None:
                chunks.append(s)
            else:
                time.sleep(0.01)  # nothing ready yet; don't busy-spin
    finally:
        mini.media.stop_recording()
    if not chunks:
        return np.zeros(0, np.float32), sr
    audio = np.concatenate(chunks).astype(np.float32)
    if audio.ndim > 1:          # if stereo, downmix to mono for the STT
        audio = audio.mean(axis=1)
    return audio, sr

def transcribe(audio, sr):
    if audio.size == 0:
        return ""
    # faster-whisper wants 16 kHz mono float32; resample if the mic differs.
    if sr != 16000:
        import math
        n = int(math.floor(audio.size * 16000 / sr))
        audio = np.interp(np.linspace(0, audio.size, n, endpoint=False),
                          np.arange(audio.size), audio).astype(np.float32)
    segments, _ = stt.transcribe(audio, language="en")
    return " ".join(seg.text for seg in segments).lower().strip()

# command keyword -> gesture
def look_up(mini):   mini.goto_target(head=create_head_pose(pitch=-25, degrees=True), duration=0.8)
def look_down(mini): mini.goto_target(head=create_head_pose(pitch=25, degrees=True), duration=0.8)
def center(mini):    mini.goto_target(head=create_head_pose(), duration=0.8)

COMMANDS = {
    "look up": look_up,
    "look down": look_down,
    "center": center,
    "reset": center,
}

with ReachyMini() as mini:
    mini.wake_up()
    try:
        while True:
            print("Listening…")
            text = transcribe(*listen(mini, seconds=3.0))
            print(f"Heard: {text!r}")
            if "stop" in text or "goodbye" in text:
                break
            for phrase, action in COMMANDS.items():
                if phrase in text:
                    action(mini)
                    break
    finally:
        center(mini)
        mini.goto_sleep()
