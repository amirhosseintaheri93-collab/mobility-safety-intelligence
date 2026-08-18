"""Generate the app's original, copyright-safe music and interface sounds.

The two short loops are setar-inspired sound sketches, not recordings or
attempts to reproduce a particular performer or composition. Everything is
synthesised deterministically with NumPy and Python's standard ``wave`` module.
"""

from __future__ import annotations

from pathlib import Path
import wave

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "assets" / "audio"
SAMPLE_RATE = 22_050
LOOP_SECONDS = 18.0


def envelope(length: int, attack: float = 0.01, release: float = 0.18) -> np.ndarray:
    """Return a smooth attack/release envelope for a note-sized buffer."""
    curve = np.ones(length, dtype=np.float64)
    attack_samples = min(length, max(1, int(attack * SAMPLE_RATE)))
    release_samples = min(length, max(1, int(release * SAMPLE_RATE)))
    curve[:attack_samples] = np.sin(np.linspace(0, np.pi / 2, attack_samples)) ** 2
    curve[-release_samples:] *= np.cos(np.linspace(0, np.pi / 2, release_samples)) ** 2
    return curve


def plucked_string(
    frequency: float,
    seconds: float,
    rng: np.random.Generator,
    brightness: float = 0.52,
) -> np.ndarray:
    """A compact Karplus-Strong pluck with a gentle bridge-like resonance."""
    length = max(1, int(seconds * SAMPLE_RATE))
    delay = max(2, int(round(SAMPLE_RATE / frequency)))
    ring = rng.uniform(-1.0, 1.0, delay)
    signal = np.empty(length, dtype=np.float64)
    damping = 0.9955
    for index in range(length):
        current = ring[index % delay]
        following = ring[(index + 1) % delay]
        signal[index] = current
        ring[index % delay] = damping * (
            brightness * current + (1.0 - brightness) * following
        )
    time = np.arange(length) / SAMPLE_RATE
    resonance = 0.16 * np.sin(2 * np.pi * frequency * 2.01 * time)
    return (signal + resonance) * envelope(length, attack=0.004, release=0.2)


def soft_pad(frequency: float, seconds: float) -> np.ndarray:
    length = max(1, int(seconds * SAMPLE_RATE))
    time = np.arange(length) / SAMPLE_RATE
    vibrato = 0.006 * np.sin(2 * np.pi * 0.19 * time)
    phase = 2 * np.pi * frequency * time + vibrato
    pad = np.sin(phase) + 0.28 * np.sin(2 * phase) + 0.12 * np.sin(3 * phase)
    return pad * envelope(length, attack=1.1, release=1.4)


def add_note(target: np.ndarray, note: np.ndarray, start_seconds: float, gain: float) -> None:
    start = int(start_seconds * SAMPLE_RATE)
    if start >= len(target):
        return
    end = min(len(target), start + len(note))
    target[start:end] += gain * note[: end - start]


def finish_loop(signal: np.ndarray, peak: float = 0.19) -> np.ndarray:
    """Remove DC, soften loop boundaries, and master to a deliberately low level."""
    signal = signal - np.mean(signal)
    edge = int(0.32 * SAMPLE_RATE)
    signal[:edge] *= np.linspace(0, 1, edge)
    signal[-edge:] *= np.linspace(1, 0, edge)
    maximum = float(np.max(np.abs(signal))) or 1.0
    return np.tanh(signal / maximum * 1.35) * peak


def acoustic_loop() -> np.ndarray:
    rng = np.random.default_rng(1407)
    result = np.zeros(int(LOOP_SECONDS * SAMPLE_RATE), dtype=np.float64)
    # A deliberately original modal phrase with small ornamental turns.
    scale = [146.83, 164.81, 179.50, 196.00, 220.00, 246.94, 261.63, 293.66]
    melody = [0, 3, 4, 2, 1, 3, 5, 4, 2, 3, 1, 0, 4, 5, 3, 2, 1, 0]
    beat = 0.75
    for index, degree in enumerate(melody):
        start = 0.34 + index * beat * 1.22
        duration = 1.15 if index % 4 else 1.7
        add_note(
            result,
            plucked_string(scale[degree], duration, rng, brightness=0.56),
            start,
            0.34,
        )
        if index in {3, 7, 11, 15}:
            ornament = plucked_string(scale[max(0, degree - 1)] * 2, 0.32, rng, 0.62)
            add_note(result, ornament, start + 0.30, 0.12)
    for start in np.arange(0.2, LOOP_SECONDS - 1.0, 3.0):
        add_note(result, plucked_string(73.42, 2.2, rng, 0.48), float(start), 0.11)
    return finish_loop(result, peak=0.17)


def electronic_loop() -> np.ndarray:
    rng = np.random.default_rng(2311)
    result = np.zeros(int(LOOP_SECONDS * SAMPLE_RATE), dtype=np.float64)
    add_note(result, soft_pad(73.42, LOOP_SECONDS), 0, 0.10)
    add_note(result, soft_pad(110.00, LOOP_SECONDS), 0, 0.055)
    scale = [146.83, 164.81, 179.50, 196.00, 220.00, 246.94, 293.66]
    melody = [0, 4, 2, 5, 3, 1, 4, 6, 4, 2, 3, 1, 0, 3, 4, 2]
    for index, degree in enumerate(melody):
        start = 0.45 + index * 1.05
        note = plucked_string(scale[degree], 1.25, rng, brightness=0.69)
        add_note(result, note, start, 0.24)
        # A faint delayed octave gives the electronic sketch its spacious edge.
        echo = plucked_string(scale[degree] * 2, 0.72, rng, brightness=0.72)
        add_note(result, echo, start + 0.28, 0.055)
    time = np.arange(len(result)) / SAMPLE_RATE
    pulse = np.sin(2 * np.pi * 1.0 * time) ** 18
    result += 0.018 * pulse * np.sin(2 * np.pi * 73.42 * time)
    return finish_loop(result, peak=0.16)


def select_sound() -> np.ndarray:
    seconds = 0.18
    time = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    tone = np.sin(2 * np.pi * (520 + 360 * time) * time)
    return 0.12 * tone * np.exp(-20 * time)


def whoosh_sound() -> np.ndarray:
    rng = np.random.default_rng(997)
    seconds = 0.56
    length = int(seconds * SAMPLE_RATE)
    time = np.arange(length) / SAMPLE_RATE
    noise = rng.normal(0, 1, length)
    smooth = np.convolve(noise, np.ones(31) / 31, mode="same")
    sweep = np.sin(2 * np.pi * (180 * time + 520 * time**2))
    shape = np.sin(np.pi * time / seconds) ** 2
    return 0.10 * shape * (0.55 * smooth + 0.45 * sweep)


def reveal_sound() -> np.ndarray:
    seconds = 0.88
    length = int(seconds * SAMPLE_RATE)
    result = np.zeros(length, dtype=np.float64)
    time = np.arange(length) / SAMPLE_RATE
    for offset, frequency in ((0.0, 392.0), (0.16, 493.88), (0.32, 587.33)):
        start = int(offset * SAMPLE_RATE)
        local_time = time[: length - start]
        note = np.sin(2 * np.pi * frequency * local_time)
        note += 0.24 * np.sin(2 * np.pi * frequency * 2 * local_time)
        result[start:] += 0.055 * note * np.exp(-4.8 * local_time)
    return result * envelope(length, attack=0.008, release=0.16)


def write_wave(path: Path, signal: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(signal, -1.0, 1.0)
    pcm = (clipped * np.iinfo(np.int16).max).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm.tobytes())


def main() -> None:
    assets = {
        "setar-inspired-acoustic.wav": acoustic_loop(),
        "setar-inspired-electronic.wav": electronic_loop(),
        "ui-select.wav": select_sound(),
        "ui-whoosh.wav": whoosh_sound(),
        "ui-reveal.wav": reveal_sound(),
    }
    for name, signal in assets.items():
        path = OUTPUT_DIR / name
        write_wave(path, signal)
        print(f"Wrote {path.relative_to(ROOT)} ({len(signal) / SAMPLE_RATE:.2f}s)")


if __name__ == "__main__":
    main()
