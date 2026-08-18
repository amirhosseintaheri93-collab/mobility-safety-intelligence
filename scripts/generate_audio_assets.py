"""Generate the app's original, copyright-safe interface feedback sounds."""

from __future__ import annotations

from pathlib import Path
import wave

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "assets" / "audio"
SAMPLE_RATE = 22_050


def envelope(length: int, attack: float = 0.01, release: float = 0.18) -> np.ndarray:
    """Return a smooth attack/release envelope for a note-sized buffer."""
    curve = np.ones(length, dtype=np.float64)
    attack_samples = min(length, max(1, int(attack * SAMPLE_RATE)))
    release_samples = min(length, max(1, int(release * SAMPLE_RATE)))
    curve[:attack_samples] = np.sin(np.linspace(0, np.pi / 2, attack_samples)) ** 2
    curve[-release_samples:] *= np.cos(np.linspace(0, np.pi / 2, release_samples)) ** 2
    return curve


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
