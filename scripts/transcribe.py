#!/usr/bin/env python3
"""Transcribe a video with faster-whisper and write files the clip picker can read.

Outputs (in --out):
  words.json       [{"s": start, "e": end, "w": "word"}, ...]  (used for captions)
  transcript.txt   one line per segment: [MM:SS.s-MM:SS.s] text  (read by Claude)
  meta.json        duration, language, word count
"""
import argparse
import json
import os

from faster_whisper import WhisperModel


def ts(t: float) -> str:
    m, s = divmod(t, 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:04.1f}" if h else f"{m:02d}:{s:04.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="out")
    ap.add_argument("--model", default="small")
    ap.add_argument("--language", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    model = WhisperModel(a.model, device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        a.video,
        language=a.language or None,
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
    )

    words, lines = [], []
    for seg in segments:
        lines.append(f"[{ts(seg.start)}-{ts(seg.end)}] {seg.text.strip()}")
        for w in seg.words or []:
            words.append({"s": round(w.start, 2), "e": round(w.end, 2), "w": w.word.strip()})
        print(lines[-1], flush=True)

    with open(os.path.join(a.out, "words.json"), "w") as f:
        json.dump(words, f, ensure_ascii=False)
    with open(os.path.join(a.out, "transcript.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(
            {
                "duration": round(info.duration, 2),
                "language": info.language,
                "language_probability": round(info.language_probability, 3),
                "words": len(words),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
