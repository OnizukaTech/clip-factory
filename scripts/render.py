#!/usr/bin/env python3
"""Cut, reframe to 9:16 and caption clips from a long video.

clips.json format:
[
  {"id": 1, "start": 83.2, "end": 131.0, "title": "Short title", "hook": "Optional on-screen hook text"},
  ...
]

Modes:
  track  - crop a 9:16 window that follows the largest face (talking heads, podcasts)
  fit    - whole frame centred over a blurred copy of itself (gameplay, film, slides, wide shots)
  auto   - track if a face is visible in most of the clip, otherwise fit
"""
import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

W, H = 1080, 1920


def run(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------- timing
def snap(start, end, words, pad=0.15):
    """Move cut points to word boundaries so clips never start or end mid-word."""
    if not words:
        return start, end
    inside = [w for w in words if w["s"] >= start - 0.4 and w["e"] <= end + 0.4]
    if not inside:
        return start, end
    s = max(0.0, inside[0]["s"] - pad)
    e = inside[-1]["e"] + pad
    return s, e


# ---------------------------------------------------------------- captions
def ass_time(t):
    t = max(0.0, t)
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def esc(text):
    return text.replace("\\", "").replace("{", "(").replace("}", ")")


def chunk_words(words, max_words=3, max_gap=0.6, max_len=1.6):
    chunks, cur = [], []
    for w in words:
        if cur and (
            len(cur) >= max_words
            or w["s"] - cur[-1]["e"] > max_gap
            or w["e"] - cur[0]["s"] > max_len
            or cur[-1]["w"].endswith((".", "?", "!"))
        ):
            chunks.append(cur)
            cur = []
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def write_ass(path, words, clip_start, clip_end, hook, caption_y):
    rel = []
    for w in words:
        if w["e"] <= clip_start or w["s"] >= clip_end or not w["w"]:
            continue
        rel.append({"s": max(0, w["s"] - clip_start), "e": min(clip_end, w["e"]) - clip_start, "w": w["w"]})

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,DejaVu Sans,84,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,7,3,5,60,60,0,1
Style: Hook,DejaVu Sans,64,&H00000000,&H00000000,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,3,18,0,8,80,80,220,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ev = []
    if hook:
        ev.append(f"Dialogue: 1,{ass_time(0)},{ass_time(3.5)},Hook,,0,0,0,,{{\\fad(150,250)}}{esc(hook)}")

    hi = "{\\c&H00E5FF&}"  # yellow (BGR)
    norm = "{\\c&HFFFFFF&}"
    pos = f"{{\\an5\\pos({W // 2},{caption_y})}}"
    for ch in chunk_words(rel):
        for i, w in enumerate(ch):
            start = w["s"]
            end = ch[i + 1]["s"] if i + 1 < len(ch) else w["e"]
            if end - start < 0.05:
                continue
            parts = []
            for j, x in enumerate(ch):
                t = esc(x["w"]).upper()
                parts.append(hi + t + norm if j == i else t)
            ev.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Cap,,0,0,0,,{pos}{' '.join(parts)}")

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(ev) + "\n")


# ---------------------------------------------------------------- framing
def face_track(path, step=0.25):
    """Return (fps, frame_count, width, height, list of (t, cx or None))."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cascade = cv2.CascadeClassifier(os.path.join(getattr(cv2, "data", None).haarcascades if hasattr(cv2, "data") else "", "haarcascade_frontalface_default.xml"))
    if cascade.empty():
        print("warning: face model not found, using fit mode", flush=True)
    every = max(1, int(round(fps * step)))
    scale = 640 / vw
    samples, i = [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % every == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            small = cv2.resize(frame, (640, int(vh * scale)))
            gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            faces = cascade.detectMultiScale(gray, 1.1, 6, minSize=(40, 40)) if not cascade.empty() else []
            if len(faces):
                x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
                samples.append((i / fps, (x + fw / 2) / scale))
            else:
                samples.append((i / fps, None))
        i += 1
    cap.release()
    return fps, i, vw, vh, samples


def crop_path(samples, n_frames, fps, vw, vh):
    """Per-frame crop x with a dead zone so the camera only moves when it has to."""
    cw = int(round(vh * 9 / 16)) // 2 * 2
    cw = min(cw, vw)
    centre = vw / 2
    last, raw = centre, []
    for _, cx in samples:
        if cx is not None:
            last = cx
        raw.append(last)
    # median filter to drop single-sample jumps
    k = 5
    med = [float(np.median(raw[max(0, i - k // 2): i + k // 2 + 1])) for i in range(len(raw))]
    # dead zone + easing
    cam, out = med[0] if med else centre, []
    dead = cw * 0.12
    for target in med:
        if abs(target - cam) > dead:
            cam += (target - cam) * 0.35
        out.append(cam)
    times = [t for t, _ in samples] or [0]
    ft = np.arange(n_frames) / fps
    xs = np.interp(ft, times, out if out else [centre])
    xs = np.clip(xs - cw / 2, 0, vw - cw)
    return cw, xs.astype(int)


def render_track(tmp, ass, out, fps, n, vw, vh, xs, cw):
    ff = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-",
            "-i", tmp,
            "-map", "0:v", "-map", "1:a?",
            "-vf", f"ass={ass}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out,
        ],
        stdin=subprocess.PIPE,
    )
    cap = cv2.VideoCapture(tmp)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        x = int(xs[min(i, len(xs) - 1)])
        crop = frame[:, x: x + cw]
        ff.stdin.write(cv2.resize(crop, (W, H), interpolation=cv2.INTER_LANCZOS4).tobytes())
        i += 1
    cap.release()
    ff.stdin.close()
    if ff.wait() != 0:
        sys.exit("ffmpeg failed (track)")


def render_fit(tmp, ass, out):
    vf = (
        f"[0:v]split[a][b];"
        f"[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=30:3,eq=brightness=-0.12[bg];"
        f"[b]scale={W}:-2[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,ass={ass}[v]"
    )
    run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
        "-filter_complex", vf, "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", out,
    ])


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--words", required=True)
    ap.add_argument("--clips", required=True)
    ap.add_argument("--out", default="clips")
    ap.add_argument("--mode", default="auto", choices=["auto", "track", "fit"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    words = json.load(open(a.words)) if os.path.exists(a.words) else []
    clips = json.load(open(a.clips))
    manifest = []

    for c in clips:
        cid = c.get("id", len(manifest) + 1)
        s, e = snap(float(c["start"]), float(c["end"]), words)
        dur = e - s
        tmp = os.path.join(a.out, f"_src_{cid}.mp4")
        ass = os.path.join(a.out, f"_cap_{cid}.ass")
        out = os.path.join(a.out, f"clip_{cid}.mp4")

        # accurate cut first, so framing and captions all work from t=0
        run([
            "ffmpeg", "-y", "-loglevel", "error", "-ss", f"{s:.3f}", "-i", a.video, "-t", f"{dur:.3f}",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", "-c:a", "aac", "-b:a", "192k", tmp,
        ])

        mode = c.get("mode", a.mode)
        fps, n, vw, vh, samples = face_track(tmp)
        hit = sum(1 for _, x in samples if x is not None) / max(1, len(samples))
        if mode == "auto":
            mode = "track" if hit >= 0.5 and vw > vh else "fit"
        if vw <= vh * 9 / 16 + 2:
            mode = "fit"  # already vertical-ish

        caption_y = int(H * 0.70) if mode == "track" else int(H * 0.80)
        write_ass(ass, words, s, e, c.get("hook", ""), caption_y)

        if mode == "track":
            cw, xs = crop_path(samples, n, fps, vw, vh)
            render_track(tmp, ass, out, fps, n, vw, vh, xs, cw)
        else:
            render_fit(tmp, ass, out)

        thumb = os.path.join(a.out, f"clip_{cid}.jpg")
        run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{min(1.5, dur / 2):.2f}", "-i", out, "-frames:v", "1", "-q:v", "3", thumb])
        os.remove(tmp)
        os.remove(ass)
        manifest.append({**c, "id": cid, "start": round(s, 2), "end": round(e, 2), "duration": round(dur, 1),
                         "mode": mode, "face_coverage": round(hit, 2), "file": os.path.basename(out)})
        print(f"clip {cid}: {dur:.1f}s mode={mode} faces={hit:.0%}", flush=True)

    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
