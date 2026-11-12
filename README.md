# clip-factory
Turns a long video into 2–4 vertical (1080×1920) captioned clips for YouTube Shorts and TikTok.

1. `transcribe` workflow — downloads the video (yt-dlp), transcribes it with faster-whisper, pushes the transcript to branch `job/<job>` (the source video stays as a 3-day artifact).
2. Claude reads `transcript.txt` and picks the moments.
3. `render` workflow — cuts, reframes to 9:16 (face tracking or blurred-fill) and burns in word-by-word captions. Pushes the clips to branch `job/<job>` under `clips/`.

Optional secret `YT_COOKIES`: a Netscape-format cookies.txt for YouTube, used if YouTube blocks the runner.
