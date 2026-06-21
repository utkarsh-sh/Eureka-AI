# Converts the videos to mp3
import os
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = BASE_DIR / "videos"
AUDIO_DIR = BASE_DIR / "audios"


def main():
    if not VIDEOS_DIR.exists():
        raise FileNotFoundError(f"Missing videos directory: {VIDEOS_DIR}")

    AUDIO_DIR.mkdir(exist_ok=True)
    for file_path in sorted(VIDEOS_DIR.iterdir()):
        if not file_path.is_file():
            continue

        stem = file_path.stem
        tutorial_number = stem
        if " #" in stem:
            tutorial_number = stem.split(" #", 1)[1].split(" [", 1)[0].strip()
        file_name = stem.split(" ｜ ", 1)[0]
        output_path = AUDIO_DIR / f"{tutorial_number}_{file_name}.mp3"
        print(tutorial_number, file_name)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(file_path), str(output_path)],
            check=True,
        )


if __name__ == "__main__":
    main()