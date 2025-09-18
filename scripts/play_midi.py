#!/usr/bin/env python3
"""
Play or render MIDI files to WAV using Fluidsynth and a GM SoundFont.

Usage examples:
  python3 scripts/play_midi.py --file artifacts/samples/sample.mid
  python3 scripts/play_midi.py --dir artifacts/samples --limit 3
  python3 scripts/play_midi.py --file file.mid --sf2 assets/sf2/FluidR3_GM.sf2 --keep-wav
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, List

DEFAULT_SF2_CANDIDATES = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/TimGM6mb.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    "assets/sf2/FluidR3_GM.sf2",
]


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def find_soundfont(user_path: Optional[str]) -> Optional[str]:
    if user_path:
        p = Path(user_path)
        return str(p) if p.exists() else None
    for c in DEFAULT_SF2_CANDIDATES:
        if Path(c).exists():
            return c
    return None


def render_to_wav(mid_path: str, sf2_path: str, out_wav: str, samplerate: int = 44100, gain: float = 0.8) -> None:
    cmd = [
        "fluidsynth",
        "-ni", sf2_path, mid_path,
        "-F", out_wav,
        "-r", str(samplerate),
        "-g", str(gain),
    ]
    subprocess.run(cmd, check=True)


def play_wav(wav_path: str) -> None:
    if which("ffplay"):
        subprocess.run(["ffplay", "-nodisp", "-autoexit", wav_path], check=True)
    elif which("aplay"):
        subprocess.run(["aplay", wav_path], check=True)
    else:
        print("No audio player found (ffplay/aplay). WAV saved at:", wav_path)


def list_midis(root: str, glob_pat: str) -> List[Path]:
    return sorted(Path(root).rglob(glob_pat))


def main() -> None:
    ap = argparse.ArgumentParser(description="Play or render MIDI files using Fluidsynth")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", type=str, help="Path to a single .mid/.midi file")
    g.add_argument("--dir", type=str, help="Directory to scan for MIDI files")

    ap.add_argument("--glob", type=str, default="*.mid", help="Glob pattern when using --dir (default: *.mid)")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of files when using --dir")
    ap.add_argument("--sf2", type=str, default=None, help="Path to a GM SoundFont .sf2")
    ap.add_argument("--samplerate", type=int, default=44100, help="Render sample rate")
    ap.add_argument("--gain", type=float, default=0.8, help="Fluidsynth gain (0..10)")
    ap.add_argument("--keep-wav", action="store_true", help="Keep rendered WAV file(s)")
    ap.add_argument("--outdir", type=str, default="artifacts/audio", help="Directory for rendered WAVs")

    args = ap.parse_args()

    if which("fluidsynth") is None:
        print("fluidsynth not found. Install it and a GM SoundFont (.sf2).", file=sys.stderr)
        sys.exit(1)

    sf2 = find_soundfont(args.sf2)
    if not sf2:
        print("No SoundFont (.sf2) found. Provide one with --sf2 or place it in a standard path.", file=sys.stderr)
        for cand in DEFAULT_SF2_CANDIDATES:
            print("  candidate:", cand, file=sys.stderr)
        sys.exit(1)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    if args.file:
        files = [Path(args.file)]
    else:
        files = list_midis(args.dir, args.glob)
        if args.limit and args.limit > 0:
            files = files[: args.limit]

    if not files:
        print("No MIDI files found.")
        return

    for p in files:
        mid = str(p)
        wav = str(Path(args.outdir) / (p.stem + ".wav"))
        print(f"Rendering {mid} -> {wav} using {sf2}")
        render_to_wav(mid, sf2, wav, samplerate=args.samplerate, gain=args.gain)
        try:
            play_wav(wav)
        finally:
            if not args.keep_wav and Path(wav).exists():
                Path(wav).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
