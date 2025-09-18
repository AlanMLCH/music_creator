# scripts/inspect_midis.py
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pretty_midi


def human_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def safe_mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def instrument_name(inst: pretty_midi.Instrument) -> str:
    if inst.is_drum:
        return "Drums"
    try:
        return pretty_midi.program_to_instrument_name(inst.program)
    except Exception:
        return inst.name or f"Program_{inst.program}"


def polyphony_stats(instruments: List[pretty_midi.Instrument]) -> Tuple[int, float]:
    """Return (max_polyphony, avg_polyphony) across the track."""
    events = []
    for inst in instruments:
        for n in inst.notes:
            events.append((n.start, +1))
            events.append((n.end, -1))
    if not events:
        return 0, 0.0
    events.sort()
    cur = 0
    max_poly = 0
    area = 0.0
    last_t = events[0][0]
    for t, delta in events:
        dt = t - last_t
        if dt > 1e-9:
            area += cur * dt
            last_t = t
        cur += delta
        if cur > max_poly:
            max_poly = cur
    duration = events[-1][0] - events[0][0]
    avg_poly = area / duration if duration > 0 else float(cur)
    return max_poly, avg_poly


def analyze_file(path: Path) -> Dict:
    m = pretty_midi.PrettyMIDI(str(path))
    duration = m.get_end_time()
    tempos, tempo_times = m.get_tempo_changes()
    tsigs = m.time_signature_changes

    inst_summaries = []
    total_notes = 0
    all_pitches = []
    all_velocities = []

    for inst in m.instruments:
        nn = len(inst.notes)
        total_notes += nn
        pitches = [n.pitch for n in inst.notes]
        vels = [n.velocity for n in inst.notes]
        all_pitches.extend(pitches)
        all_velocities.extend(vels)

        inst_summaries.append(
            dict(
                name=instrument_name(inst),
                notes=nn,
                min_pitch=min(pitches) if pitches else None,
                max_pitch=max(pitches) if pitches else None,
                avg_velocity=safe_mean(vels),
            )
        )

    max_poly, avg_poly = polyphony_stats(m.instruments)
    notes_per_sec = total_notes / duration if duration > 0 else 0.0

    return dict(
        file=str(path),
        size_bytes=path.stat().st_size,
        duration_sec=duration,
        n_instruments=len(m.instruments),
        total_notes=total_notes,
        notes_per_sec=notes_per_sec,
        max_polyphony=max_poly,
        avg_polyphony=avg_poly,
        n_tempo_changes=len(tempos),
        tempos=list(map(float, tempos)),
        n_time_sig_changes=len(tsigs),
        time_sigs=[f"{ts.numerator}/{ts.denominator}@{ts.time:.2f}s" for ts in tsigs],
        instruments=inst_summaries,
    )


def print_report(info: Dict) -> None:
    p = Path(info["file"])
    print("=" * 80)
    print(f"File: {p.name}  ({human_size(info['size_bytes'])})")
    print(f"Path: {p}")
    print(f"Duration: {info['duration_sec']:.2f}s")
    print(f"Instruments: {info['n_instruments']} | Total notes: {info['total_notes']} | Notes/sec: {info['notes_per_sec']:.2f}")
    print(f"Polyphony: max={info['max_polyphony']} avg={info['avg_polyphony']:.2f}")
    print(f"Tempo changes: {info['n_tempo_changes']} | Tempos (first 5): {info['tempos'][:5]}")
    print(f"Time signatures: {info['n_time_sig_changes']} | {', '.join(info['time_sigs'][:5]) if info['time_sigs'] else '-'}")
    print("- Instruments:")
    for inst in info["instruments"]:
        print(
            f"  - {inst['name']:<20}  notes={inst['notes']:>5}  "
            f"pitch=[{inst['min_pitch']},{inst['max_pitch']}]  "
            f"avg_vel={inst['avg_velocity']:.1f}"
        )


def main():
    ap = argparse.ArgumentParser(description="Inspect generated MIDI files and print stats.")
    ap.add_argument("--dir", type=str, default="artifacts/samples", help="Directory with .mid files")
    ap.add_argument("--glob", type=str, default="*.mid", help="Glob pattern (default: *.mid)")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of files to inspect")
    ap.add_argument("--to-csv", type=str, default="", help="Optional: write per-file summary to CSV")
    args = ap.parse_args()

    root = Path(args.dir)
    paths = sorted(root.rglob(args.glob))
    if args.limit and args.limit > 0:
        paths = paths[: args.limit]

    if not paths:
        print(f"No files matched {root}/{args.glob}")
        return

    rows = []
    for p in paths:
        try:
            info = analyze_file(p)
            print_report(info)
            rows.append(
                {
                    "file": info["file"],
                    "size_bytes": info["size_bytes"],
                    "duration_sec": round(info["duration_sec"], 3),
                    "n_instruments": info["n_instruments"],
                    "total_notes": info["total_notes"],
                    "notes_per_sec": round(info["notes_per_sec"], 3),
                    "max_polyphony": info["max_polyphony"],
                    "avg_polyphony": round(info["avg_polyphony"], 3),
                    "n_tempo_changes": info["n_tempo_changes"],
                    "n_time_sig_changes": info["n_time_sig_changes"],
                }
            )
        except Exception as e:
            print(f"[ERROR] {p}: {e}")

    if args.to_csv:
        try:
            import csv

            out = Path(args.to_csv)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=list(rows[0].keys()) if rows else [
                        "file","size_bytes","duration_sec","n_instruments",
                        "total_notes","notes_per_sec","max_polyphony",
                        "avg_polyphony","n_tempo_changes","n_time_sig_changes"
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)
            print(f"Wrote CSV: {out}")
        except Exception as e:
            print(f"[ERROR] writing CSV: {e}")


if __name__ == "__main__":
    main()
