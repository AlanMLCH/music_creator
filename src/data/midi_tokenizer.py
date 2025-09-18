from typing import List, Tuple
import miditoolkit


PAD = 0
BOS = 1
EOS = 2
NOTE_ON_BASE = 3 # 3..130
NOTE_OFF_BASE = 131 # 131..258
TIME_SHIFT_BASE = 259 # 259..(259+TIME_BINS-1)
TIME_BINS = 125 # 10ms bins => up to 1.24s per token
VOCAB_SIZE = TIME_SHIFT_BASE + TIME_BINS # 384


def note_on(p: int) -> int: return NOTE_ON_BASE + int(p)


def note_off(p: int) -> int: return NOTE_OFF_BASE + int(p)


def time_shift_bin(delta_ms: float) -> int:
    steps = int(max(0, min(TIME_BINS - 1, round(delta_ms / 10.0))))
    return TIME_SHIFT_BASE + steps


def is_time_shift(tok: int) -> bool: return tok >= TIME_SHIFT_BASE


# MIDI -> tokens (very simple; consider REMI for production)


def midi_to_tokens(path: str) -> List[int]:
    midi = miditoolkit.midi.parser.MidiFile(path)
    events: List[Tuple[float, int]] = []


    # crude tempo handling; assumes single tempo if present
    tempo = midi.tempo_changes[0].tempo if midi.tempo_changes else 120
    tpq = midi.ticks_per_beat
    ms_per_beat = 60000 / tempo


    def tick_to_ms(tick: int) -> float:
        beats = tick / tpq
        return beats * ms_per_beat


    for inst in midi.instruments:
        for n in inst.notes:
            start_ms = tick_to_ms(n.start)
            end_ms = tick_to_ms(n.end)
            events.append((start_ms, note_on(n.pitch)))
            events.append((end_ms, note_off(n.pitch)))


    events.sort(key=lambda x: (x[0], 0 if NOTE_OFF_BASE <= x[1] < TIME_SHIFT_BASE else 1))


    tokens: List[int] = [BOS]
    last_t = 0.0
    for t_ms, tok in events:
        delta = max(0.0, t_ms - last_t)
        if delta > 0:
            tokens.append(time_shift_bin(delta))
        tokens.append(tok)
        last_t = t_ms
    tokens.append(EOS)
    return tokens


# tokens -> MIDI


def tokens_to_midi(tokens: List[int], out_path: str) -> None:
    midi = miditoolkit.midi.parser.MidiFile()
    inst = miditoolkit.Instrument(program=0, is_drum=False, name="piano")
    midi.instruments = [inst]


    cur_ms = 0.0
    on_dict = {}
    tempo = 120
    tpq = 480
    midi.ticks_per_beat = tpq
    midi.tempo_changes = [miditoolkit.TempoChange(tempo=tempo, time=0)]


    def ms_to_tick(ms: float) -> int:
        beat_ms = 60000 / tempo
        beats = ms / beat_ms
        return int(round(beats * tpq))


    for tok in tokens:
        if tok in (BOS, PAD):
            continue
        if tok == EOS:
            break
        if is_time_shift(tok):
            steps = tok - TIME_SHIFT_BASE
            cur_ms += steps * 10.0
            continue
        if NOTE_ON_BASE <= tok < NOTE_OFF_BASE:
            pitch = tok - NOTE_ON_BASE
            if pitch not in on_dict:
                on_dict[pitch] = cur_ms
            continue
        if NOTE_OFF_BASE <= tok < TIME_SHIFT_BASE:
            pitch = tok - NOTE_OFF_BASE
            if pitch in on_dict:
                start_ms = on_dict.pop(pitch)
                n = miditoolkit.Note(velocity=80, pitch=pitch,
                                    start=ms_to_tick(start_ms), end=ms_to_tick(cur_ms))
                inst.notes.append(n)
    midi.instruments[0] = inst
    midi.dump(out_path)