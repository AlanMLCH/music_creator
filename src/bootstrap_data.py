# Optional: copy a few demo MIDIs into ./data/midi for a smoke test
import shutil
from pathlib import Path


SRC = Path('demo_midis')
DST = Path('data/midi')


DST.mkdir(parents=True, exist_ok=True)
if SRC.exists():
    for p in SRC.glob('*.midi'):
        shutil.copy(p, DST / p.name)
        print('copied', p)
else:
    print('Place some .mid files into data/midi/')