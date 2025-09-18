import argparse
from xml.parsers.expat import model
import torch
from typing import List
from src.models.transformer_gpt import CausalTransformerLM
from src.utils.positional_encoding import SinusoidalPositionalEncoding
from src.data.midi_tokenizer import tokens_to_midi, VOCAB_SIZE, BOS, EOS


def sample(model, device, start_token: int, max_len: int, temperature: float = 1.0) -> List[int]:
    model.eval()
    x = torch.tensor([[start_token]], dtype=torch.long, device=device)
    out = [start_token]
    with torch.no_grad():
        for _ in range(max_len):
            T = x.size(1)
            mask = model.causal_mask(T, device)
            logits = model(x, tgt_mask=mask)[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            tok = int(next_tok.item())
            out.append(tok)
            if tok == EOS:
                break
            x = torch.cat([x, next_tok], dim=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--length', type=int, default=1500)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--out', type=str, default='artifacts/samples/sample.midi')
    args = ap.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CausalTransformerLM(vocab_size=VOCAB_SIZE, d_model=512, nhead=8, num_layers=6,
    dim_feedforward=2048, dropout=0.1).to(device)
    model.pos = SinusoidalPositionalEncoding(512).to(device)
    sd = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(sd)


    toks = sample(model, device, start_token=BOS, max_len=args.length, temperature=args.temperature)
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tokens_to_midi(toks, args.out)
    print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()