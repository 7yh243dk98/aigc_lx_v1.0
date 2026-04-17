import torch
from collections import Counter
d = torch.load('res/emopia_tokens/emopia_tokens.pt', weights_only=False)
print(f"samples: {len(d)}")
c = Counter(x['emotion_id'] for x in d)
print(f"emotions: {dict(sorted(c.items()))}")
s = d[0]
codes = s['audio_codes']
print(f"codes shape: {codes.shape}, range: [{codes.min()}, {codes.max()}]")
print(f"emotion_id type: {type(s['emotion_id'])}, value: {s['emotion_id']}")
