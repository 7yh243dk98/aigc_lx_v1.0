import scipy.io.wavfile as wavfile
import numpy as np, os

for folder in ["output/compare", "output/e2e_generated", "output/verify_fix"]:
    path = os.path.join(r"D:\pyprojects\aigc-m", folder)
    if not os.path.exists(path):
        continue
    print(f"\n=== {folder} ===")
    for f in sorted(os.listdir(path)):
        if f.endswith('.wav'):
            sr, audio = wavfile.read(os.path.join(path, f))
            peak = np.max(np.abs(audio))
            # int16 range: 0-32767
            print(f"  {f:30s} peak_int16={peak:6d}  ({peak/32767*100:.1f}%)")
