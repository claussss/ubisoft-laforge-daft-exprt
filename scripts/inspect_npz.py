import numpy as np
import sys

def inspect_npz(path):
    try:
        data = np.load(path)
        print(f"Keys: {list(data.keys())}")
        if 'mel_spec' in data:
            mel = data['mel_spec']
            print(f"Shape: {mel.shape}")
            print(f"Min: {mel.min()}")
            print(f"Max: {mel.max()}")
            print(f"Mean: {mel.mean()}")
            print(f"Std: {mel.std()}")
            
            # Check for silence/clipping
            print(f"Values < -10: {np.sum(mel < -10)}")
            print(f"Values > 10: {np.sum(mel > 10)}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_npz(sys.argv[1])
