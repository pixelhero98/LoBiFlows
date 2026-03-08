"""Download and prepare FI-2010 LOB data for LoBiFlow experiments.

FI-2010 from fairdata.fi stores data transposed: each file is (149 x T)
where rows 0..39 are the L2 LOB features in the layout:
    ask_p1, ask_v1, bid_p1, bid_v1, ask_p2, ask_v2, bid_p2, bid_v2, ...
    (i.e. interleaved per level).

We re-arrange to (T x 40) in the layout expected by LoBiFlow:
    ask_p1..ask_p10, ask_v1..ask_v10, bid_p1..bid_p10, bid_v1..bid_v10
"""

import os
import io
import zipfile
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "data_fi2010")
os.makedirs(OUT_DIR, exist_ok=True)

# The FI-2010 txt files use the layout (per level group of 4 rows):
#   ask_price_level_i, ask_vol_level_i, bid_price_level_i, bid_vol_level_i
# for i = 1..10, giving rows 0..39 of the 149-row matrix.

# We'll try multiple sources
URLS = [
    "https://raw.githubusercontent.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/master/data/data.zip",
]

def download_deeplob_data():
    """Download from DeepLOB GitHub (common community source)."""
    import urllib.request
    for url in URLS:
        print(f"Trying: {url}")
        try:
            resp = urllib.request.urlopen(url, timeout=60)
            data = resp.read()
            print(f"Downloaded {len(data)} bytes")
            return data
        except Exception as e:
            print(f"  Failed: {e}")
    return None

def extract_fi2010_from_deeplob_zip(zip_bytes):
    """Extract and convert FI-2010 data from DeepLOB zip format."""
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    print(f"Zip contents: {zf.namelist()}")
    
    # Look for the main data files
    all_data = []
    for name in sorted(zf.namelist()):
        if name.endswith('.txt') and 'NoAuction' in name:
            print(f"  Loading: {name}")
            with zf.open(name) as f:
                arr = np.loadtxt(f)
            print(f"    Shape: {arr.shape}")
            all_data.append((name, arr))
    
    return all_data

def rearrange_fi2010_to_lobiflow(arr, levels=10):
    """Convert FI-2010 interleaved layout to LoBiFlow layout.
    
    Input layout (rows): ask_p1, ask_v1, bid_p1, bid_v1, ask_p2, ask_v2, ...
    Output layout (cols): ask_p1..pL, ask_v1..vL, bid_p1..pL, bid_v1..vL
    """
    # FI-2010 is (149 x T) or (40 x T) — rows are features, columns are time
    # First 40 rows are L2 LOB in interleaved format
    if arr.shape[0] < arr.shape[1]:
        # Transposed: (features x time) -> (time x features)
        lob_features = arr[:40, :].T  # (T, 40)
    else:
        lob_features = arr[:, :40]  # (T, 40)
    
    T = lob_features.shape[0]
    L = levels
    
    # Input interleaved: ask_p1, ask_v1, bid_p1, bid_v1, ask_p2, ...
    ask_p = np.zeros((T, L), dtype=np.float32)
    ask_v = np.zeros((T, L), dtype=np.float32)
    bid_p = np.zeros((T, L), dtype=np.float32)
    bid_v = np.zeros((T, L), dtype=np.float32)
    
    for i in range(L):
        ask_p[:, i] = lob_features[:, 4*i + 0]
        ask_v[:, i] = lob_features[:, 4*i + 1]
        bid_p[:, i] = lob_features[:, 4*i + 2]
        bid_v[:, i] = lob_features[:, 4*i + 3]
    
    # Output layout: ask_p1..pL, ask_v1..vL, bid_p1..pL, bid_v1..vL
    out = np.concatenate([ask_p, ask_v, bid_p, bid_v], axis=1)  # (T, 4*L)
    return out.astype(np.float32)


if __name__ == "__main__":
    print("Downloading FI-2010 data...")
    data = download_deeplob_data()
    
    if data is None:
        print("ERROR: Could not download FI-2010 data.")
        exit(1)
    
    files = extract_fi2010_from_deeplob_zip(data)
    
    if not files:
        # Try loading the zip as a single array
        zf = zipfile.ZipFile(io.BytesIO(data))
        for name in sorted(zf.namelist()):
            if name.endswith(('.npy', '.txt', '.csv')):
                print(f"  Found: {name}")
                with zf.open(name) as f:
                    if name.endswith('.npy'):
                        arr = np.load(f)
                    else:
                        arr = np.loadtxt(f)
                print(f"    Shape: {arr.shape}")
                files.append((name, arr))
    
    for name, arr in files:
        print(f"\nProcessing: {name} shape={arr.shape}")
        lob = rearrange_fi2010_to_lobiflow(arr, levels=10)
        print(f"  Rearranged to: {lob.shape}")
        
        # Basic sanity
        print(f"  Ask price range: [{lob[:, 0].min():.4f}, {lob[:, 0].max():.4f}]")
        print(f"  Bid price range: [{lob[:, 20].min():.4f}, {lob[:, 20].max():.4f}]")
        print(f"  Ask vol range:   [{lob[:, 10].min():.4f}, {lob[:, 10].max():.4f}]")
        
        base = os.path.splitext(os.path.basename(name))[0]
        out_path = os.path.join(OUT_DIR, f"{base}_lobiflow.npy")
        np.save(out_path, lob)
        print(f"  Saved: {out_path}")
    
    # Also save a single merged file
    if len(files) > 1:
        merged = np.concatenate([rearrange_fi2010_to_lobiflow(arr, 10) for _, arr in files], axis=0)
        merged_path = os.path.join(OUT_DIR, "fi2010_merged_lobiflow.npy")
        np.save(merged_path, merged)
        print(f"\nMerged all: {merged.shape} -> {merged_path}")
    elif len(files) == 1:
        merged_path = os.path.join(OUT_DIR, "fi2010_merged_lobiflow.npy")
        np.save(merged_path, rearrange_fi2010_to_lobiflow(files[0][1], 10))
        print(f"\nSingle file saved as merged: {merged_path}")
    
    print("\nDone!")
