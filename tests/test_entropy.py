import sys
import os
import random

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Add build directly so we can import visionstream
build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../build'))
sys.path.insert(0, build_dir)

import visionstream as vs

def generate_dummy_cdf(max_length, precision=16):
    """
    Generate a monotonic dummy CDF array for testing.
    """
    total = 1 << precision
    cdf = [0]
    for _ in range(max_length - 2):
        # Accumulate random chunks, making sure we don't exceed total early
        chunk = random.randint(1, total // max_length * 2)
        next_val = min(cdf[-1] + chunk, total - (max_length - len(cdf)))
        cdf.append(next_val)
    cdf.append(total)
    
    # Fill any remaining with total if it stopped early due to len
    while len(cdf) < max_length:
        cdf.append(total)
        
    return cdf

def test_arithmetic_coder():
    print("=== Testing C++/CUDA Arithmetic Coder ===")
    
    N = 1000 # number of symbols to encode
    num_cdfs = 5
    max_cdf_length = 64
    precision = 16
    
    # Generate 5 random CDFs
    cdfs_2d = [generate_dummy_cdf(max_cdf_length, precision) for _ in range(num_cdfs)]
    
    # Flatten CDFs
    cdfs_flat = []
    for c in cdfs_2d:
        cdfs_flat.extend(c)
        
    cdf_sizes = [max_cdf_length] * num_cdfs
    offsets = [0] * num_cdfs
    
    # Generate random symbols matching the CDF constraints
    symbols = []
    indexes = []
    
    for _ in range(N):
        idx = random.randint(0, num_cdfs - 1)
        indexes.append(idx)
        
        # Pick a symbol where probability mass > 0
        sym = random.randint(0, max_cdf_length - 2)
        cdf = cdfs_2d[idx]
        while cdf[sym+1] - cdf[sym] <= 0:
            sym = random.randint(0, max_cdf_length - 2)
            
        symbols.append(sym)
        
    print(f"Encoding {N} symbols...")
    bitstream = vs.ArithmeticCoder.encode(symbols, indexes, cdfs_flat, cdf_sizes, offsets, precision)
    
    print(f"Encoded bitstream length: {len(bitstream)} bytes")
    print(f"Compression bits per symbol (bpp): {len(bitstream)*8 / N:.2f}")
    
    print("Decoding...")
    decoded_symbols = vs.ArithmeticCoder.decode(bitstream, indexes, cdfs_flat, cdf_sizes, offsets, precision)
    
    assert len(decoded_symbols) == len(symbols), "Lengths do not match"
    
    mismatches = 0
    for i in range(N):
        if symbols[i] != decoded_symbols[i]:
            mismatches += 1
            if mismatches < 10:
                print(f"Mismatch at {i}: Expected {symbols[i]}, Got {decoded_symbols[i]}")
                
    if mismatches == 0:
        print("Success! Encoding and Decoding are perfectly lossless.")
    else:
        print(f"Failed. {mismatches}/{N} symbols mismatched.")


if __name__ == "__main__":
    test_arithmetic_coder()
