#pragma once
#include <vector>
#include <cstdint>
#include <memory>

namespace visionstream {

/**
 * Representation of an Entropy Model containing 
 * Cumulative Distribution Functions (CDFs) used for Arithmetic Coding.
 */
struct EntropyModel {
    // CDFs shape: [num_cdfs, cdf_length]
    // Flattened array for fast C++ / CUDA memory access
    std::vector<int32_t> cdfs;
    
    // Size of each CDF (useful if variable length, though typically padded)
    std::vector<int32_t> cdf_sizes; 
    
    // Symbol offset for each CDF (to handle negative symbols)
    std::vector<int32_t> offsets;   

    int num_cdfs;
    int max_cdf_length;
    int precision; // typically 16 bits

    EntropyModel() : num_cdfs(0), max_cdf_length(0), precision(16) {}
    
    // Helper to get CDF pointer for a specific index
    const int32_t* get_cdf(int index) const {
        return cdfs.data() + (index * max_cdf_length);
    }
};

} // namespace visionstream
