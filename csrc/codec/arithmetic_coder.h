#pragma once

#include <vector>
#include <cstdint>
#include <string>

namespace visionstream {

/**
 * High-performance Arithmetic Coder (Range Coder)
 * Provides both Host (CPU) and future Device (CUDA) accelerated encoding/decoding.
 */
class ArithmeticCoder {
public:
    /**
     * Encode an array of symbols using the provided CDFs.
     * 
     * @param symbols: [N] array of symbols to encode
     * @param indexes: [N] array of indexes indicating which CDF to use for each symbol
     * @param cdfs: Flattened CDFs of shape [num_cdfs, max_cdf_length]
     * @param cdf_sizes: Actual length of each CDF
     * @param offsets: Negative offset mapped to 0 for each CDF
     * @param precision: CDF precision bits (usually 16)
     * @return: Compressed bytes as std::string
     */
    static std::string encode(
        const std::vector<int16_t>& symbols,
        const std::vector<int16_t>& indexes,
        const std::vector<int32_t>& cdfs,
        const std::vector<int32_t>& cdf_sizes,
        const std::vector<int32_t>& offsets,
        int precision = 16
    );

    /**
     * Decode an array of symbols from compressed bytes
     * 
     * @param bitstream: The compressed byte string
     * @param indexes: The spatial/channel indexes specifying which CDF to use for each step
     * @param cdfs, cdf_sizes, offsets, precision: Same as encode
     * @return: Decoded symbols of the same length as `indexes`
     */
    static std::vector<int16_t> decode(
        const std::string& bitstream,
        const std::vector<int16_t>& indexes,
        const std::vector<int32_t>& cdfs,
        const std::vector<int32_t>& cdf_sizes,
        const std::vector<int32_t>& offsets,
        int precision = 16
    );

    // TODO: Expose explicit batched CUDA rANS / AC operations later 
};

} // namespace visionstream
