#include "arithmetic_coder.h"
#include <stdexcept>
#include <iostream>

namespace visionstream {

// Basic 32-bit Range Encoder Implementation
// Inspired by standard Martin algorithm and torchac
class RangeEncoder {
public:
    RangeEncoder() : low_(0), range_(0xFFFFFFFF), cache_(0), cache_size_(0), stream_() {}

    void encode(uint32_t cum_freq, uint32_t freq, uint32_t total_freq) {
        low_ += cum_freq * (range_ / total_freq);
        range_ = freq * (range_ / total_freq);

        while (range_ < (1U << 24)) {
            range_ <<= 8;
            shift_low();
        }
    }

    std::string finish() {
        for (int i = 0; i < 5; ++i) {
            shift_low();
        }
        return stream_;
    }

private:
    void shift_low() {
        uint32_t carry = low_ >> 32;
        uint32_t low32 = static_cast<uint32_t>(low_);
        
        if (carry != 0 || low32 < 0xFF000000) {
            if (cache_size_ > 0) {
                stream_.push_back(static_cast<char>(cache_ + carry));
                for (uint32_t i = 0; i < cache_size_ - 1; ++i) {
                    stream_.push_back(static_cast<char>(carry ? 0x00 : 0xFF));
                }
            }
            cache_ = low32 >> 24;
            cache_size_ = 1;
        } else {
            cache_size_++;
        }
        low_ = (low32 << 8) & 0xFFFFFFFF;
    }

    uint64_t low_;
    uint32_t range_;
    uint32_t cache_;
    uint32_t cache_size_;
    std::string stream_;
};

class RangeDecoder {
public:
    RangeDecoder(const std::string& stream) : stream_(stream), ptr_(0), code_(0), range_(0xFFFFFFFF) {
        for (int i = 0; i < 4; ++i) {
            code_ = (code_ << 8) | read_byte();
        }
    }

    uint32_t get_cum_freq(uint32_t total_freq) {
        return code_ / (range_ / total_freq);
    }

    void decode(uint32_t cum_freq, uint32_t freq, uint32_t total_freq) {
        code_ -= cum_freq * (range_ / total_freq);
        range_ = freq * (range_ / total_freq);

        while (range_ < (1U << 24)) {
            range_ <<= 8;
            code_ = (code_ << 8) | read_byte();
        }
    }

private:
    uint32_t read_byte() {
        if (ptr_ < stream_.size()) {
            return static_cast<uint8_t>(stream_[ptr_++]);
        }
        return 0;
    }

    const std::string& stream_;
    size_t ptr_;
    uint32_t code_;
    uint32_t range_;
};

std::string ArithmeticCoder::encode(
    const std::vector<int16_t>& symbols,
    const std::vector<int16_t>& indexes,
    const std::vector<int32_t>& cdfs,
    const std::vector<int32_t>& cdf_sizes,
    const std::vector<int32_t>& offsets,
    int precision
) {
    if (symbols.size() != indexes.size()) {
        throw std::invalid_argument("symbols and indexes must be the same size.");
    }

    uint32_t total_freq = 1 << precision;
    RangeEncoder encoder;

    for (size_t i = 0; i < symbols.size(); ++i) {
        int idx = indexes[i];
        int sym = symbols[i];
        int offset = offsets[idx];
        
        int cdf_idx = sym - offset;
        int max_len = cdf_sizes[idx];
        
        // Handle out-of-bounds symbols gracefully (clip)
        if (cdf_idx < 0) cdf_idx = 0;
        if (cdf_idx >= max_len - 1) cdf_idx = max_len - 2;

        int max_cdf_length = cdfs.size() / offsets.size();
        const int32_t* cdf = cdfs.data() + (idx * max_cdf_length);

        uint32_t cdf_low = cdf[cdf_idx];
        uint32_t cdf_high = cdf[cdf_idx + 1];
        uint32_t freq = cdf_high - cdf_low;

        // Validate frequency to prevent division by zero or infinite loop
        if (freq == 0 || freq >= total_freq) {
             throw std::runtime_error("Invalid CDF step encountered. Probability mass is <= 0 or too large.");
        }

        encoder.encode(cdf_low, freq, total_freq);
    }

    return encoder.finish();
}

std::vector<int16_t> ArithmeticCoder::decode(
    const std::string& bitstream,
    const std::vector<int16_t>& indexes,
    const std::vector<int32_t>& cdfs,
    const std::vector<int32_t>& cdf_sizes,
    const std::vector<int32_t>& offsets,
    int precision
) {
    uint32_t total_freq = 1 << precision;
    RangeDecoder decoder(bitstream);
    
    std::vector<int16_t> out_symbols(indexes.size());

    int max_cdf_length = cdfs.size() / offsets.size();

    for (size_t i = 0; i < indexes.size(); ++i) {
        int idx = indexes[i];
        int max_len = cdf_sizes[idx];
        const int32_t* cdf = cdfs.data() + (idx * max_cdf_length);

        uint32_t target = decoder.get_cum_freq(total_freq);

        if (target >= total_freq) {
             target = total_freq - 1;
        }

        // Binary search or linear search for the symbol in CDF using target
        // For small CDFs, linear search is fine. Since this is high-perf, we can linear scan.
        int cdf_idx = 0;
        for (; cdf_idx < max_len - 1; ++cdf_idx) {
            if (target < static_cast<uint32_t>(cdf[cdf_idx + 1])) {
                break;
            }
        }

        if (cdf_idx == max_len - 1) {
            cdf_idx = max_len - 2;
        }

        uint32_t cdf_low = cdf[cdf_idx];
        uint32_t cdf_high = cdf[cdf_idx + 1];
        uint32_t freq = cdf_high - cdf_low;

        if (freq == 0) freq = 1; // Failsafe

        decoder.decode(cdf_low, freq, total_freq);
        out_symbols[i] = static_cast<int16_t>(cdf_idx + offsets[idx]);
    }

    return out_symbols;
}

} // namespace visionstream
