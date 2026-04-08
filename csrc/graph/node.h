#ifndef VISIONSTREAM_CORE_GRAPH_NODE_H
#define VISIONSTREAM_CORE_GRAPH_NODE_H

#include "core/memory/vision_buffer.h"
#include <memory>
#include <string>
#include <chrono>

namespace visionstream {

class Node {
public:
    Node(const std::string& name) : name_(name), is_bypassed_(false), last_latency_ms_(0.0) {}
    virtual ~Node() = default;

    // Interface that all nodes (including Python custom nodes) must implement
    virtual std::shared_ptr<VisionBuffer> process(std::shared_ptr<VisionBuffer> input) = 0;

    // Node Toggle functionality
    bool is_bypassed() const { return is_bypassed_; }
    void set_bypassed(bool bypass) { is_bypassed_ = bypass; }

    // Metadata & metrics
    const std::string& name() const { return name_; }
    double last_latency_ms() const { return last_latency_ms_; }

protected:
    void record_latency(double ms) { last_latency_ms_ = ms; }

    std::string name_;
    bool is_bypassed_;
    double last_latency_ms_;
};

} // namespace visionstream

#endif // VISIONSTREAM_CORE_GRAPH_NODE_H
