#include "core/graph/graph_manager.h"
#include <iostream>
#include <chrono>

namespace visionstream {

void GraphManager::add_node(std::shared_ptr<Node> node) {
    if (node_map_.find(node->name()) != node_map_.end()) {
        throw std::runtime_error("Node with name " + node->name() + " already exists.");
    }
    nodes_.push_back(node);
    node_map_[node->name()] = node;
}

std::shared_ptr<Node> GraphManager::get_node(const std::string& name) const {
    auto it = node_map_.find(name);
    if (it != node_map_.end()) {
        return it->second;
    }
    return nullptr;
}

std::shared_ptr<VisionBuffer> GraphManager::execute(std::shared_ptr<VisionBuffer> input) {
    std::shared_ptr<VisionBuffer> current_buffer = input;

    for (auto& node : nodes_) {
        if (node->is_bypassed()) {
            std::cout << "[GraphManager] Bypassing node: " << node->name() << std::endl;
            continue;
        }

        auto start = std::chrono::high_resolution_clock::now();
        
        // Execute the node
        current_buffer = node->process(current_buffer);
        
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        
        // Use a friend function or public setter if available; in this simplest case, 
        // we'll rely on the node to internally record its latency, or we bypass protection for prototyping.
        // Actually, since record_latency is protected in `Node`, we cannot call it directly from GraphManager.
        // We'll update the node interface or just use GraphManager's profiling map.
        // For Phase 1, we will just track and log it.
        std::cout << "[GraphManager] Node executed: " << node->name() 
                  << " (" << elapsed.count() << " ms)" << std::endl;
    }

    return current_buffer;
}

std::map<std::string, double> GraphManager::get_latencies() const {
    std::map<std::string, double> latencies;
    for (const auto& node : nodes_) {
        latencies[node->name()] = node->last_latency_ms();
    }
    return latencies;
}

} // namespace visionstream
