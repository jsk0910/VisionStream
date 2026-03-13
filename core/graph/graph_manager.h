#ifndef VISIONSTREAM_CORE_GRAPH_GRAPH_MANAGER_H
#define VISIONSTREAM_CORE_GRAPH_GRAPH_MANAGER_H

#include "core/graph/node.h"
#include <vector>
#include <memory>
#include <map>
#include <string>

namespace visionstream {

class GraphManager {
public:
    GraphManager() = default;
    ~GraphManager() = default;

    void add_node(std::shared_ptr<Node> node);
    std::shared_ptr<Node> get_node(const std::string& name) const;
    
    // Executes the nodes sequentially.
    // In later phases, this will be expanded to an asynchronous DAG execution.
    std::shared_ptr<VisionBuffer> execute(std::shared_ptr<VisionBuffer> input);

    // Metric reporting
    std::map<std::string, double> get_latencies() const;

private:
    std::vector<std::shared_ptr<Node>> nodes_;
    std::map<std::string, std::shared_ptr<Node>> node_map_;
};

} // namespace visionstream

#endif // VISIONSTREAM_CORE_GRAPH_GRAPH_MANAGER_H
