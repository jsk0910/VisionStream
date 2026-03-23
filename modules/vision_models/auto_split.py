"""
Universal Auto-Splitter module using PyTorch FX for introspecting
and partitioning arbitrary PyTorch models.
"""
import torch
import torch.nn as nn
import torch.fx as fx
from typing import List, Dict, Union
from torch.fx.passes.split_module import split_module


class AutoSplitter:
    """
    Analyzes and partitions any traced PyTorch model into sub-modules.
    Useful for creating flexible split points from a UI.
    """
    def __init__(self, model: nn.Module):
        """
        Args:
            model: The standard PyTorch module to be analyzed and split.
        """
        self.original_model = model
        self.original_model.eval()
        
        # Trace the model into a GraphModule
        try:
            self.traced = fx.symbolic_trace(model)
        except Exception as e:
            raise RuntimeError(
                f"Failed to symbolically trace the model. "
                f"Ensure the model is FX-traceable. Error: {e}"
            )

    def get_ui_graph_structure(self) -> List[Dict[str, Union[str, List[str]]]]:
        """
        Extracts structural data of the traced model graph suitable for UI rendering.
        
        Returns:
            A list of dictionary objects representing nodes (layers/operations).
            Each node contains its 'id' (name), 'op_type' (call_module, call_function, etc.),
            'target' (underlying class/function name), and 'prev_nodes' (dependencies).
        """
        nodes_info = []
        for node in self.traced.graph.nodes:
            # We skip highly internal placeholder or output nodes if they clutter,
            # but usually they are necessary to visualize data flow cleanly.
            prev_nodes = [n.name for n in node.args if isinstance(n, fx.Node)]
            
            node_data = {
                "id": node.name,
                "op_type": node.op,
                "target": str(node.target) if not callable(node.target) else node.target.__name__,
                "prev_nodes": prev_nodes
            }
            nodes_info.append(node_data)
            
        return nodes_info

    def create_split_modules(self, split_node_names: List[str]) -> nn.ModuleList:
        """
        Splits the graph into multiple sequential/parallel independent parts
        based on the provided split points.
        
        Args:
            split_node_names: List of node names that act as boundaries.
                              E.g. ['conv1', 'layer2.0.relu']
                              A boundary node is included in the partition BEFORE the cut.
                              So Partition 0 holds nodes up to split_node_names[0],
                              Partition 1 holds nodes from the next node up to split_node_names[1], etc.
                              
        Returns:
            nn.ModuleList containing sub-GraphModules.
            Usually: [Part_0, Part_1, ... Part_N]
        """
        # Create a mapping dictionary: Node -> partition index
        partition_map: Dict[fx.Node, int] = {}
        
        current_part = 0
        split_set = set(split_node_names)
        
        # Traverse the list of nodes sequentially
        # Note: In PyTorch FX, nodes are topologically sorted.
        for node in self.traced.graph.nodes:
            partition_map[node] = current_part
            
            if node.name in split_set:
                # the current node itself is assigned to the current partition,
                # but the following nodes belong to the next partition.
                current_part += 1
                split_set.remove(node.name)
                
        if len(split_set) > 0:
            raise ValueError(f"Could not find some split nodes in the graph: {split_set}")

        # Use split_module to do the heavy lifting
        # It takes the traced model, the partition_map, and returns a dict of separated models
        split_res = split_module(
            self.traced, 
            None, 
            lambda node: partition_map[node]
        )
        
        # The result is a larger GraphModule containing submodules.
        # It typically names submodules 'submod_0', 'submod_1', etc.
        # We can extract them into a ModuleList.
        
        sub_modules = []
        for i in range(current_part + 1):
            submod_name = f"submod_{i}"
            if hasattr(split_res, submod_name):
                sub_modules.append(getattr(split_res, submod_name))
            else:
                # If a partition had no nodes, it might be missing
                raise RuntimeError(f"Expected extracted partition {submod_name} but it was missing.")

        return nn.ModuleList(sub_modules)
