# 
# 样例来自于
# https://medium.com/@tansiahuat/mastering-pytorch-graphmodule-how-to-extract-subgraphs-efficiently-072a8aeaa436
import torch
import copy
from typing import List

def extract_subgraph(
    gm: torch.fx.GraphModule,
    input_names: List[str],
    output_names: List[str],
) -> torch.fx.GraphModule:

    # Assert that input and output names are valid
    assert set(input_names + output_names).issubset(
        {node.name for node in gm.graph.nodes}
    )

    # Copy to avoid modifying the original graph
    gm = copy.deepcopy(gm)

    # Set new inputs
    for node in gm.graph.nodes:
        if node.name in input_names:
            node.op, node.target, node.args, node.kwargs = 'placeholder', node.name, (), {}
    
    # Set new outputs
    output_node = gm.graph.find_nodes(op='output')[0]
    output_node.args = (tuple([node for node in gm.graph.nodes if node.name in output_names]),)

    # Eliminate dead code
    gm.graph.eliminate_dead_code()

    # Remove unused placeholders
    for node in gm.graph.find_nodes(op='placeholder'):
        if node.name not in input_names:
            gm.graph.erase_node(node)

    return torch.fx.GraphModule(gm, gm.graph)


# Define a simple model
model = torch.nn.Sequential(
 torch.nn.Linear(2, 4),
 torch.nn.Linear(4, 8),
 torch.nn.Linear(8, 1),
)

# Export the model to GraphModule
gm = torch.export.export(model, (torch.randn(1, 2),)).module()

# gm.graph.print_tabular()

# Find the linear nodes
linear_nodes = gm.graph.find_nodes(op='call_function', target=torch.ops.aten.linear.default)
assert len(linear_nodes) == 3

# In this example, we want to extract the second linear layer as a subgraph
# So the input should be the output of the first linear layer
# And the output should be the output of the second linear layer
sub_gm = extract_subgraph(gm, [linear_nodes[0].name], [linear_nodes[1].name])
sub_gm.graph.print_tabular()