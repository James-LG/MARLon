from enum import Enum
from numpy import mean

class NodeType(Enum):
    NORMAL = 1
    INPUT = 2
    OUTPUT = 3

class Connection():
    def __init__(self, input_node, output_node, weight, is_enabled):
        self.input_node = input_node
        self.output_node = output_node
        self.weight = weight
        self.is_enabled = is_enabled

class Node():
    def __init__(self, node_type):
        self.node_type = node_type
        self.value = None
        self.inputs = []
        self.outputs = []

class RandomModel():
    def __init__(self, n_iterations, n_inputs, n_outputs):
        self.n_iterations = n_iterations
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.nodes = []
        self.connections = []

        output_nodes = []
        for i in range(n_outputs):
            output_node = Node(NodeType.OUTPUT)
            output_nodes.append(output_node)

        for i in range(n_inputs):
            input_node = Node(NodeType.INPUT)
            self.nodes.append(input_node)

            for output_node in output_nodes:
                connection = Connection(input_node, output_node, 1.0, True)
                input_node.outputs.append(connection)
                output_node.inputs.append(connection)
                self.connections.append(connection)

        self.nodes += output_nodes
        pass

    def predict(self, input_values):
        input_nodes = self.get_input_nodes()
        assert len(input_values) == len(input_nodes)

        for node in self.nodes:
            node.value = None

        for con in self.connections:
            con.value = None

        for val, i_node in zip(input_values, input_nodes):
            i_node.value = val

        def all_inputs_ready(node):
            for input_con in node.inputs:
                if not input_con.input_node.value:
                    return False
            return True

        outputs = []
        while len(outputs) < self.n_outputs:
            for con in self.connections:
                print('con.input_node.value', con.input_node.value)
                if not con.output_node.value and con.input_node.value and all_inputs_ready(con.output_node):
                    input_values = [x.input_node.value for x in con.output_node.inputs]
                    print('input_values', input_values)
                    con.output_node.value = mean(input_values)

                    if con.output_node.node_type == NodeType.OUTPUT:
                        outputs.append(con.output_node.value)
        
        return outputs

    def get_input_nodes(self):
        return list(filter(lambda x: x.node_type == NodeType.INPUT, self.nodes))

    def train(self, env):
        for i in range(self.n_iterations):
            continue

if __name__ == "__main__":
    model = RandomModel(1, 4, 2)
    print('output', model.predict([2, 4, 6, 8]))