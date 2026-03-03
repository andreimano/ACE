import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from e3nn.nn import BatchNorm
from e3nn.o3 import Irreps

from .o3_building_blocks import O3TensorProduct, O3TensorProductSwishGate
from .instance_norm import InstanceNorm


class SEGNN(nn.Module):
    """Steerable E(3) equivariant message passing network"""

    def __init__(
        self,
        input_irreps,
        hidden_irreps,
        output_irreps,
        edge_attr_irreps,
        node_attr_irreps,
        num_layers,
        norm=None,
        pool="avg",
        task="graph",
        additional_message_irreps=None,
        args=None,
    ):
        super().__init__()
        self.task = task

        self.embedding_layer = O3TensorProduct(
            input_irreps, hidden_irreps, node_attr_irreps
        )

        # Message passing layers.
        layers = []
        for i in range(num_layers):
            layers.append(
                SEGNNLayer(
                    hidden_irreps,
                    hidden_irreps,
                    hidden_irreps,
                    edge_attr_irreps,
                    node_attr_irreps,
                    norm=norm,
                    additional_message_irreps=additional_message_irreps,
                    args=args,
                )
            )
        self.layers = nn.ModuleList(layers)

        # Prepare for output irreps, since the attrs will disappear after pooling
        if task == "graph":
            pooled_irreps = (
                (output_irreps * hidden_irreps.num_irreps).simplify().sort().irreps
            )
            self.pre_pool1 = O3TensorProductSwishGate(
                hidden_irreps, hidden_irreps, node_attr_irreps
            )
            self.pre_pool2 = O3TensorProduct(
                hidden_irreps, pooled_irreps, node_attr_irreps
            )
            self.post_pool1 = O3TensorProductSwishGate(pooled_irreps, pooled_irreps)
            self.post_pool2 = O3TensorProduct(pooled_irreps, output_irreps)
            self.init_pooler(pool)
        elif task == "node":
            self.pre_pool1 = O3TensorProductSwishGate(
                hidden_irreps, hidden_irreps, node_attr_irreps
            )
            self.pre_pool2 = O3TensorProduct(
                hidden_irreps, output_irreps, node_attr_irreps
            )

    def set_equivariance(self, disable_equivariance):
        for layer in self.layers:
            layer.disable_equivariance = disable_equivariance

    def get_gammas(self):
        return [gamma for layer in self.layers for gamma in layer.get_gammas()]

    def init_pooler(self, pool):
        """Initialise pooling mechanism"""
        if pool == "avg":
            self.pooler = global_mean_pool
        elif pool == "sum":
            self.pooler = global_add_pool

    def catch_isolated_nodes(self, graph):
        """Ensure isolated nodes receive attributes with correct trivial irreps."""
        max_index = graph.edge_index.max().item() if graph.edge_index.numel() > 0 else -1
        if graph.contains_isolated_nodes() and max_index + 1 != graph.num_nodes:
            nr_add_attr = graph.num_nodes - (max_index + 1)
            feature_dim = graph.node_attr.shape[-1]
            add_attr = graph.node_attr.new_zeros((nr_add_attr, feature_dim))
            graph.node_attr = torch.cat((graph.node_attr, add_attr), -2)
        graph.node_attr[:, 0] = 1.0

    def forward(self, graph):
        """SEGNN forward pass"""
        x, pos, edge_index, edge_attr, node_attr, batch = (
            graph.x,
            graph.pos,
            graph.edge_index,
            graph.edge_attr,
            graph.node_attr,
            graph.batch,
        )
        try:
            additional_message_features = graph.additional_message_features
        except AttributeError:
            additional_message_features = None

        self.catch_isolated_nodes(graph)

        x = self.embedding_layer(x, node_attr)

        # Pass messages
        for layer in self.layers:
            x = layer(
                x, edge_index, edge_attr, node_attr, batch, additional_message_features
            )

        # Pre pool
        x = self.pre_pool1(x, node_attr)
        x = self.pre_pool2(x, node_attr)

        if self.task == "graph":
            # Pool over nodes
            x = self.pooler(x, batch)

            # Predict
            x = self.post_pool1(x)
            x = self.post_pool2(x)
        return x


class SEGNNLayer(MessagePassing):
    """E(3) equivariant message passing layer."""

    def __init__(
        self,
        input_irreps,
        hidden_irreps,
        output_irreps,
        edge_attr_irreps,
        node_attr_irreps,
        norm=None,
        additional_message_irreps=None,
        args=None,
    ):
        super().__init__(node_dim=-2, aggr="add")
        self.hidden_irreps = hidden_irreps

        message_input_irreps = (2 * input_irreps + additional_message_irreps).simplify()
        update_input_irreps = (input_irreps + hidden_irreps).simplify()

        self.message_layer_1 = O3TensorProductSwishGate(
            message_input_irreps, hidden_irreps, edge_attr_irreps
        )
        self.message_layer_2 = O3TensorProductSwishGate(
            hidden_irreps, hidden_irreps, edge_attr_irreps
        )
        self.update_layer_1 = O3TensorProductSwishGate(
            update_input_irreps, hidden_irreps, node_attr_irreps
        )
        self.update_layer_2 = O3TensorProduct(
            hidden_irreps, hidden_irreps, node_attr_irreps
        )

        if args.disable_equivariance:
            self.lin_noneq1 = nn.Sequential(
                nn.Linear(hidden_irreps.dim * 2, hidden_irreps.dim * 2),
                nn.GELU(),                
                nn.Linear(hidden_irreps.dim * 2, hidden_irreps.dim),
                nn.GELU(),
                nn.Linear(hidden_irreps.dim, hidden_irreps.dim),
            )

            self.lin_noneq2 = nn.Sequential(
                nn.Linear(hidden_irreps.dim, hidden_irreps.dim * 2),
                nn.GELU(),
                nn.Linear(hidden_irreps.dim * 2, hidden_irreps.dim),
                nn.GELU(),
                nn.Linear(hidden_irreps.dim, hidden_irreps.dim),
            )

            self.gammas = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for i in range(2)])
            self.disable_equivariance = True
        else:
            self.gammas = []
            self.disable_equivariance = False

        self.setup_normalisation(norm)

    def get_gammas(self):
        gammas_list = []
        
        for gamma in self.gammas:
            gammas_list.append(gamma)

        return gammas_list

    def setup_normalisation(self, norm):
        """Set up normalisation, either batch or instance norm"""
        self.norm = norm
        self.feature_norm = None
        self.message_norm = None

        if norm == "batch":
            self.feature_norm = BatchNorm(self.hidden_irreps)
            self.message_norm = BatchNorm(self.hidden_irreps)
        elif norm == "instance":
            self.feature_norm = InstanceNorm(self.hidden_irreps)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        node_attr,
        batch,
        additional_message_features=None,
    ):
        """Propagate messages along edges"""
        x = self.propagate(
            edge_index,
            x=x,
            node_attr=node_attr,
            edge_attr=edge_attr,
            additional_message_features=additional_message_features,
        )
        # Normalise features
        if self.feature_norm:
            if self.norm == "batch":
                x = self.feature_norm(x)
            elif self.norm == "instance":
                x = self.feature_norm(x, batch)
        return x

    def message(self, x_i, x_j, edge_attr, additional_message_features):
        """Create messages"""
        if additional_message_features is None:
            input = torch.cat((x_i, x_j), dim=-1)
        else:
            input = torch.cat((x_i, x_j, additional_message_features), dim=-1)

        message = self.message_layer_1(input, edge_attr)
        message = self.message_layer_2(message, edge_attr)

        if self.message_norm:
            message = self.message_norm(message)
        return message

    def update(self, message, x, node_attr):
        """Update note features"""
        input = torch.cat((x, message), dim=-1)

        if self.disable_equivariance:
            update = self.update_layer_1(input, node_attr) + self.gammas[0] * self.lin_noneq1(input)
            update = self.update_layer_2(update, node_attr) + self.gammas[1] * self.lin_noneq2(update)
        else:
            update = self.update_layer_1(input, node_attr)
            update = self.update_layer_2(update, node_attr)

        x += update  # Residual connection
        return x
