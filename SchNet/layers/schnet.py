# Implementation adapted from https://pytorch-geometric.readthedocs.io/en/stable/_modules/torch_geometric/nn/models/schnet.html


from math import pi as PI
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Embedding, Linear, ModuleList, Sequential
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing, SumAggregation, radius_graph
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from torch_geometric.typing import OptTensor

    

qm9_target_dict: Dict[int, str] = {
    0: "dipole_moment",
    1: "isotropic_polarizability",
    2: "homo",
    3: "lumo",
    4: "gap",
    5: "electronic_spatial_extent",
    6: "zpve",
    7: "energy_U0",
    8: "energy_U",
    9: "enthalpy_H",
    10: "free_energy",
    11: "heat_capacity",
}

class SchNet(torch.nn.Module):
    r"""The continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper that uses
    the interactions blocks of the form.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_filters (int, optional): The number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): The number of interaction blocks.
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        interaction_graph (callable, optional): The function used to compute
            the pairwise interaction graph and interatomic distances. If set to
            :obj:`None`, will construct a graph based on :obj:`cutoff` and
            :obj:`max_num_neighbors` properties.
            If provided, this method takes in :obj:`pos` and :obj:`batch`
            tensors and should return :obj:`(edge_index, edge_weight)` tensors.
            (default :obj:`None`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        readout (str, optional): Whether to apply :obj:`"add"` or :obj:`"mean"`
            global aggregation. (default: :obj:`"add"`)
        dipole (bool, optional): If set to :obj:`True`, will use the magnitude
            of the dipole moment to make the final prediction, *e.g.*, for
            target 0 of :class:`torch_geometric.datasets.QM9`.
            (default: :obj:`False`)
        mean (float, optional): The mean of the property to predict.
            (default: :obj:`None`)
        std (float, optional): The standard deviation of the property to
            predict. (default: :obj:`None`)
        atomref (torch.Tensor, optional): The reference of single-atom
            properties.
            Expects a vector of shape :obj:`(max_atomic_number, )`.
    """

    url = "http://www.quantum-machine.org/datasets/trained_schnet_models.zip"

    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        interaction_graph: Optional[Callable] = None,
        max_num_neighbors: int = 32,
        readout: str = "add",
        dipole: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        atomref: OptTensor = None,
        args=None,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.dipole = dipole
        self.sum_aggr = SumAggregation()
        self.readout = aggr_resolver("sum" if self.dipole else readout)
        self.mean = mean
        self.std = std
        self.scale = None
        self.args = args

        self.gammas = None

        if args.disable_equi:
            self.pos_linear = Linear(3, hidden_channels)
            self.pos_interactions = nn.ModuleList(
                nn.Sequential(nn.BatchNorm1d(hidden_channels), nn.Linear(hidden_channels, hidden_channels))
                for _ in range(num_interactions)
            )

            if args.n_gammas == 'single':
                self.gammas = nn.Parameter(torch.tensor(1.0))
            elif args.n_gammas == 'nn':
                self.gammas = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.LazyLinear(hidden_channels),
                            nn.GELU(),
                            nn.Linear(hidden_channels, hidden_channels//2), 
                            nn.GELU(), 
                            nn.Linear(hidden_channels//2, hidden_channels//3), 
                            nn.GELU(), 
                            nn.Linear(hidden_channels//3, 1)
                        ) for _ in range(num_interactions)
                    ]
                )
            else:
                self.gammas = nn.ParameterList([nn.Parameter(torch.tensor(1.)) for _ in range(num_interactions)])

        if self.dipole:
            import ase

            atomic_mass = torch.from_numpy(ase.data.atomic_masses)
            self.register_buffer("atomic_mass", atomic_mass)

        # Support z == 0 for padding atoms so that their embedding vectors
        # are zeroed and do not receive any gradients.
        self.embedding = Embedding(100, hidden_channels, padding_idx=0)

        if interaction_graph is not None:
            self.interaction_graph = interaction_graph
        else:
            self.interaction_graph = RadiusInteractionGraph(cutoff, max_num_neighbors)

        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(
                hidden_channels, num_gaussians, num_filters, cutoff
            )
            self.interactions.append(block)

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, 1)

        self.register_buffer("initial_atomref", atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)

    def get_gammas(self):
        if isinstance(self.gammas, nn.ParameterList):
            return self.gammas
        elif isinstance(self.gammas, nn.ModuleList):
            return self.gammas
        elif self.gammas is not None:
            return [self.gammas]
        else:
            return []

    def forward(self, z: Tensor, pos: Tensor, batch: OptTensor = None, disable_equivariance=False) -> Tensor:
        r"""Forward pass.

        Args:
            z (torch.Tensor): Atomic number of each atom with shape
                :obj:`[num_atoms]`.
            pos (torch.Tensor): Coordinates of each atom with shape
                :obj:`[num_atoms, 3]`.
            batch (torch.Tensor, optional): Batch indices assigning each atom
                to a separate molecule with shape :obj:`[num_atoms]`.
                (default: :obj:`None`)
        """


        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        if disable_equivariance:
            h_pos = self.pos_linear(pos)

        gamma_values = []

        for idx, interaction in enumerate(self.interactions):
            if disable_equivariance:
                if isinstance(self.gammas, nn.ParameterList):
                    gamma = self.gammas[idx]
                else:
                    gamma = self.gammas

                h = h + interaction(h, edge_index, edge_weight, edge_attr)

                h_pos = self.pos_interactions[idx](h_pos)

                if self.args.n_gammas == 'nn':
                    h_gamma = torch.concat((h.detach(), h_pos.detach()), dim=1)
                    gamma = self.gammas[idx](h_gamma)

                h = h + gamma * h_pos
                gamma_values.append(gamma.norm())
            else:
                h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.dipole:
            # Get center of mass.
            mass = self.atomic_mass[z].view(-1, 1)
            M = self.sum_aggr(mass, batch, dim=0)
            c = self.sum_aggr(mass * pos, batch, dim=0) / M
            h = h * (pos - c.index_select(0, batch))

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        out = self.readout(h, batch, dim=0)

        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if self.scale is not None:
            out = self.scale * out

        if self.args.n_gammas == 'nn':
            return out, gamma_values
        else:
            return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"hidden_channels={self.hidden_channels}, "
            f"num_filters={self.num_filters}, "
            f"num_interactions={self.num_interactions}, "
            f"num_gaussians={self.num_gaussians}, "
            f"cutoff={self.cutoff})"
        )


class RadiusInteractionGraph(torch.nn.Module):
    r"""Creates edges based on atom positions :obj:`pos` to all points within
    the cutoff distance.

    Args:
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance with the
            default interaction graph method.
            (default: :obj:`32`)
    """

    def __init__(self, cutoff: float = 10.0, max_num_neighbors: int = 32):
        super().__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def forward(self, pos: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Forward pass.

        Args:
            pos (Tensor): Coordinates of each atom.
            batch (LongTensor, optional): Batch indices assigning each atom to
                a separate molecule.

        :rtype: (:class:`LongTensor`, :class:`Tensor`)
        """
        edge_index = radius_graph(
            pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors
        )
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        return edge_index, edge_weight


class InteractionBlock(torch.nn.Module):
    def __init__(
        self, hidden_channels: int, num_gaussians: int, num_filters: int, cutoff: float
    ):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(
            hidden_channels, hidden_channels, num_filters, self.mlp, cutoff
        )
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, edge_attr: Tensor
    ) -> Tensor:
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        nn: Sequential,
        cutoff: float,
    ):
        super().__init__(aggr="add")
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, edge_attr: Tensor
    ) -> Tensor:
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j: Tensor, W: Tensor) -> Tensor:
        return x_j * W


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
    ):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist: Tensor) -> Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x: Tensor) -> Tensor:
        return F.softplus(x) - self.shift
