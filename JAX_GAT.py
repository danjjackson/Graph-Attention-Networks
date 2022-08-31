from flax import linen as nn
import jax.numpy as jnp
from jax.numpy import DeviceArray
import numpy as np
from einops import rearrange, reduce

class GATModel(nn.Module):
    num_nodes: int
    num_heads: int
    input_size: int
    c_hidden: int
    num_classes: int
    edge_index: DeviceArray
    dropout: float
    bias: bool

    def setup(self):
        super().__init__()

        adj_mask = self.make_adj_mask(self.edge_index)

        # collect GAT layers
        self.GAT1 = GATLayer(
            adjacency_mask=adj_mask,
            c_in=self.input_size,
            c_out=self.c_hidden,
            num_heads=self.num_heads,
            concat=True,
            activation=nn.elu
        )
        self.GAT2 = GATLayer(
            adjacency_mask=adj_mask,
            c_in=self.c_hidden,
            c_out=self.num_classes,
            num_heads=1,
            concat=False,
            activation=None
        )

    def __call__(self, x):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        x = self.GAT1(x)
        x = self.GAT2(x)
        return x

    def make_adj_mask(self, edge_index):
        A = jnp.full(
            shape=(self.num_nodes, self.num_nodes),
            fill_value=-9e15,
            dtype=jnp.int32)
        A = A + jnp.identity(self.num_nodes, dtype=jnp.int32) * 9e15
        for i, j in edge_index.T:
            A = A.at[i.item(), j.item()].set(0)
        return A

class GATLayer(nn.Module):
    adjacency_mask: DeviceArray  # Adjacency Matrix
    c_in : int  # Dimensionality of input features
    c_out : int  # Dimensionality of output features
    num_heads : int  # Number of heads, i.e. attention mechanisms to apply in parallel.
    activation : callable # Activation function
    concat : bool # If True, the output of the different heads is concatenated instead of averaged.
    alpha : float = 0.2  # Negative slope of the LeakyReLU activation.
    dropout: float = 0.2
    bias: bool = True

    def setup(self):
        if self.concat:
            assert self.c_out % self.num_heads == 0, \
                "Number of output features " \
                "must be a multiple of the count of heads."
            c_out_per_head = self.c_out // self.num_heads
        else:
            c_out_per_head = self.c_out

        # Sub-modules and parameters needed in the layer
        self.embedding = nn.Dense(
            c_out_per_head * self.num_heads,
            kernel_init=nn.initializers.glorot_uniform()
        )
        self.a_left = self.param(
            'a_left',
            nn.initializers.glorot_uniform(),
            (1, c_out_per_head, self.num_heads)
        )
        self.a_right = self.param(
            'a_right',
            nn.initializers.glorot_uniform(),
            (c_out_per_head, 1,  self.num_heads)
        )

    def __call__(self, node_feats):
        """
        Inputs:
            node_feats - Input features of the node. Shape: [batch_size, c_in]
        """

        # Apply linear layer and sort nodes by head
        node_feats = self.embedding(node_feats)
        node_feats = rearrange(
            node_feats,
            'n (h c) -> n h c',
            h=self.num_heads
        )

        src_scores = jnp.einsum(
            'nhc,Ich->Inh',
            node_feats,
            self.a_left
        )
        tgt_scores = jnp.einsum(
            'nhc,cIh->nIh',
            node_feats,
            self.a_right
        )

        all_scores = nn.leaky_relu(src_scores + tgt_scores, self.alpha)
        all_scores = rearrange(all_scores, 'n1 n2 h -> h n1 n2')

        # Mask out nodes that do not have an edge between them
        masked_scores = all_scores + self.adjacency_mask

        # Weighted average of attention
        attn = nn.softmax(masked_scores, axis=2)

        node_feats = jnp.einsum('hij,jhc->ihc', attn, node_feats)

        # If heads should be concatenated, we can do this by reshaping.
        # Otherwise, take mean
        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            node_feats = rearrange(
                node_feats, 'n h c-> n (h c)')
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            node_feats = reduce(
                node_feats, 'n h c -> n c', 'mean')

        return node_feats if self.activation is None \
            else self.activation(node_feats)