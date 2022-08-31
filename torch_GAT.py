import torch
import torch.nn as nn

from einops import rearrange, reduce

import torch_geometric.nn as geom_nn

from utils import LayerType


class GATModel(nn.Module):

    def __init__(
            self,
            dataset,
            num_heads,
            c_hidden=64,
            num_layers=2,
            dropout=0.1,
            layer_type=LayerType.IMP2
    ):
        super().__init__()

        GATLayer = get_layer_type(layer_type)

        feature_sizes = [
            dataset.num_node_features,
            c_hidden,
            dataset.num_classes
        ]
        self.num_nodes = dataset.data.num_nodes
        adjacency_matrix = self.make_adjacency_matrix(dataset.data.edge_index)

        assert num_layers == len(num_heads) == len(
            feature_sizes) - 1, f'Enter valid arch params.'

        # collect GAT layers
        layers = []

        for i in range(num_layers):
            layers += [
                GATLayer(
                    in_channels=feature_sizes[i],
                    out_channels=feature_sizes[i + 1],
                    heads=num_heads[i],
                    concat=True if i < num_layers - 1 else False,
                    dropout=dropout,
                    bias=True,
                    num_nodes=self.num_nodes,
                    adjacency_matrix=adjacency_matrix
                )
            ]
            if i < num_layers - 1:
                layers += [
                    nn.ELU(inplace=True),
                    nn.Dropout(0.1)
                ]

        self.layers = nn.ModuleList(layers)

    def make_adjacency_matrix(self, edge_index):
        A = torch.eye(self.num_nodes, dtype=torch.int32)
        for src, tgt in edge_index.T:
            src = src.item()
            tgt = tgt.item()
            A[src][tgt] = 1
        return A


    def forward(self, x, edge_index):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for l in self.layers:
            if  isinstance(l, BaseGATLayer):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x

class BaseGATLayer(torch.nn.Module):
    """
    Base class for all implementations as there is much code that would otherwise be copy/pasted.
    """

    nodes_dim = 0  # node dimension/axis
    head_dim = 1

    def __init__(
            self,
            in_channels,
            out_channels,
            heads,
            num_nodes,
            concat=True,
            dropout=0.6,
            bias=True,
            **kwargs
    ):

        super().__init__()

        self.heads = heads
        self.out_channels = out_channels
        self.head_feat_dim = out_channels//heads
        self.num_nodes = num_nodes
        self.concat = concat

        self.embedding = nn.Linear(in_channels, out_channels, bias=False)

        self.a_left = nn.Parameter(torch.Tensor(self.head_feat_dim, heads, 1))
        self.a_right = nn.Parameter(torch.Tensor(self.head_feat_dim, heads, 1))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(self.head_feat_dim))
        else:
            self.register_parameter('bias', None)

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=dropout)

        self.init_params()

    def init_params(self):

        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.a_left)
        nn.init.xavier_uniform_(self.a_right)

        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def concat_bias(self, output_features):

        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            output_features = rearrange(
                output_features, 'n h f-> n (h f)')
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            output_features = reduce(
                output_features, 'n h c -> n c', 'mean')

        if self.bias is not None:
            output_features += self.bias

        return output_features


class EfficientGATLayer(BaseGATLayer):
    """
    Implementation #3 was inspired by PyTorch Geometric: https://github.com/rusty1s/pytorch_geometric
    But, it's hopefully much more readable! (and of similar performance)
    It's suitable for both transductive and inductive settings. In the inductive setting we just merge the graphs
    into a single graph with multiple components and this layer is agnostic to that fact! <3
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            heads,
            num_nodes,
            concat=True,
            dropout=0.1,
            bias=True
    ):
        # Delegate initialization to the base class
        super().__init__(in_channels, out_channels, heads, num_nodes, concat, dropout, bias)

    def forward(self, node_features, edge_index):

        node_features = self.dropout(node_features)

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        embeddings = self.embedding(node_features)

        embeddings = rearrange(
            embeddings,
            'n (h c) -> n h c',
            h=self.heads
        )

        embeddings = self.dropout(embeddings)  # in the official GAT imp they did dropout here as well

        # Apply the scoring function (* represents element-wise (a.k.a. Hadamard) product)
        # shape = (N, NH, FOUT) * (1, NH, FOUT) -> (N, NH, 1) -> (N, NH) because sum squeezes the last dimension
        # Optimization note: torch.sum() is as performant as .sum() in my experiments
        # src_scores = (embeddings * self.a_left).sum(dim=-1)
        # tgt_scores = (embeddings * self.a_rightt).sum(dim=-1)

        src_scores = torch.einsum(
            'nhc, chi -> nh',
            embeddings,
            self.a_left,
        )#.sum(dim=-1)
        tgt_scores = torch.einsum(
            'nhc, chi -> nh',
            embeddings,
            self.a_right,
        )#.sum(dim=-1)

        # We simply copy (lift) the scores for source/target nodes based on the edge index. Instead of preparing all
        # the possible combinations of scores we just prepare those that will actually be used and those are defined
        # by the edge index.
        # scores shape = (E, NH), embeddings_lifted shape = (E, NH, FOUT), E - number of edges in the graph
        src_scores_lifted, tgt_scores_lifted, embeddings_lifted = self.lift(
            src_scores, tgt_scores, embeddings, edge_index)
        
        edge_scores = self.leakyReLU(src_scores_lifted + tgt_scores_lifted)

        # shape = (E, NH, 1)
        edge_attn = self.neighbourhood_softmax(edge_scores, edge_index[1])
        edge_attn = self.dropout(edge_attn)

        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (E, NH, FOUT) * (E, NH, 1) -> (E, NH, FOUT), 1 gets broadcast into FOUT
        embeddings_lifted_weighted = embeddings_lifted * edge_attn

        # This part sums up weighted and projected neighbourhood feature vectors for every target node
        # shape = (N, NH, FOUT)
        output_features = self.aggregate_neighbours(
            embeddings_lifted_weighted,
            edge_index,
        )

        output_features = self.concat_bias(output_features)

        return output_features
    
    def lift(self, scores_source, scores_target, embeddings, edge_index):
        """
        Lifts i.e. duplicates certain vectors depending on the edge index.
        One of the tensor dims goes from N -> E
        """
        src_nodes_index = edge_index[0]
        trg_nodes_index = edge_index[1]

        # Using index_select is faster than "normal" indexing
        scores_source = scores_source.index_select(
            self.nodes_dim,
            src_nodes_index
        )
        scores_target = scores_target.index_select(
            self.nodes_dim,
            trg_nodes_index
        )
        embeddings_lifted = embeddings.index_select(
            self.nodes_dim,
            src_nodes_index
        )

        return scores_source, scores_target, embeddings_lifted

    def neighbourhood_softmax(self, edge_scores, trg_index):

        # Calculate the numerator. Make logits <= 0 so that e^logit <= 1
        #edge_scores = edge_scores - edge_scores.max()
        exp_edge_scores = edge_scores.exp()

        # Calculate the denominator.
        # Shape = (E, H)
        sum_exp_edge_scores = self.sum_edge_scores(exp_edge_scores, trg_index)

        # Softmax over the neighbouring nodes
        attn = exp_edge_scores / (sum_exp_edge_scores + 1e-16)


        # shape = (E, NH) -> (E, NH, 1) so that we can do element-wise
        # multiplication with projected node features
        return attn.unsqueeze(-1)

    def sum_edge_scores(self, exp_scores_per_edge, trg_index):
        # The shape must be the same as in exp_scores_per_edge
        # (required by scatter_add_) i.e. from E -> (E, NH)
        trg_index_broadcasted = self.expand_dims(
            trg_index,
            exp_scores_per_edge
        )

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        neighbourhood_sums = torch.zeros(
            size=[self.num_nodes, self.heads],
            dtype=exp_scores_per_edge.dtype,
            device=exp_scores_per_edge.device
        )

        # position i will contain a sum of exp scores of all the nodes
        # that point to the node i (as dictated by the target index)
        neighbourhood_sums.scatter_add_(
            self.nodes_dim,
            trg_index_broadcasted,
            exp_scores_per_edge
        )

        # Expand again so that we can use it as a softmax denominator.
        # e.g. node i's sum will be copied to all the locations where
        # the source nodes pointed to i (as dictated by the target index)
        # shape = (N, NH) -> (E, NH)
        return neighbourhood_sums.index_select(self.nodes_dim, trg_index)

    def aggregate_neighbours(self, weighted_embeddings, edge_index):

        output_features = torch.zeros(
            size=[self.num_nodes, self.heads, self.head_feat_dim],
            dtype=weighted_embeddings.dtype,
            device=weighted_embeddings.device
        )

        # shape = (E) -> (E, H, C_OUT)
        trg_index_broadcasted = self.expand_dims(
            edge_index[1],
            weighted_embeddings
        )

        # aggregation step - we accumulate projected,
        # weighted node features for all the attention heads
        # shape = (E, H, C_OUT) -> (N, H, C_OUT)
        output_features.scatter_add_(
            self.nodes_dim,
            trg_index_broadcasted,
            weighted_embeddings
        )

        return output_features

    def expand_dims(self, this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)


class BetterGATLayer(BaseGATLayer):

    def __init__(
            self,
            in_channels,
            out_channels,
            num_nodes,
            adjacency_matrix,
            heads=8,
            concat=True,
            dropout=0.3,
            bias=True):
        """
        Inputs:
            c_in - Dimensionality of input features
            c_out - Dimensionality of output features
            num_heads - Number of heads, i.e. attention mechanisms to apply in parallel. The
                        output features are equally split up over the heads if concat_heads=True.
            concat_heads - If True, the output of the different heads is concatenated instead of averaged.
            alpha - Negative slope of the LeakyReLU activation.
        """
        

        super().__init__(in_channels, out_channels, heads, num_nodes, concat, dropout, bias)

        self.adjacency_matrix = adjacency_matrix
        self.softmax = nn.Softmax(-1)

    def forward(self, node_features, edge_index):
        #
        # Step 1: Linear Projection + regularization (using linear layer instead of matmul as in imp1)
        #

        assert self.adjacency_matrix.shape == (self.num_nodes, self.num_nodes), \
            f'Expected connectivity matrix with shape=({self.num_nodes},{self.num_nodes}), got shape={self.adjacency_matrix.shape}.'

        # shape = (N, FIN) * (FIN, NH*FOUT) -> (N, NH, FOUT) where NH - number of heads, FOUT - num of output features
        # We project the input node features into NH independent output features (one for each attention head)
        embeddings = self.embedding(node_features)

        embeddings = rearrange(
            embeddings,
            'n (h c) -> n h c',
            h=self.heads
        )

        embeddings = self.dropout(embeddings)  # in the official GAT imp they did dropout here as well

        src_scores = torch.einsum(
            'nhc, chi -> hni',
            embeddings,
            self.a_left,
        )
        tgt_scores = torch.einsum(
            'nhc, chi -> hin',
            embeddings,
            self.a_right,
        )

        all_scores = self.leakyReLU(src_scores + tgt_scores)
        self.adjacency_matrix = self.adjacency_matrix.type_as(all_scores).bool()

        masked_scores = all_scores.masked_fill_(~self.adjacency_matrix, float('-inf'))
        attn = self.softmax(masked_scores)


        output_features = torch.einsum(
            'hnn, nhc->nhc',
            attn,
            embeddings
        )

        output_features = self.concat_bias(output_features)

        return output_features


class NaiveGATLayer(BaseGATLayer):

    def __init__(
            self,
            in_channels,
            out_channels,
            heads,
            concat,
            activation,
            dropout,
            bias,
            adjacency_dict
    ):

        super().__init__(in_channels, out_channels, heads, concat,
                         activation, dropout,
                         bias)

        self.a_left = torch.squeeze(self.a_left, -1)
        self.a_right = torch.squeeze(self.a_right, -1)
        self.adjacency_dict=adjacency_dict

    def forward(self, node_feats, edge_index):

        node_feats = self.dropout(node_feats)

        # Shape = (N, C_IN) * (C_IN, C_OUT) = (N, C_OUT) -> (N, H, H_dim)
        embeddings = self.embedding(node_feats)
        embeddings = rearrange(
            embeddings,
            'n (h c) -> n h c',
            h=self.heads
        )

        embeddings = self.dropout(embeddings)

        scores_dict = {}
        for node, neighbours in self.adjacency_dict.items():
            scores_dict[node] = {}
            for neighbour in neighbours:
                score = torch.einsum(
                    'hc, ch -> h',
                    torch.cat((embeddings[node], embeddings[neighbour]), 1),
                    torch.concat((self.a_left, self.a_right), 0),
                )
                score = self.leakyReLU(score)
                scores_dict[node][neighbour] = torch.exp(score)

        totals_dict = {}
        for node, neighbours in scores_dict.items():
            total = 0
            for neighbour in neighbours:
                neighbour_node = scores_dict[neighbour]
                incoming_score = neighbour_node[node]
                total += incoming_score
            totals_dict[node] = total

        attn_dict = {}
        for node, neighbours in scores_dict.items():
            attn_dict[node] = {}
            total = totals_dict[node]
            for neighbour in neighbours:
                scores = scores_dict[neighbour][node]
                normalised_score = torch.div(scores, total)
                attn_dict[node][neighbour] = normalised_score

        output_features = torch.zeros(
            [self.num_nodes, self.heads, self.head_feat_dim],
            dtype=embeddings.dtype,
            device=embeddings.device)

        for node, neighbours in attn_dict.items():
            feature = 0
            for neighbour, attn in neighbours.items():
                feature += torch.einsum(
                    'hc, h -> hc',
                    embeddings[neighbour],
                    attn)
            output_features[node] += feature

        output_features = self.concat_bias(
            output_features
        )

        return output_features


def get_layer_type(layer_type):
    assert isinstance(layer_type, LayerType), \
        f'Expected {LayerType} got {type(layer_type)}.'

    if layer_type == LayerType.IMP1:
        return NaiveGATLayer
    elif layer_type == LayerType.IMP2:
        return BetterGATLayer
    elif layer_type == LayerType.IMP3:
        return EfficientGATLayer
    elif layer_type == LayerType.IMP4:
        return geom_nn.GATConv
    else:
        raise Exception(f'Layer type {layer_type} not yet supported.')