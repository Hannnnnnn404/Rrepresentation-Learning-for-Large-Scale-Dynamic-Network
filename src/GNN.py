import networkx as nx
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager
import numpy as np
import torch
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data, Batch


class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        # 增加Dropout以减少过拟合
        self.dropout = torch.nn.Dropout(p=0.5)
        # 为全局特征预留的线性层
        self.global_feature_lin = torch.nn.Linear(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)  # To get the final 128-d vector

    def forward(self, x, edge_index, batch):
        # Apply GCN layers
        x = torch.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x, edge_index))
        # Global pooling
        x = global_mean_pool(x, batch)  # Aggregates node features across the whole graph
        # Pass through a final fully connected layer to get to the desired output size
        x = self.fc(x)

        return x


def generate_adjacency_matrix(subgraph_node_index, subgraph_nodes, subgraph_edges):
    """
    Generate adjacency matrices for all subgraphs at initial time parallel.

    Parameters:
    subgraph_node_index (dict): A dictionary mapping (subgraph_id, node) to the node's index in the subgraph.
    subgraph_nodes (dict): A dictionary mapping subgraph IDs to sets of nodes.
    subgraph_edges (dict): A dictionary mapping subgraph IDs to sets of edges.

    Returns:
    dict: A dictionary mapping subgraph IDs to their adjacency matrices.
    """
    adj_matrix_dict = {}
    with ProcessPoolExecutor(max_workers=len(subgraph_nodes.keys())) as executor:
        futures = {
            executor.submit(generate_subgraph_adj_matrix, subgraph_node_index, subgraph_nodes_set,
                            subgraph_edges[subgraph_id], subgraph_id): subgraph_id
            for subgraph_id, subgraph_nodes_set in subgraph_nodes.items()
        }

        for future in futures:
            adj_matrix, subgraph_id = future.result()
            adj_matrix_dict[subgraph_id] = adj_matrix
    return adj_matrix_dict


def generate_subgraph_adj_matrix(subgraph_node_index, subgraph_nodes_set, subgraph_edges_set, subgraph_id):
    """
    Generate the adjacency matrix for a single subgraph.

    Parameters:
    subgraph_node_index (dict): A dictionary mapping (subgraph_id, node) to the node's index in the subgraph.
    subgraph_nodes_set (set): A set of nodes in the subgraph.
    subgraph_edges_set (set): A set of edges in the subgraph.
    subgraph_id (int): The ID of the subgraph.

    Returns:
    tuple: A tuple containing the adjacency matrix and the subgraph ID.
    """
    print("generate subgraph adj matrix for", subgraph_id)
    nodes_index = {subgraph_node_index[(subgraph_id, node)] for node in subgraph_nodes_set}
    edges_index = {(subgraph_node_index[(subgraph_id, edge[0])], subgraph_node_index[(subgraph_id, edge[1])]) for edge
                   in subgraph_edges_set}
    # 创建子图
    G = nx.Graph()
    G.add_nodes_from(nodes_index)
    G.add_edges_from(edges_index)

    # 生成邻接矩阵
    adj_matrix = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    print(adj_matrix)
    return adj_matrix, subgraph_id


def update_node_index_map(subgraph_nodes, subgraph_node_index):
    """
    Update the node index map for all subgraphs in parallel.

    Parameters:
    subgraph_nodes (dict): A dictionary mapping subgraph IDs to sets of nodes.
    subgraph_node_index (dict): A dictionary mapping (subgraph_id, node) to the node's index in the subgraph.

    Returns:
    None
    """
    with ProcessPoolExecutor(max_workers=len(subgraph_nodes.keys())) as executor:
        for subgraph_id, subgraph_nodes_set in subgraph_nodes.items():
            executor.submit(update_subgraph_node_index_map, subgraph_id, subgraph_nodes_set, subgraph_node_index)


def update_subgraph_node_index_map(subgraph_id, subgraph_nodes_set, subgraph_node_index):
    """
    Update the node index map for a single subgraph by adding new nodes.

    Parameters:
    subgraph_id (int): The ID of the subgraph.
    subgraph_nodes_set (set): A set of nodes in the subgraph.
    subgraph_node_index (dict): A dictionary mapping (subgraph_id, node) to the node's index in the subgraph.

    Returns:
    None
    """
    max_index = sum(subgraph_id for key in subgraph_node_index.keys() if key[0] == subgraph_id)  # 找到每个subgraph的最大的index
    current_index = max_index

    for node in subgraph_nodes_set:
        if (subgraph_id, node) not in subgraph_node_index.keys():
            subgraph_node_index[(subgraph_id, node)] = current_index
            current_index += 1


def update_adjacency_matrix(subgraph_adj_matrices, subgraph_node_index, update_nodes, update_edges, expired_edges):
    """
    Update the adjacency matrix for a subgraph by adding new nodes and edges and removing expired edges.

    Parameters:
    subgraph_adj_matrices (dict): A dictionary mapping subgraph IDs to their adjacency matrices.
    subgraph_id (int): The ID of the subgraph to update.
    update_nodes (dict): A dictionary mapping subgraph IDs to sets of updated nodes to be added.
    update_edges (dict): A dictionary mapping subgraph IDs to sets of updated edges to be added.
    subgraph_node_index (dict): A dictionary mapping (subgraph_id, node) to the node's index in the subgraph.
    expired_edges (dict): A dictionary mapping subgraph IDs to sets of expired edges to be removed.

    Returns:
    dict: The updated dictionary of adjacency matrices.
    """
    update_subgraph_ids = set(update_nodes.keys()).union(set(update_edges.keys()))
    with ProcessPoolExecutor(max_workers=len(update_subgraph_ids)) as executor:
        for subgraph_id in update_subgraph_ids:
            executor.submit(update_subgraph_adj_matrix, subgraph_adj_matrices, subgraph_id, update_nodes, update_edges,
                            subgraph_node_index, expired_edges)
    return subgraph_adj_matrices, update_subgraph_ids


def update_subgraph_adj_matrix(subgraph_adj_matrices, subgraph_id, update_nodes, update_edges, subgraph_node_index,
                               expired_edges):
    """
    Update the adjacency matrices for subgraphs by adding new nodes and edges, and removing expired edges.

    Parameters:
    subgraph_adj_matrices (dict): A dictionary mapping subgraph IDs to their adjacency matrices.
    subgraph_node_index (dict): A dictionary mapping (subgraph_id, node) to the node's index in the subgraph.
    update_nodes (dict): A dictionary mapping subgraph IDs to sets of updated nodes to be added.
    update_edges (dict): A dictionary mapping subgraph IDs to sets of updated edges to be added.
    expired_edges (dict): A dictionary mapping subgraph IDs to sets of expired edges to be removed.

    Returns:
    tuple: The updated dictionary of adjacency matrices and a set of updated subgraph IDs.
    """
    old_adj_matrix = subgraph_adj_matrices[subgraph_id]
    old_size = old_adj_matrix.shape[0]
    new_size = old_size
    if subgraph_id in update_nodes.keys():
        new_nodes = update_nodes[subgraph_id]
        new_size = old_size + len(new_nodes)

    new_adj_matrix = np.zeros((new_size, new_size))
    new_adj_matrix[:old_size, :old_size] = old_adj_matrix

    current_index = old_size
    if subgraph_id in update_nodes.keys():  # Check if new_nodes is not None and not empty
        for node in update_nodes[subgraph_id]:
            subgraph_node_index[(subgraph_id, node)] = current_index
            current_index += 1

    if subgraph_id in update_edges.keys():
        for edge in update_edges[subgraph_id]:
            idx1 = subgraph_node_index[(subgraph_id, edge[0])]
            idx2 = subgraph_node_index[(subgraph_id, edge[1])]
            new_adj_matrix[idx1, idx2] = 1
            new_adj_matrix[idx2, idx1] = 1

    if subgraph_id in expired_edges.keys():
        for edge in expired_edges[subgraph_id]:
            idx1 = subgraph_node_index[(subgraph_id, edge[0])]
            idx2 = subgraph_node_index[(subgraph_id, edge[1])]
            new_adj_matrix[idx1, idx2] = 0
            new_adj_matrix[idx2, idx1] = 0

    subgraph_adj_matrices[subgraph_id] = new_adj_matrix
    print(f"Updated adjacency matrix for subgraph {subgraph_id}:\n{new_adj_matrix}")
    return subgraph_adj_matrices


def compute_subgraph_embedding(adj_matrix):
    """
    Compute the subgraph embedding

    Parameters:
    adj_matrix (numpy.ndarray): adjacency matrix of subgraph

    Returns:
    numpy.ndarray: subgraph embedding
    """
    torch.cuda.empty_cache()  # 清空缓存以释放GPU内存
    print("Preparing subgraph embedding")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_nodes = adj_matrix.shape[0]
    # 创建节点特征，假设使用单位矩阵作为初始特征
    x = torch.ones((num_nodes, 1), dtype=torch.float).to(device)
    edge_index = pyg_utils.dense_to_sparse(torch.tensor(adj_matrix, dtype=torch.float).to(device))[0]
    batch = torch.zeros(num_nodes, dtype=torch.long).to(device)  # 因为每个子图独立，所以batch都是0

    data = Data(x=x, edge_index=edge_index, batch=batch).to(device)
    model = GNN(in_channels=1, hidden_channels=32, out_channels=64).to(device)
    model.eval()
    with torch.no_grad():
        embedding = model(data.x, data.edge_index, data.batch)

    return embedding.cpu().detach().numpy()


def initial_embedding_computation(subgraph_adj_matrices):
    """
    Compute all subgraph embeddings for initial Topology

    Parameters:
    subgraph_adj_matrices (dict): A dictionary mapping subgraph IDs to their adjacency matrices.

    Returns:
    dict: 映射子图ID到其嵌入表示的字典
    """
    subgraph_embeddings = {}
    args_list = [adj_matrix for adj_matrix in subgraph_adj_matrices.values()]

    with ThreadPoolExecutor(max_workers=len(subgraph_adj_matrices.keys())) as executor:
        results = executor.map(compute_subgraph_embedding, args_list)

    for subgraph_id, embedding in zip(subgraph_adj_matrices.keys(), results):
        subgraph_embeddings[subgraph_id] = embedding

    return subgraph_embeddings


def update_embedding_computation(subgraph_adj_matrices, update_subgraph_ids, subgraph_embeddings):
    """
    Recompute the updated subgraph embeddings

    Parameters:
    subgraph_adj_matrices (dict): A dictionary mapping subgraph IDs to their adjacency matrices.
    update_subgraph_ids (set): A set of subgraph IDs that need to be updated.
    subgraph_embeddings (dict): A dictionary mapping subgraph IDs to their embeddings.

    Returns:
    dict: updated subgraph embedding
    """
    args_list = [subgraph_adj_matrices[subgraph_id] for subgraph_id in update_subgraph_ids]

    with ThreadPoolExecutor(max_workers=len(update_subgraph_ids)) as executor:
        results = executor.map(compute_subgraph_embedding, args_list)

    for subgraph_id, embedding in zip(update_subgraph_ids, results):
        subgraph_embeddings[subgraph_id] = embedding

    return subgraph_embeddings

