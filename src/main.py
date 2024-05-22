import os
import glob
import itertools
from multiprocessing import Manager
import time
from graph_construction import generate_initial_topology
from graph_update import generate_update_topology
from GNN import update_node_index_map, generate_adjacency_matrix, update_adjacency_matrix, initial_embedding_computation, update_embedding_computation

if __name__ == "__main__":
    manager = Manager()
    global_nodes = manager.dict()  # 全局节点集与其所在子图的映射
    subgraph_connections = manager.dict()  # 跨子图连接
    subgraph_edges = manager.dict()  # 所有子图的边集与其所在子图的映射{subgraph_id: edges}
    subgraph_nodes = manager.dict()  # 所有子图的节点集与其所在子图的映射{subgraph_id: nodes}
    subgraph_id_gen = itertools.count()  # 子图ID生成器
    paths_activity = manager.dict()  # 全局路径活动记录
    edge_usage = manager.dict()  # 全局边的使用情况
    link_usage = manager.dict()  # 全局子图连接的使用情况
    subgraph_node_index = manager.dict()  # 子图中节点的映射 {(node, subgraph_id): index_in_subgraph}
    subgraph_embeddings = {}  # embedding的记录
    start_time = time.time()

    root_dir = "/home/hanhan210302/BGP_Anomaly_Data_txt"
    date_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    start_date = sorted(date_dirs)[0]
    date_path = os.path.join(root_dir, start_date)
    files = glob.glob(os.path.join(date_path + "/rib/5/bview." + start_date + ".0000.txt"), recursive=True)
    timestamp = start_date + '0000'

    # construction initial Topology
    generate_initial_topology(files, global_nodes, subgraph_connections, subgraph_id_gen, subgraph_nodes, subgraph_edges, paths_activity, edge_usage, link_usage, timestamp)

    # map patch node to index
    update_node_index_map(subgraph_nodes, subgraph_node_index)

    # generate np adj matrix for each patch
    subgraph_adj_matrices = generate_adjacency_matrix(subgraph_node_index, subgraph_nodes, subgraph_edges)

    # compute GNN embedding for each patch
    subgraph_embeddings[timestamp] = initial_embedding_computation(subgraph_adj_matrices)
    print("Embedding: ", subgraph_embeddings[timestamp])
    end_time = time.time()
    print(f"Graph Constrction: {end_time - start_time} s")

    last_timestamp = timestamp
    for date in sorted(date_dirs):
        date_path = os.path.join(root_dir, date)
        files = glob.glob(os.path.join(date_path + "/updates/0/updates." + date + ".*.txt"), recursive=True)
        for file_path in sorted(files):
            date_time = file_path.split('.')
            time1 = date_time[2]
            timestamp = date + time1
            start_time = time.time()
            update_nodes, update_edges = manager.dict(), manager.dict()  # 字典记录每个subgraph更新的nodes/edges集合
            expired_edges = manager.dict()
            files = glob.glob(os.path.join(date_path + "/updates/5/updates." + date + "." + time1 + ".txt"),
                              recursive=True)
            # update topology
            update_nodes, update_edges, expired_edges = generate_update_topology(files, global_nodes,
                                                                                 subgraph_connections, subgraph_nodes,
                                                                                 subgraph_edges, update_nodes,
                                                                                 update_edges, expired_edges,
                                                                                 paths_activity, edge_usage, link_usage,
                                                                                 timestamp)
            # update np adj matrix
            subgraph_adj_matrices, update_subgraph_ids = update_adjacency_matrix(subgraph_adj_matrices,
                                                                                 subgraph_node_index, update_nodes,
                                                                                 update_edges, expired_edges)
            # update embeddings
            subgraph_embeddings[timestamp] = update_embedding_computation(subgraph_adj_matrices, update_subgraph_ids,
                                                                          subgraph_embeddings[last_timestamp])
            print("Embedding: ", len(subgraph_embeddings[timestamp]))
            last_timestamp = timestamp
            end_time = time.time()
            print(f"Graph update: {end_time - start_time} s")