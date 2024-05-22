from concurrent.futures import ProcessPoolExecutor
from data_loader import parse_as_path
from graph_operations import add_node, add_edge, add_connections, update_edge_usage, update_connection_usage
from sliding_window import manage_expired_paths

def find_nearest_subgraph_id(previous_node, global_nodes):
    """
    Find the nearest subgraph ID for a given node.

    Parameters:
    previous_node (str): The previous node in the AS path.
    global_nodes (dict): A global dictionary mapping nodes to subgraph IDs.

    Returns:
    int: The subgraph ID of the nearest subgraph. If the previous node is None, returns 0 (default subgraph ID).
    """
    return global_nodes[previous_node] if previous_node is not None else 0  # 假设0为默认子图ID


def update_subgraph(chunk, global_nodes, subgraph_connections, subgraph_nodes, subgraph_edges, update_nodes,
                    update_edges, paths_activity,
                    edge_usage, link_usage, timestamp):
    """
    Process a chunk of update data to update subgraphs and global information.

    Parameters:
    chunk (list): A list of lines representing a chunk of update data.
    global_nodes (dict): A global dictionary mapping nodes to subgraph IDs.
    subgraph_connections (dict): A dictionary mapping subgraph IDs to connected subgraph IDs.
    subgraph_nodes (dict): A dictionary mapping subgraph IDs to sets of nodes.
    subgraph_edges (dict): A dictionary mapping subgraph IDs to sets of edges.
    update_nodes (dict): A dictionary to track updated nodes for each subgraph.
    update_edges (dict): A dictionary to track updated edges for each subgraph.
    paths_activity (dict): A dictionary to track active paths and their timestamps.
    edge_usage (dict): A dictionary to track the usage counts of edges.
    link_usage (dict): A dictionary to track the usage counts of links between subgraphs.
    timestamp (str): The timestamp associated with the update data.

    Returns:
    tuple: Updated `update_nodes` and `update_edges` dictionaries.
    """
    print("update subgraph")
    for line in chunk:
        as_path = parse_as_path(line)
        path_id = hash(tuple(as_path))
        if path_id in paths_activity:
            paths_activity[path_id]['timestamp'] = timestamp
            continue
        else:
            paths_activity[path_id] = {'timestamp': timestamp, 'path_edges': set(), 'path_links': set()}
            paths_activity[path_id]['timestamp'] = timestamp

        previous_node = None
        path_edges = set()
        path_links = set()
        for node in as_path:
            if node not in global_nodes:
                # 节点未出现在任何子图中，寻找最佳子图添加
                nearest_subgraph_id = find_nearest_subgraph_id(previous_node, global_nodes)  # 寻找最近的子图
                global_nodes[node] = nearest_subgraph_id
                add_node(subgraph_nodes, nearest_subgraph_id, node)
                add_node(update_nodes, nearest_subgraph_id, node)

            if previous_node:
                # 添加边，检查跨子图情况
                previous_graph_id = global_nodes[previous_node]
                current_graph_id = global_nodes[node]
                if previous_graph_id != current_graph_id:  # 添加子图之间的连接
                    link = (previous_graph_id, current_graph_id)
                    path_links.add(link)
                    add_connections(subgraph_connections, previous_graph_id, current_graph_id)
                    update_connection_usage(link_usage, link, increase=True)
                else:
                    edge = (previous_node, node)
                    add_edge(subgraph_edges, previous_graph_id, edge)
                    path_edges.add((previous_node, node, current_graph_id))
                    update_edge_usage(edge_usage, edge, increase=True)
                    add_edge(update_edges, previous_graph_id, edge)
            previous_node = node

        paths_activity[path_id]['path_edges'] = path_edges
        paths_activity[path_id]['path_links'] = path_links

    return update_nodes, update_edges


def handle_updates_file(filename, global_nodes, subgraph_connections, subgraph_nodes, subgraph_edges, update_nodes,
                        update_edges, paths_activity, edge_usage, link_usage, timestamp):
    """
    Handle and process an updates file to update the topology and subgraphs parallel.

    Parameters:
    filename (str): The name of the updates file to process.
    global_nodes (dict): A global dictionary mapping nodes to subgraph IDs.
    subgraph_connections (dict): A dictionary mapping subgraph IDs to connected subgraph IDs.
    subgraph_nodes (dict): A dictionary mapping subgraph IDs to sets of nodes.
    subgraph_edges (dict): A dictionary mapping subgraph IDs to sets of edges.
    update_nodes (dict): A dictionary to track updated nodes for each subgraph.
    update_edges (dict): A dictionary to track updated edges for each subgraph.
    paths_activity (dict): A dictionary to track active paths and their timestamps.
    edge_usage (dict): A dictionary to track the usage counts of edges.
    link_usage (dict): A dictionary to track the usage counts of links between subgraphs.
    timestamp (str): The timestamp associated with the updates file.

    Returns:
    None
    """
    print(f"Processing updates for {filename}...")
    with open(filename, 'r') as file:
        lines = file.readlines()

    chunk_size = len(lines) // 10
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    with ProcessPoolExecutor() as executor:
        for idx, chunk in enumerate(chunks):
            executor.submit(update_subgraph, chunk, global_nodes, subgraph_connections, subgraph_nodes, subgraph_edges,
                            update_nodes, update_edges, paths_activity, edge_usage, link_usage, timestamp)
    return


def generate_update_topology(files, global_nodes, subgraph_connections, subgraph_nodes, subgraph_edges,
                             update_nodes, update_edges, expired_edges,
                             paths_activity, edge_usage, link_usage, timestamp):
    """
    Generate the updated topology by processing update files in parallel.

    Parameters:
    files (list): A list of file paths to update files.
    date (str): The date associated with the update files.
    time (str): The time associated with the update files.
    global_nodes (dict): A global dictionary mapping nodes to subgraph IDs.
    subgraph_connections (dict): A dictionary mapping subgraph IDs to connected subgraph IDs.
    subgraph_nodes (dict): A dictionary mapping subgraph IDs to sets of nodes.
    subgraph_edges (dict): A dictionary mapping subgraph IDs to sets of edges.
    update_nodes (dict): A dictionary to track updated nodes for each subgraph.
    update_edges (dict): A dictionary to track updated edges for each subgraph.
    expired_edges (dict): A dictionary to track expired edges.
    paths_activity (dict): A dictionary to track active paths and their timestamps.
    edge_usage (dict): A dictionary to track the usage counts of edges.
    link_usage (dict): A dictionary to track the usage counts of links between subgraphs.
    timestamp (str): The timestamp associated with the update files.

    Returns:
    tuple: Updated `update_nodes`, `update_edges`, and `expired_edges` dictionaries.
    """
    with ProcessPoolExecutor(max_workers=len(files)) as executor:
        print("submit 2")
        futures = [
            executor.submit(handle_updates_file, file_path, global_nodes, subgraph_connections, subgraph_nodes,
                            subgraph_edges, update_nodes, update_edges, paths_activity, edge_usage, link_usage,
                            timestamp)
            for file_path in sorted(files)]

    manage_expired_paths(paths_activity, edge_usage, link_usage, subgraph_edges, subgraph_connections, expired_edges,
                         timestamp)
    print("Total unique nodes across all subgraphs:", len(global_nodes))
    print("ToTal unique active paths in topology:", len(paths_activity.keys()))
    for subgraph_id, subgraph in subgraph_nodes.items():
        print(
            f"Patch = {subgraph_id}, Nodes = {len(subgraph_nodes[subgraph_id])}, Edges = {len(subgraph_edges[subgraph_id])},Connected subgraphs = {len(subgraph_connections[subgraph_id])}")

    return update_nodes, update_edges, expired_edges
