from concurrent.futures import ProcessPoolExecutor
from data_loader import parse_as_path
from graph_operations import add_node, add_edge, add_connections, update_edge_usage, update_connection_usage


def generate_subgraph(lines, global_nodes, subgraph_connections, subgraph_id, subgraph_nodes, subgraph_edges,
                      paths_activity, edge_usage, link_usage, timestamp):
    """
    Process a block of lines from a file, construct a subgraph, and record inter-subgraph connections.

    Parameters:
    lines (list): A list of lines from the file representing BGP data.
    global_nodes (dict): A global dictionary mapping nodes to subgraph IDs.
    subgraph_connections (dict): A dictionary mapping subgraph IDs to connected subgraph IDs.
    subgraph_id (int): The ID of the subgraph being generated.
    subgraph_nodes (dict): A dictionary mapping subgraph IDs to sets of nodes.
    subgraph_edges (dict): A dictionary mapping subgraph IDs to sets of edges.
    paths_activity (dict): A dictionary tracking active paths and their timestamps.
    edge_usage (dict): A dictionary tracking the usage counts of edges.
    link_usage (dict): A dictionary tracking the usage counts of links between subgraphs.
    timestamp (str): The timestamp associated with the current batch of lines.

    Returns:
    None
    """
    print("generating subgraph", subgraph_id)
    subgraph_nodes[subgraph_id] = set()
    subgraph_edges[subgraph_id] = set()
    for line in lines:
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
                global_nodes[node] = subgraph_id  # 记录节点所在的子图ID
                add_node(subgraph_nodes, subgraph_id, node)
            if previous_node:
                # 检查跨子图连接
                previous_graph_id = global_nodes[previous_node]
                current_graph_id = global_nodes[node]
                if previous_graph_id != current_graph_id:  # 不同子图
                    link = (previous_graph_id, current_graph_id)
                    path_links.add(link)
                    add_connections(subgraph_connections, previous_graph_id, current_graph_id)
                    update_connection_usage(link_usage, link, increase=True)
                else:  # 同一个子图
                    edge = (previous_node, node)
                    add_edge(subgraph_edges, previous_graph_id, edge)
                    path_edges.add((previous_node, node, current_graph_id))
                    update_edge_usage(edge_usage, edge, increase=True)

            previous_node = node

        paths_activity[path_id]['path_edges'] = path_edges
        paths_activity[path_id]['path_links'] = path_links

    return


def handle_rib_file(filename, global_nodes, subgraph_connections, subgraph_id_gen, subgraph_nodes, subgraph_edges,
                    paths_activity, edge_usage, link_usage, timestamp):
    """
    Handle and process a RIB (Routing Information Base) file to generate subgraphs parallel.
    Each chunk is a subgraph.

    Parameters:
    filename (str): The name of the RIB file to process.
    global_nodes (dict): A global dictionary mapping nodes to subgraph IDs.
    subgraph_connections (dict): A dictionary mapping subgraph IDs to connected subgraph IDs.
    subgraph_id_gen (iterator): An iterator to generate unique subgraph IDs.
    subgraph_nodes (dict): A dictionary mapping subgraph IDs to sets of nodes.
    subgraph_edges (dict): A dictionary mapping subgraph IDs to sets of edges.
    paths_activity (dict): A dictionary to track active paths and their timestamps.
    edge_usage (dict): A dictionary to track the usage counts of edges.
    link_usage (dict): A dictionary to track the usage counts of links between subgraphs.
    timestamp (str): The timestamp associated with the RIB file.

    Returns:
    None
    """
    with open(filename, 'r') as file:
        lines = file.readlines()

    subgraph_size = 50
    chunk_size = len(lines) // subgraph_size
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    print(len(chunks))
    with ProcessPoolExecutor() as executor:
        for idx, chunk in enumerate(chunks):
            executor.submit(generate_subgraph, chunk, global_nodes, subgraph_connections, next(subgraph_id_gen),
                            subgraph_nodes, subgraph_edges, paths_activity, edge_usage, link_usage,
                            timestamp)

    return


'''文件级别的并行处理，每个文件的处理作为一个独立的任务提交给进程池。并行级别设置为文件数。'''
def generate_initial_topology(files, global_nodes, subgraph_connections, subgraph_id_gen, subgraph_nodes,
                              subgraph_edges, paths_activity, edge_usage, link_usage, timestamp):
    """
    Generate the initial topology by processing RIB files in parallel.

    Parameters:
    files (list): A list of file paths to RIB files.
    global_nodes (dict): A global dictionary mapping nodes to subgraph IDs.
    subgraph_connections (dict): A dictionary mapping subgraph IDs to connected subgraph IDs.
    subgraph_id_gen (iterator): An iterator to generate unique subgraph IDs.
    subgraph_nodes (dict): A dictionary mapping subgraph IDs to sets of nodes.
    subgraph_edges (dict): A dictionary mapping subgraph IDs to sets of edges.
    paths_activity (dict): A dictionary to track active paths and their timestamps.
    edge_usage (dict): A dictionary to track the usage counts of edges.
    link_usage (dict): A dictionary to track the usage counts of links between subgraphs.

    Returns:
    None
    """
    with ProcessPoolExecutor(max_workers=len(files)) as executor:
        print("submit 1")
        futures = [
            executor.submit(handle_rib_file, file_path, global_nodes, subgraph_connections, subgraph_id_gen,
                            subgraph_nodes, subgraph_edges, paths_activity, edge_usage, link_usage, timestamp)
            for file_path in sorted(files)]

    print("Total unique nodes across all subgraphs:", len(global_nodes))
    print("ToTal unique active paths in topology:", len(paths_activity.keys()))
    for subgraph_id, subgraph in subgraph_nodes.items():
        print(
            f"Patch = {subgraph_id}, Nodes = {len(subgraph_nodes[subgraph_id])}, Edges = {len(subgraph_edges[subgraph_id])},Connected subgraphs = {len(subgraph_connections[subgraph_id])}")
