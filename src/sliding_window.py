from graph_operations import add_edge

WINDOW_SIZE = 40000


def remove_expired_edge(edge, edge_usage, subgraph_edges, expired_edges):
    """
    Remove an expired edge from the subgraph and update the expired edges sets record.

    Parameters:
    edge (tuple): A tuple representing the edge (node1, node2, subgraph_id).
    edge_usage (dict): A dictionary that maps edges to their usage counts.
    subgraph_edges (dict): A dictionary mapping subgraph IDs to sets of edges.
    expired_edges (dict): A dictionary to track expired edges for each subgraph.

    Returns:
    None
    """
    removed_edge = (edge[0], edge[1])
    subgraph_id = edge[2]
    if removed_edge not in edge_usage.keys():
        return
    if edge_usage[removed_edge] <= 0:  # 删除edge
        subgraph_edges[subgraph_id].remove(removed_edge)
        add_edge(expired_edges, subgraph_id, removed_edge)
        del edge_usage[edge]
        print("delete edge", edge)


def remove_expired_link(link, link_usage, subgraph_connections):
    """
    Remove an expired link between subgraphs.

    Parameters:
    link (tuple): A tuple representing the link (subgraph_id1, subgraph_id2).
    link_usage (dict): A dictionary that maps links to their usage counts.
    subgraph_connections (dict): A dictionary mapping subgraph IDs to connected subgraph IDs.

    Returns:
    None
    """
    if link not in link_usage:
        return
    if link_usage[link] <= 0:
        subgraph_connections[link[0]].remove(link[1])
        del link_usage[link]
        print("delete link", link)


def manage_expired_paths(paths_activity, edge_usage, link_usage, subgraph_edges, subgraph_connections, expired_edges, timestamp):
    """
    Sliding Window to remove expired paths and update the subgraphs and global information.

    Parameters:
    paths_activity (dict): A dictionary to track active paths and their timestamps.
    edge_usage (dict): A dictionary to track the usage counts of edges.
    link_usage (dict): A dictionary to track the usage counts of links between subgraphs.
    subgraph_edges (dict): A dictionary mapping subgraph IDs to sets of edges.
    subgraph_connections (dict): A dictionary mapping subgraph IDs to connected subgraph IDs.
    expired_edges (dict): A dictionary to track expired edges for each subgraph.
    timestamp (str): The current timestamp.

    Returns:
    tuple: Updated `expired_paths` and `expired_edges` dictionaries.
    """
    expired_paths = []
    for path_id, path_data in paths_activity.items():
        last_time = path_data['timestamp']  # 从嵌套字典中取出timestamp

        if int(timestamp) - int(last_time) >= WINDOW_SIZE:
            expired_paths.append(path_id)

    for p_id in expired_paths:  # 过期的PATH删除
        print("delete path ", p_id)
        for edge in paths_activity[p_id]['path_edges']:
            remove_expired_edge(edge, edge_usage, subgraph_edges, expired_edges)
        for link in paths_activity[p_id]['path_links']:
            remove_expired_link(link, link_usage, subgraph_connections)
        del paths_activity[p_id]

    return expired_paths, expired_edges

