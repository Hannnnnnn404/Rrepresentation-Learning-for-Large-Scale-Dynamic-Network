
def add_node(subgraph_nodes, subgraph_id, node):
    """
    Add a node to a subgraph.

    Parameters:
    subgraph_nodes (dict): Dictionary mapping subgraph IDs to sets of nodes.
    subgraph_id (int): The ID of the subgraph to which the node will be added.
    node (str): The node to add to the subgraph.

    Returns:
    None
    """
    if subgraph_id not in subgraph_nodes:
        subgraph_nodes[subgraph_id] = set()

    temp_set = subgraph_nodes[subgraph_id]
    temp_set.add(node)
    subgraph_nodes[subgraph_id] = temp_set


def add_edge(subgraph_edges, subgraph_id, edge):
    """
    Add an edge to a subgraph.

    Parameters:
    subgraph_edges (dict): Dictionary mapping subgraph IDs to sets of edges.
    subgraph_id (int): The ID of the subgraph to which the edge will be added.
    edge (tuple): A tuple representing the edge to add (node1, node2).

    Returns:
    None
    """
    if subgraph_id not in subgraph_edges:
        subgraph_edges[subgraph_id] = set()
    temp_set = subgraph_edges[subgraph_id]
    temp_set.add(edge)
    subgraph_edges[subgraph_id] = temp_set


def add_connections(subgraph_connections, graph_id1, graph_id2):
    """
    Add a connection between two subgraphs.

    Parameters:
    subgraph_connections (dict): Dictionary mapping subgraph IDs to sets of connected subgraph IDs.
    subgraph_id_1 (int): The ID of the first subgraph.
    subgraph_id_2 (int): The ID of the second subgraph.

    Returns:
    None
    """
    if graph_id1 not in subgraph_connections:
        subgraph_connections[graph_id1] = set()
    if graph_id2 not in subgraph_connections:
        subgraph_connections[graph_id2] = set()

    # 获取连接集，添加新连接，然后更新字典
    conn_set1 = subgraph_connections[graph_id1]
    conn_set2 = subgraph_connections[graph_id2]

    conn_set1.add(graph_id2)
    conn_set2.add(graph_id1)

    # 重新赋值以确保更新被同步
    subgraph_connections[graph_id1] = conn_set1
    subgraph_connections[graph_id2] = conn_set2


def update_edge_usage(edge_usage, edge, increase=True):
    """
    Update the usage count of a given edge.

    Parameters:
    edge_usage (dict): A dictionary that maps edges to their usage counts.
    edge (tuple): A tuple representing the edge (node1, node2).
    increase (bool): A flag indicating whether to increase (True) or decrease (False) the usage count. Default is True.

    Returns:
    None
    """
    if increase:
        if edge in edge_usage:
            edge_usage[edge] += 1
        else:
            edge_usage[edge] = 1
    else:
        if edge in edge_usage:
            edge_usage[edge] -= 1


def update_connection_usage(link_usage, link, increase=True):
    """
    Update the usage count of a connection (link) between two subgraphs.

    Parameters:
    link_usage (dict): A dictionary that maps links to their usage counts.
    link (tuple): A tuple representing the link (subgraph_id1, subgraph_id2).
    increase (bool): A flag indicating whether to increase (True) or decrease (False) the usage count. Default is True.

    Returns:
    None
    """
    if increase:
        if link in link_usage:
            link_usage[link] += 1
        else:
            link_usage[link] = 1
    else:
        if link in link_usage:
            link_usage[link] -= 1
