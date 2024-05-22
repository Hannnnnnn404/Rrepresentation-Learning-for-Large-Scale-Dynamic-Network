# Rrepresentation Learning for Large Scale Dynamic Network
### Module

#### `data_loader.py`
Responsible for parsing AS paths and other information from BGP data files.

#### `GNN.py`
Defines the structure of the Graph Neural Network (GNN) model. Includes initialization and forward propagation, primarily for learning embeddings of graph data.

#### `graph_construction.py`
Contains functions for constructing subgraphs. Processes file blocks to construct subgraphs and records inter-subgraph connections.

#### `graph_operations.py`
Contains functions for performing various operations on the graph, such as adding nodes, adding edges, and updating connection usage.

#### `graph_update.py`
Responsible for updating the structure and properties of the graph. Contains functions for processing update data blocks and updating subgraphs and global information.

#### `hierarchical_cluster.py`
Implements hierarchical clustering algorithms to cluster subgraphs into larger structures. Suitable for hierarchical clustering tasks on graph-structured data.

#### `main.py`
The entry point of the project, responsible for coordinating the workflow of various modules. Calls functions from other modules to perform data loading, graph construction, updates, and embedding computations.

#### `sliding_window.py`
Implements the sliding window algorithm for processing time-series data. Suitable for managing and updating dynamic graphs, ensuring data is processed within a given time window.

## Run
Run `main.py` to start the entire workflow.
