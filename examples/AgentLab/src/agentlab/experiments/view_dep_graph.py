"""Dirty script to visualize the dependency graph of a benchmark, e.g. webarena, vsisualwebarena,
etc. You may have to detust it to make it work for you."""

import math
import bgym
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np


def clean_dict(dependency_dict: dict[str, list[str]]) -> dict[str, list[str]]:
    new_dep = {}
    for key, deps in dependency_dict.items():
        new_key = key.split(".")[-1]

        new_dep[new_key] = [dep.split(".")[-1] for dep in deps]
    return new_dep


def dict_to_networkx(dependency_dict: dict[str, list[str]]) -> nx.DiGraph:

    G = nx.DiGraph()
    i = 0
    # Add edges from each node to its dependencies
    for node, dependencies in dependency_dict.items():
        i += 1
        if i > 20:
            pass

        print(node, dependencies)
        # Add edges from the node to each of its dependencies
        for dep in dependencies:
            G.add_edge(dep, node)
    return G


def plot_graph(G, ax, title=None, node_color="lightblue", node_size=40, font_size=8):
    """
    Plot a single graph component on the given matplotlib axis.

    Args:
        G: NetworkX graph (should be a single connected component)
        ax: Matplotlib axis to plot on
        title: Optional title for the subplot
        node_color: Color for the nodes
        node_size: Size of the nodes
        font_size: Size of the node labels
    """
    # Use a simple layout for better performance
    # pos = nx.spring_layout(G, k=0.1, iterations=100)

    pos = nx.kamada_kawai_layout(G)

    # pos = nx.spectral_layout(G)

    def name_to_size(name):
        if "-" in name:
            start, end = name.split("-")

            n_nodes = int(end) - int(start) + 1
        else:
            n_nodes = 1
        size_factor = node_size / 10
        return n_nodes * size_factor

    # compute size based on name
    sizes = [name_to_size(name) for name in G.nodes]

    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        node_color=node_color,
        node_size=sizes,
        font_size=font_size,
        font_weight="normal",
        arrows=True,
        arrowsize=15,
    )

    if title:
        ax.set_title(title)
    ax.axis("off")


def plot_components_grid(
    components, max_cols=4, node_color="lightblue", node_size=2000, font_size=10
):
    """
    Plot components in a grid layout.

    Args:
        components: List of NetworkX graphs, one per component
        max_cols: Maximum number of columns in the grid
        node_color: Color for the nodes
        node_size: Size of the nodes
        font_size: Size of the node labels

    Returns:
        matplotlib figure
    """
    n_components = len(components)

    if n_components == 0:
        print("No components found")
        return None

    # Calculate grid dimensions
    ncols = min(n_components, max_cols)
    nrows = math.ceil(n_components / ncols)

    # Create figure with a reasonable size per subplot
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    fig.suptitle("Dependency Graph Components", size=16)

    # Make axes iterable even if there's only one
    if n_components == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    # Plot each component
    for idx, component in enumerate(components):
        i, j = divmod(idx, ncols)
        title = f"Component {idx+1} ({component.number_of_nodes()} nodes)"
        plot_graph(
            component,
            axes[i, j],
            title,
            node_color=node_color,
            node_size=node_size,
            font_size=font_size,
        )

    # Remove empty subplots
    for idx in range(n_components, nrows * ncols):
        i, j = divmod(idx, ncols)
        axes[i, j].remove()

    plt.tight_layout()
    return fig


def compress_sequential_chains(dep_dict: dict[str, list[str]]) -> dict[str, list[str]]:
    """
    Compress chains of sequential numbers in a dependency dictionary.
    Returns a new dictionary with compressed chains using range notation.

    Args:
        dep_dict: Dictionary mapping string numbers to list of string number dependencies

    Returns:
        Dictionary with compressed chains using range notation
    """
    # Convert to integers for easier processing
    int_dict = {int(k): [int(x) for x in v] for k, v in dep_dict.items()}

    # Find chains
    chains = []
    current_chain = []

    # Sort nodes for sequential processing
    nodes = sorted(int_dict.keys())

    i = 0
    while i < len(nodes):
        node = nodes[i]

        # Start new chain
        if not current_chain:
            current_chain = [node]
            i += 1
            continue

        # Check if this node continues the chain
        last_node = current_chain[-1]

        # Conditions for chain continuation:
        # 1. Numbers are consecutive
        # 2. Current node has exactly one dependency
        # 3. That dependency is the previous node in chain
        # 4. The previous node has exactly one successor
        is_consecutive = node == last_node + 1
        has_single_dep = len(int_dict[node]) == 1
        deps_on_last = has_single_dep and int_dict[node][0] == last_node
        last_has_single_successor = sum(1 for k, v in int_dict.items() if last_node in v) == 1

        if is_consecutive and deps_on_last and last_has_single_successor:
            current_chain.append(node)
        else:
            if len(current_chain) > 1:
                chains.append(current_chain)
            current_chain = [node]

        i += 1

    # Add last chain if it exists
    if len(current_chain) > 1:
        chains.append(current_chain)

    # Create compressed dictionary
    compressed_dict = {}
    processed_nodes = set()

    # Add compressed chains
    for chain in chains:
        chain_name = f"{chain[0]}-{chain[-1]}"
        # Find dependencies of first node in chain
        deps = int_dict[chain[0]]
        compressed_dict[chain_name] = [str(d) for d in deps]
        processed_nodes.update(chain)

    # Add remaining non-chain nodes
    for node in nodes:
        if node not in processed_nodes:
            compressed_dict[str(node)] = [str(d) for d in int_dict[node]]

    # Update dependencies to use compressed names
    for k in compressed_dict:
        deps = compressed_dict[k]
        new_deps = []
        for dep in deps:
            dep_int = int(dep)
            # Find if this dependency is part of a chain
            chain_found = False
            for chain in chains:
                if dep_int in chain:
                    new_deps.append(f"{chain[0]}-{chain[-1]}")
                    chain_found = True
                    break
            if not chain_found:
                new_deps.append(dep)
        compressed_dict[k] = new_deps

    return compressed_dict


def compress_chains(G):
    """
    Compress chains in a directed graph by merging nodes that have single parent and single child.

    Args:
        G: NetworkX directed graph

    Returns:
        NetworkX directed graph with compressed chains
    """
    G_compressed = G.copy()
    processed_nodes = set()

    while True:
        # Find nodes with exactly one parent and one child
        nodes_to_compress = []
        for node in list(
            G_compressed.nodes()
        ):  # Create a list to avoid modification during iteration
            if node in processed_nodes:
                continue

            predecessors = list(G_compressed.predecessors(node))
            successors = list(G_compressed.successors(node))

            if len(predecessors) == 1 and len(successors) == 1:
                pred = predecessors[0]
                succ = successors[0]

                # Skip if any node in the chain is already processed
                if pred in processed_nodes or succ in processed_nodes:
                    continue

                # Only compress if middle node has single parent/child
                pred_preds = list(G_compressed.predecessors(pred))
                succ_succs = list(G_compressed.successors(succ))

                if len(pred_preds) <= 1 and len(succ_succs) <= 1:
                    nodes_to_compress.append((pred, node, succ))
                    processed_nodes.update([pred, node, succ])

        if not nodes_to_compress:
            break

        # Process each chain
        for pred, mid, succ in nodes_to_compress:
            if not all(G_compressed.has_node(n) for n in [pred, mid, succ]):
                continue

            # Create new merged node name
            new_node = ",".join(str(n) for n in [pred, mid, succ])

            # Add the new node
            G_compressed.add_node(new_node)

            # Add edges from all predecessors of first node
            for p in list(G_compressed.predecessors(pred)):
                G_compressed.add_edge(p, new_node)

            # Add edges to all successors of last node
            for s in list(G_compressed.successors(succ)):
                G_compressed.add_edge(new_node, s)

            # Remove the old nodes
            G_compressed.remove_nodes_from([pred, mid, succ])

    return G_compressed


# benchmark = bgym.DEFAULT_BENCHMARKS["webarena"]()
benchmark = bgym.DEFAULT_BENCHMARKS["visualwebarena"]()

dep_graph = benchmark.dependency_graph_over_tasks()
dep_graph = clean_dict(dep_graph)

dep_graph = compress_sequential_chains(dep_graph)
graph = dict_to_networkx(dep_graph)

# graph = compress_chains(graph)

components = nx.weakly_connected_components(graph)
components = [graph.subgraph(component).copy() for component in components]
plot_components_grid(components)
plt.show()
