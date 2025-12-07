import networkx as nx

from mtsp_dp import mtsp_dp


def php_solver_from_tsp(G, H):
    """
    PHP solver via reduction to metric TSP.

    Parameters:
        G (nx.Graph or nx.DiGraph): Graph where nodes represent locations.
        H (list): List of home nodes that must be visited.

    Returns:
        list: A tour (list of nodes) starting and ending at 0 that visits every home.
    """
    if 0 not in G:
        raise ValueError("Graph must contain node 0 for the starting location.")

    homes = sorted(set(H))
    reduced_nodes = [0] + homes

    # Work on an undirected copy to simplify distance computations.
    undirected = nx.Graph()
    for u, v, data in G.edges(data=True):
        undirected.add_edge(u, v, weight=float(data["weight"]))

    # Pre-compute all-pairs shortest path distances and paths.
    dist_iter = nx.all_pairs_dijkstra_path_length(undirected, weight="weight")
    dist = {u: dict(lengths) for u, lengths in dist_iter}
    path_iter = nx.all_pairs_dijkstra_path(undirected, weight="weight")
    all_paths = {u: dict(paths) for u, paths in path_iter}

    # Build the complete metric graph for the reduction.
    reduced_graph = nx.DiGraph()
    for u in reduced_nodes:
        reduced_graph.add_node(u)
    for i, u in enumerate(reduced_nodes):
        for v in reduced_nodes[i + 1 :]:
            weight = dist[u][v]
            reduced_graph.add_edge(u, v, weight=weight)
            reduced_graph.add_edge(v, u, weight=weight)

    tsp_tour = mtsp_dp(reduced_graph)

    # Expand the tour to the original graph using the stored shortest paths.
    tour = [tsp_tour[0]]
    for i in range(1, len(tsp_tour)):
        u = tsp_tour[i - 1]
        v = tsp_tour[i]
        segment = all_paths[u][v]
        tour.extend(segment[1:])  # skip duplicated start

    if tour[-1] != 0:
        tour.append(0)
    if len(tour) == 1:
        tour.append(0)

    return tour
