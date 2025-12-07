import math
import networkx as nx


def mtsp_dp(G: nx.Graph):
    """
    Solve Metric TSP (M-TSP) using Held-Karp DP.

    Input requirement (per spec):
      - G is COMPLETE
      - triangle inequality holds
      - contains node 0

    Returns:
      - tour: [0, ..., 0] visiting every node exactly once (except 0 repeated at end)
    """
    if 0 not in G:
        raise ValueError("Graph must contain node 0 for the starting point.")

    nodes = sorted(G.nodes())
    if len(nodes) == 1:
        return [0, 0]

    # Strictly enforce "complete graph" requirement
    nV = len(nodes)
    for i in range(nV):
        for j in range(i + 1, nV):
            if not G.has_edge(nodes[i], nodes[j]) and not G.has_edge(nodes[j], nodes[i]):
                raise ValueError("M-TSP input graph must be complete per spec.")

    # Distances: on a complete graph, dist[u][v] == edge weight (but we keep robust)
    dist_iter = nx.all_pairs_dijkstra_path_length(G, weight="weight")
    dist = {u: dict(lengths) for u, lengths in dist_iter}

    other_nodes = [node for node in nodes if node != 0]
    n = len(other_nodes)
    full_mask = (1 << n) - 1

    dp = {}      # dp[(mask, last_idx)] = min cost
    parent = {}  # parent[(mask, last_idx)] = (prev_mask, prev_idx)

    # init: visit one node then stop
    for idx, node in enumerate(other_nodes):
        mask = 1 << idx
        dp[(mask, idx)] = dist[0][node]
        parent[(mask, idx)] = (0, -1)

    # transitions
    for mask in range(1, full_mask + 1):
        for last_idx in range(n):
            if not (mask & (1 << last_idx)):
                continue
            key = (mask, last_idx)
            if key not in dp:
                continue
            cur_cost = dp[key]
            last_node = other_nodes[last_idx]

            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                nxt_node = other_nodes[nxt]
                nxt_mask = mask | (1 << nxt)
                new_cost = cur_cost + dist[last_node][nxt_node]
                dp_key = (nxt_mask, nxt)
                if new_cost < dp.get(dp_key, math.inf):
                    dp[dp_key] = new_cost
                    parent[dp_key] = (mask, last_idx)

    # close the tour back to 0
    best_cost = math.inf
    best_last = None
    for idx in range(n):
        key = (full_mask, idx)
        if key not in dp:
            continue
        cost = dp[key] + dist[other_nodes[idx]][0]
        if cost < best_cost:
            best_cost = cost
            best_last = idx

    if best_last is None:
        raise ValueError("TSP DP solver could not find a complete tour.")

    # reconstruct order
    mask = full_mask
    idx = best_last
    order = []
    while mask:
        order.append(other_nodes[idx])
        mask, idx = parent[(mask, idx)]
    order.reverse()

    return [0] + order + [0]
