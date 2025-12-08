import networkx as nx
from typing import Dict, List, Set, Tuple, Optional


# =========================
# Main solver
# =========================

def ptp_solver(G: nx.DiGraph, H: list, alpha: float):
    """
    PTP solver.

    Constraints enforced:
    - tour begins and ends at 0
    - tour follows existing edges (via shortest-path expansion)
    - pick-up locations are nodes on the final expanded tour
    - each friend picked exactly once
    - pick-up for friend h must be h or a neighbor of h
    """
    if 0 not in G:
        raise ValueError("Graph must contain node 0 for the starting location.")

    # Build an undirected weighted graph (the input DiGraph is equivalent by construction)
    undirected = nx.Graph()
    for u, v, data in G.edges(data=True):
        undirected.add_edge(int(u), int(v), weight=float(data["weight"]))

    # Precompute all-pairs shortest path distances and paths
    dist_iter = nx.all_pairs_dijkstra_path_length(undirected, weight="weight")
    dist: Dict[int, Dict[int, float]] = {u: dict(lengths) for u, lengths in dist_iter}
    path_iter = nx.all_pairs_dijkstra_path(undirected, weight="weight")
    all_paths: Dict[int, Dict[int, List[int]]] = {u: dict(paths) for u, paths in path_iter}

    homes = list(map(int, H))

    # Allowed pickup nodes for each friend home: {home} ∪ N(home)
    neighbor_map: Dict[int, Set[int]] = {}
    for h in homes:
        nbrs = set(undirected.neighbors(h))
        nbrs.add(h)
        neighbor_map[h] = nbrs

    # Candidate nodes to consider inserting/removing.
    # (Good default: only homes+neighbors matter for feasibility/walking; 0 already included.)
    candidate_nodes: List[int] = sorted(set().union({0}, *neighbor_map.values()))

    # Start macro tour as [0,0]
    macro_tour = compress_macro_tour([0, 0])

    # Local search (insert/delete)
    current_eval = evaluate_macro_tour(
        macro_tour, homes, neighbor_map, dist, all_paths, alpha
    )

    while True:
        best_tour = None
        best_eval = None

        for node in candidate_nodes:
            if node == 0:
                continue

            if node_in_tour(macro_tour, node):
                cand_tour = remove_node(macro_tour, node)
                if cand_tour is None:
                    continue
                cand_eval = evaluate_macro_tour(
                    cand_tour, homes, neighbor_map, dist, all_paths, alpha
                )
            else:
                cand_tour, cand_eval = best_insertion(
                    macro_tour, node, homes, neighbor_map, dist, all_paths, alpha
                )
                if cand_tour is None:
                    continue

            if best_eval is None or better_eval(cand_eval, best_eval):
                best_tour, best_eval = cand_tour, cand_eval

        # Accept only if it improves over current
        if best_eval is not None and better_eval(best_eval, current_eval):
            macro_tour, current_eval = best_tour, best_eval
        else:
            break

    # IMPORTANT FIX: repair on MACRO tour (never on expanded tour list)
    macro_tour = repair_macro_tour_to_feasible(
        macro_tour, homes, neighbor_map, dist, all_paths
    )

    # Expand to a valid edge-by-edge tour using shortest paths
    tour = expand_macro_tour(macro_tour, all_paths)

    # Build pick_up_locs_dict strictly satisfying constraints (now feasible by construction)
    pick_up_locs_dict = build_pickup_dict(
        tour=tour,
        homes=homes,
        neighbor_map=neighbor_map,
        dist=dist,
    )

    return tour, pick_up_locs_dict


# =========================
# Evaluation helpers
# =========================

class TourEvaluation:
    __slots__ = ("total_cost", "infeasible_count", "driving_cost", "walking_cost", "assignments")

    def __init__(self, total_cost, infeasible_count, driving_cost, walking_cost, assignments):
        self.total_cost = float(total_cost)
        self.infeasible_count = int(infeasible_count)
        self.driving_cost = float(driving_cost)   # already multiplied by alpha
        self.walking_cost = float(walking_cost)
        # assignments: dict {pickup_node: [home,...]} (may include fallback if infeasible)
        self.assignments = assignments


def expanded_node_set_from_macro(macro_tour: List[int], all_paths: Dict[int, Dict[int, List[int]]]) -> Set[int]:
    """Return the set of nodes visited by the expanded shortest-path tour of this macro tour."""
    macro_tour = compress_macro_tour(macro_tour)
    visited: Set[int] = set()
    for i in range(1, len(macro_tour)):
        u = macro_tour[i - 1]
        v = macro_tour[i]
        seg = all_paths[u][v]
        visited.update(seg)
    if not visited:
        visited.add(0)
    return visited


def evaluate_macro_tour(
    macro_tour: List[int],
    homes: List[int],
    neighbor_map: Dict[int, Set[int]],
    dist: Dict[int, Dict[int, float]],
    all_paths: Dict[int, Dict[int, List[int]]],
    alpha: float,
) -> TourEvaluation:
    """
    Evaluate a MACRO tour, but walking feasibility is checked against the *expanded* tour nodes.
    Driving is computed as alpha * sum shortest-path distances between consecutive macro nodes.
    """
    macro_tour = compress_macro_tour(macro_tour)

    # Driving cost (already alpha-weighted)
    driving_cost = 0.0
    for i in range(1, len(macro_tour)):
        driving_cost += dist[macro_tour[i - 1]][macro_tour[i]] * alpha

    # IMPORTANT FIX: pickup availability is based on expanded tour nodes (tau), not macro nodes only
    tau_nodes = expanded_node_set_from_macro(macro_tour, all_paths)

    infeasible = 0
    walking_cost = 0.0
    assignments: Dict[int, List[int]] = {}

    for h in homes:
        allowed_in_tau = neighbor_map[h] & tau_nodes
        if allowed_in_tau:
            pickup = min(allowed_in_tau, key=lambda node: dist[h][node])
            walking_cost += dist[h][pickup]
            assignments.setdefault(pickup, []).append(h)
        else:
            infeasible += 1
            # fallback only for search guidance; not valid final solution if infeasible remains
            fallback = min(tau_nodes, key=lambda node: dist[h][node])
            walking_cost += dist[h][fallback]
            assignments.setdefault(fallback, []).append(h)

    total_cost = driving_cost + walking_cost
    return TourEvaluation(total_cost, infeasible, driving_cost, walking_cost, assignments)


def better_eval(a: TourEvaluation, b: TourEvaluation) -> bool:
    """Return True if eval a is better than eval b (lexicographic: infeasible_count, then total_cost)."""
    if a.infeasible_count != b.infeasible_count:
        return a.infeasible_count < b.infeasible_count
    return a.total_cost < b.total_cost - 1e-6


# =========================
# Tour construction helpers
# =========================

def compress_macro_tour(tour: List[int]) -> List[int]:
    """Remove consecutive duplicates; ensure starts/ends with 0."""
    if not tour:
        return [0, 0]
    macro = [int(tour[0])]
    for node in tour[1:]:
        node = int(node)
        if node != macro[-1]:
            macro.append(node)

    if macro[0] != 0:
        macro.insert(0, 0)
    if macro[-1] != 0:
        macro.append(0)
    if len(macro) == 1:
        macro.append(0)
    return macro


def expand_macro_tour(macro_tour: List[int], all_paths: Dict[int, Dict[int, List[int]]]) -> List[int]:
    """Expand macro tour into an edge-by-edge tour by concatenating shortest paths."""
    macro_tour = compress_macro_tour(macro_tour)
    tour = [macro_tour[0]]
    for i in range(1, len(macro_tour)):
        u = macro_tour[i - 1]
        v = macro_tour[i]
        segment = all_paths[u][v]
        tour.extend(segment[1:])
    if tour[0] != 0:
        tour.insert(0, 0)
    if tour[-1] != 0:
        tour.append(0)
    if len(tour) == 1:
        tour.append(0)
    return tour


def build_pickup_dict(
    tour: List[int],
    homes: List[int],
    neighbor_map: Dict[int, Set[int]],
    dist: Dict[int, Dict[int, float]],
) -> Dict[int, List[int]]:
    """
    Construct a pick_up_locs_dict that strictly satisfies:
    - pickup location in final expanded tour
    - for each friend h, pickup in {h} ∪ N(h)
    - each friend appears exactly once
    """
    tour_set = set(map(int, tour))

    pick_up_locs_dict: Dict[int, List[int]] = {}
    for h in homes:
        allowed = neighbor_map[h] & tour_set
        if not allowed:
            # should not happen after macro repair; keep defensive behavior
            pickup = h if h in tour_set else 0
        else:
            pickup = min(allowed, key=lambda node: dist[h][node])
        pick_up_locs_dict.setdefault(pickup, []).append(h)

    # Keep only pickup nodes that are in tour (required)
    pick_up_locs_dict = {k: v for k, v in pick_up_locs_dict.items() if k in tour_set}
    return pick_up_locs_dict


# =========================
# Local search operators
# =========================

def node_in_tour(macro_tour: List[int], node: int) -> bool:
    macro_tour = compress_macro_tour(macro_tour)
    node = int(node)
    # exclude endpoints 0
    return any(step == node for step in macro_tour[1:-1])


def remove_node(macro_tour: List[int], node: int) -> Optional[List[int]]:
    macro_tour = compress_macro_tour(macro_tour)
    node = int(node)
    if node == 0:
        return None
    new_tour = [macro_tour[0]]
    for step in macro_tour[1:-1]:
        if step != node:
            new_tour.append(step)
    new_tour.append(macro_tour[-1])
    new_tour = compress_macro_tour(new_tour)
    if len(new_tour) < 2:
        return None
    return new_tour


def best_insertion(
    macro_tour: List[int],
    node: int,
    homes: List[int],
    neighbor_map: Dict[int, Set[int]],
    dist: Dict[int, Dict[int, float]],
    all_paths: Dict[int, Dict[int, List[int]]],
    alpha: float,
) -> Tuple[Optional[List[int]], Optional[TourEvaluation]]:
    macro_tour = compress_macro_tour(macro_tour)
    node = int(node)

    best_tour = None
    best_eval = None

    for idx in range(len(macro_tour) - 1):
        candidate = macro_tour[: idx + 1] + [node] + macro_tour[idx + 1 :]
        cand_eval = evaluate_macro_tour(candidate, homes, neighbor_map, dist, all_paths, alpha)
        if best_eval is None or better_eval(cand_eval, best_eval):
            best_tour = candidate
            best_eval = cand_eval

    if best_tour is not None:
        best_tour = compress_macro_tour(best_tour)
    return best_tour, best_eval


# =========================
# Feasibility repair (MACRO tour only)
# =========================

def repair_macro_tour_to_feasible(
    macro_tour: List[int],
    homes: List[int],
    neighbor_map: Dict[int, Set[int]],
    dist: Dict[int, Dict[int, float]],
    all_paths: Dict[int, Dict[int, List[int]]],
) -> List[int]:
    """
    Ensure: for every home h, at least one allowed pickup node (h or neighbor)
    appears on the *expanded* tour nodes of this macro_tour.

    Repair strategy:
    - Compute expanded node set tau_nodes from macro_tour
    - For any missing h, insert the best allowed node p in neighbor_map[h]
      into the macro tour at the cheapest position (min added driving distance).
    """
    macro_tour = compress_macro_tour(macro_tour)

    def tau_nodes(t: List[int]) -> Set[int]:
        return expanded_node_set_from_macro(t, all_paths)

    changed = True
    while changed:
        changed = False
        tau = tau_nodes(macro_tour)

        missing = [h for h in homes if len(neighbor_map[h] & tau) == 0]
        if not missing:
            break

        # Greedy: fix one missing friend at a time with the globally cheapest insertion
        best_global = None  # (delta, insert_pos, pickup_node)
        best_h = None

        for h in missing:
            allowed = list(neighbor_map[h])  # candidates to insert
            for p in allowed:
                # find best insertion position for p
                best_pos = None
                best_delta = None
                for i in range(len(macro_tour) - 1):
                    a = macro_tour[i]
                    b = macro_tour[i + 1]
                    delta = dist[a][p] + dist[p][b] - dist[a][b]
                    if best_delta is None or delta < best_delta:
                        best_delta = delta
                        best_pos = i + 1
                if best_pos is not None:
                    cand = (best_delta, best_pos, p)
                    if best_global is None or cand[0] < best_global[0] - 1e-12:
                        best_global = cand
                        best_h = h

        if best_global is None:
            # Should not happen in connected graphs; return as-is defensively
            return macro_tour

        _, pos, p = best_global
        macro_tour = compress_macro_tour(macro_tour[:pos] + [p] + macro_tour[pos:])
        changed = True

    return macro_tour
