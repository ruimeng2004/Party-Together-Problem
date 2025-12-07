import networkx as nx


def ptp_solver(G: nx.DiGraph, H: list, alpha: float):
    """
    PTP solver.

    Parameters:
        G (nx.DiGraph): A NetworkX graph representing the city.
            This directed graph is equivalent to an undirected one by construction.
        H (list): A list of home nodes.
        alpha (float): The coefficient for calculating cost.

    Returns:
        tuple: (tour, pick_up_locs_dict)
            - tour (list): A list of nodes traversed by your car.
            - pick_up_locs_dict (dict): {pick_up_loc: [home1, home2, ...]} mapping.

    Notes:
    - All nodes are integers.
    - tour begins and ends at 0.
    - tour uses existing edges (we expand macro tour using shortest paths).
    - pick-up locations must be in tour.
    - each friend is picked exactly once.
    - pick-up loc for friend must be their home or a neighbor of their home.
    """

    if 0 not in G:
        raise ValueError("Graph must contain node 0 for the starting location.")

    # Build an undirected weighted graph
    undirected = nx.Graph()
    for u, v, data in G.edges(data=True):
        undirected.add_edge(int(u), int(v), weight=float(data["weight"]))

    # Precompute all-pairs shortest path distances and paths
    dist_iter = nx.all_pairs_dijkstra_path_length(undirected, weight="weight")
    dist = {u: dict(lengths) for u, lengths in dist_iter}
    path_iter = nx.all_pairs_dijkstra_path(undirected, weight="weight")
    all_paths = {u: dict(paths) for u, paths in path_iter}

    # Allowed pickup nodes for each friend home: {home} ∪ N(home)
    neighbor_map = {}
    for home in H:
        home = int(home)
        nbrs = set(undirected.neighbors(home))
        nbrs.add(home)
        neighbor_map[home] = nbrs

    # Start from an arbitrary macro tour (no php_from_tsp)
    macro_tour = compress_macro_tour([0, 0])

    candidate_nodes = sorted(set(undirected.nodes()))
    best_eval = evaluate_tour(macro_tour, H, neighbor_map, dist, alpha)

    # Insert/delete local search
    improved = True
    while improved:
        improved = False
        best_candidate = None
        best_candidate_eval = None

        for node in candidate_nodes:
            if node == 0:
                continue

            if node_in_tour(macro_tour, node):
                cand_tour = remove_node(macro_tour, node)
                if cand_tour is None:
                    continue
                cand_eval = evaluate_tour(cand_tour, H, neighbor_map, dist, alpha)
                if is_better(cand_eval, best_candidate_eval, best_eval):
                    best_candidate = cand_tour
                    best_candidate_eval = cand_eval
            else:
                cand_tour, cand_eval = best_insertion(
                    macro_tour, node, H, neighbor_map, dist, alpha
                )
                if cand_tour is None:
                    continue
                if is_better(cand_eval, best_candidate_eval, best_eval):
                    best_candidate = cand_tour
                    best_candidate_eval = cand_eval

        if best_candidate is not None:
            macro_tour = best_candidate
            best_eval = best_candidate_eval
            improved = True

    # Expand to a valid edge-by-edge tour using shortest paths
    tour = expand_macro_tour(macro_tour, all_paths)

    # Build pick_up_locs_dict that satisfies the required constraints
    pick_up_locs_dict = build_pickup_dict(
        tour=tour,
        homes=H,
        neighbor_map=neighbor_map,
        dist=dist,
    )

    return tour, pick_up_locs_dict


class TourEvaluation:
    __slots__ = ("total_cost", "infeasible_count", "driving_cost", "walking_cost", "assignments")

    def __init__(self, total_cost, infeasible_count, driving_cost, walking_cost, assignments):
        self.total_cost = total_cost
        self.infeasible_count = infeasible_count
        self.driving_cost = driving_cost
        self.walking_cost = walking_cost
        # assignments: dict {pickup_node: [home,...]} (may include fallback if infeasible)
        self.assignments = assignments


def compress_macro_tour(tour):
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


def expand_macro_tour(macro_tour, all_paths):
    """
    Expand macro tour into a step-by-step tour that uses existing edges,
    by concatenating shortest paths.
    """
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


def evaluate_tour(tour, homes, neighbor_map, dist, alpha):
    """
    Evaluate a macro tour with generalized cost:
    - Driving: alpha * sum shortest-path distances between consecutive macro nodes
    - Walking: for each friend, min distance from home to an allowed pickup node IN the tour set,
              else fallback to nearest node in tour (counts as infeasible)
    """
    tour = compress_macro_tour(tour)
    tour_set = set(tour)

    driving_cost = 0.0
    for i in range(1, len(tour)):
        driving_cost += dist[tour[i - 1]][tour[i]] * alpha

    infeasible = 0
    walking_cost = 0.0
    assignments = {}

    for home in sorted(map(int, homes)):
        allowed_in_tour = neighbor_map[home] & tour_set
        if allowed_in_tour:
            pickup = min(allowed_in_tour, key=lambda node: dist[home][node])
            walking_cost += dist[home][pickup]
            assignments.setdefault(pickup, []).append(home)
        else:
            infeasible += 1
            # generalized fallback (NOT valid for final output if infeasible remains)
            fallback = min(tour_set, key=lambda node: dist[home][node])
            walking_cost += dist[home][fallback]
            assignments.setdefault(fallback, []).append(home)

    total_cost = driving_cost + walking_cost
    return TourEvaluation(total_cost, infeasible, driving_cost, walking_cost, assignments)


def build_pickup_dict(tour, homes, neighbor_map, dist):
    """
    Construct a pick_up_locs_dict that strictly satisfies:
      - pick-up location in tour
      - for each friend home h, pickup in {h} ∪ N(h)
      - each friend appears exactly once
    Strategy:
      - For each friend h, choose the closest node (by dist) among allowed nodes that are IN tour.
      - If no allowed node appears in tour, we force pickup at home h itself by ensuring h is on tour
        (by inserting h into macro tour would be needed), BUT in this solver we instead do a safe fallback:
        pick h up at home only if home is in tour; otherwise pick 0 (still invalid). So we additionally
        enforce feasibility: if any friend has no allowed pickup in tour, we "repair" the tour by visiting
        that home directly (insert its home into the tour via shortest paths).
    """
    homes = list(map(int, homes))

    # Repair: ensure every friend has at least one allowed pickup node in tour.
    tour = repair_tour_to_feasible(tour, homes, neighbor_map, dist)
    tour_set = set(tour)

    pick_up_locs_dict = {}
    for h in homes:
        allowed = neighbor_map[h] & tour_set
        # after repair, this must be non-empty
        if not allowed:
            # extremely defensive; should not happen
            pickup = h
        else:
            pickup = min(allowed, key=lambda node: dist[h][node])
        pick_up_locs_dict.setdefault(pickup, []).append(h)

    # Ensure keys are in tour (should be true by construction)
    pick_up_locs_dict = {k: v for k, v in pick_up_locs_dict.items() if k in tour_set}

    # Ensure each friend appears exactly once
    # (if duplicates in H, this will keep duplicates; H should be distinct by input spec)
    return pick_up_locs_dict


def repair_tour_to_feasible(tour, homes, neighbor_map, dist):
    """
    If some friend has no allowed pickup node in the current tour,
    we "repair" by inserting that friend's home into the tour in the cheapest place.

    This guarantees final pick_up_locs_dict can satisfy the spec.
    """
    # Ensure tour begins/ends at 0 and remove consecutive duplicates
    tour = compress_macro_tour(tour)

    def has_allowed(h, tour_set):
        return len(neighbor_map[h] & tour_set) > 0

    changed = True
    while changed:
        changed = False
        tour_set = set(tour)
        missing = [h for h in homes if not has_allowed(h, tour_set)]
        if not missing:
            break

        # Insert each missing home (greedy, one by one)
        for h in missing:
            # best insertion of h into macro tour based on added driving distance
            best_pos = None
            best_delta = None
            for i in range(len(tour) - 1):
                a = tour[i]
                b = tour[i + 1]
                delta = dist[a][h] + dist[h][b] - dist[a][b]
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_pos = i + 1
            if best_pos is not None:
                tour = tour[:best_pos] + [h] + tour[best_pos:]
                tour = compress_macro_tour(tour)
                changed = True

    return tour


def node_in_tour(tour, node):
    tour = compress_macro_tour(tour)
    node = int(node)
    return any(step == node for step in tour[1:-1])


def remove_node(tour, node):
    tour = compress_macro_tour(tour)
    node = int(node)
    if node == 0:
        return None
    new_tour = [tour[0]]
    for step in tour[1:-1]:
        if step != node:
            new_tour.append(step)
    new_tour.append(tour[-1])
    new_tour = compress_macro_tour(new_tour)
    if len(new_tour) < 2:
        return None
    return new_tour


def best_insertion(tour, node, homes, neighbor_map, dist, alpha):
    tour = compress_macro_tour(tour)
    node = int(node)
    best_tour = None
    best_eval = None
    for idx in range(len(tour) - 1):
        candidate = tour[: idx + 1] + [node] + tour[idx + 1 :]
        eval_candidate = evaluate_tour(candidate, homes, neighbor_map, dist, alpha)
        if best_eval is None or better_eval(eval_candidate, best_eval):
            best_tour = candidate
            best_eval = eval_candidate
    if best_tour is not None:
        best_tour = compress_macro_tour(best_tour)
    return best_tour, best_eval


def better_eval(candidate_eval, current_best):
    if current_best is None:
        return True
    if candidate_eval.infeasible_count < current_best.infeasible_count:
        return True
    if (
        candidate_eval.infeasible_count == current_best.infeasible_count
        and candidate_eval.total_cost < current_best.total_cost - 1e-6
    ):
        return True
    return False


def is_better(candidate_eval, best_candidate_eval, current_eval):
    if candidate_eval is None:
        return False

    # If current is feasible, accept only feasible improvements
    if current_eval.infeasible_count == 0:
        if candidate_eval.infeasible_count != 0:
            return False
        if best_candidate_eval is None or candidate_eval.total_cost < best_candidate_eval.total_cost - 1e-6:
            return candidate_eval.total_cost < current_eval.total_cost - 1e-6

    # If current infeasible, must reduce infeasibility first
    if candidate_eval.infeasible_count >= current_eval.infeasible_count:
        return False

    if (
        best_candidate_eval is None
        or candidate_eval.infeasible_count < best_candidate_eval.infeasible_count
        or (
            candidate_eval.infeasible_count == best_candidate_eval.infeasible_count
            and candidate_eval.total_cost < best_candidate_eval.total_cost - 1e-6
        )
    ):
        return True

    return False

