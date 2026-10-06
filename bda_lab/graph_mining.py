"""
bda_lab/graph_mining.py
─────────────────────────────────────────────────────────────────────────────
Experiment 13: Mining Social-Network & Interaction Graphs.
  • 13a: Girvan-Newman Algorithm (Edge-betweenness community clustering)
  • 13b: Clique Percolation Method (CPM) for Overlapping Community Discovery

In modern recommendation platforms, collaborative filtering is augmented with
Graph Mining. By projecting the Bipartite User-Item rating graph into a
Movie Co-Occurrence Network (where edges represent shared high-rating users):
  1. Girvan-Newman iteratively removes edges with the highest Edge Betweenness
     Centrality to reveal cohesive thematic clusters (e.g. Matrix <-> Star Wars).
  2. Clique Percolation (k-cliques) discovers Overlapping Communities, allowing
     hybrid films (e.g. Sci-Fi + Horror) to simultaneously belong to multiple clusters.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import json
import itertools
from collections import defaultdict
from typing import Dict, List, Set, Any, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import networkx as nx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_PATH = os.path.join(BASE_DIR, "logs", "graph_communities.json")


def load_movie_metadata() -> Dict[int, Dict[str, str]]:
    """Load movie titles and genres."""
    all_path = os.path.join(DATA_DIR, "all_movies.json")
    cold_path = os.path.join(DATA_DIR, "cold_start_popular.json")
    path = all_path if os.path.exists(all_path) else cold_path

    movies = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for m in data:
                mid = m.get("movieId")
                if mid:
                    movies[int(mid)] = {
                        "title": m.get("title", f"Movie {mid}"),
                        "genres": m.get("genres", "Unknown"),
                    }
    return movies


def build_co_rating_graph(
    min_rating: float = 4.0,
    top_n_movies: int = 50,
    min_shared_users: int = 40,
    jaccard_threshold: float = 0.15
) -> Tuple[nx.Graph, Dict[int, Dict[str, str]]]:
    """
    Construct Movie Co-Rating Network.
    Nodes: Top N popular movies.
    Edges: Movies connected if Jaccard similarity of high-rating users exceeds threshold.
    """
    import csv
    ratings_path = os.path.join(DATA_DIR, "raw", "ratings.csv")
    movie_meta = load_movie_metadata()

    # movie -> set of users who rated >= min_rating
    movie_likers = defaultdict(set)
    total_ratings_per_movie = defaultdict(int)

    if os.path.exists(ratings_path):
        with open(ratings_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    try:
                        uid = int(row[0])
                        mid = int(row[1])
                        r = float(row[2])
                        total_ratings_per_movie[mid] += 1
                        if r >= min_rating:
                            movie_likers[mid].add(uid)
                    except ValueError:
                        continue

    # Select top N most rated movies
    sorted_movies = sorted(total_ratings_per_movie.keys(), key=lambda m: -total_ratings_per_movie[m])
    selected_movies = set(sorted_movies[:top_n_movies])

    G = nx.Graph()

    for mid in selected_movies:
        info = movie_meta.get(mid, {"title": f"Movie {mid}", "genres": "General"})
        G.add_node(
            mid,
            title=info["title"],
            genres=info["genres"],
            num_likers=len(movie_likers[mid])
        )

    # Compute pairwise Jaccard similarities
    movie_list = list(selected_movies)
    for i in range(len(movie_list)):
        for j in range(i + 1, len(movie_list)):
            m1, m2 = movie_list[i], movie_list[j]
            s1, s2 = movie_likers[m1], movie_likers[m2]
            intersection = len(s1 & s2)
            if intersection >= min_shared_users:
                union = len(s1 | s2)
                jaccard = intersection / union if union > 0 else 0
                if jaccard >= jaccard_threshold:
                    G.add_edge(m1, m2, weight=round(jaccard, 3), shared_users=intersection)

    return G, movie_meta


def run_girvan_newman(G: nx.Graph, num_clusters: int = 4) -> List[List[int]]:
    """
    Run Girvan-Newman community detection algorithm.
    Iteratively removes edges with highest edge betweenness centrality.
    """
    from networkx.algorithms.community import girvan_newman
    comp_gen = girvan_newman(G)
    communities = []
    # Advance generator until desired number of communities reached
    for comp in comp_gen:
        if len(comp) >= num_clusters:
            communities = [sorted(list(c)) for c in comp]
            break
    if not communities:
        # Fallback if graph is small/sparse
        communities = [sorted(list(c)) for c in nx.connected_components(G)]
    return communities


def run_clique_percolation(G: nx.Graph, k: int = 3) -> List[List[int]]:
    """
    Run Clique Percolation Method (CPM) for overlapping community detection.
    Finds k-cliques and connects them if they share k-1 vertices.
    """
    from networkx.algorithms.community import k_clique_communities
    cpm_result = list(k_clique_communities(G, k=k))
    return [sorted(list(c)) for c in cpm_result]


def execute_graph_analysis() -> Dict[str, Any]:
    """Execute complete graph mining pipeline and serialize results."""
    G, meta = build_co_rating_graph(top_n_movies=50, min_shared_users=100, jaccard_threshold=0.36)

    # Run Girvan-Newman
    gn_communities = run_girvan_newman(G, num_clusters=4)

    # Run CPM
    cpm_communities = run_clique_percolation(G, k=3)

    # Map node to GN community
    node_to_gn = {}
    for cid, comm in enumerate(gn_communities):
        for nid in comm:
            node_to_gn[nid] = cid

    # Find overlapping nodes in CPM
    cpm_membership = defaultdict(list)
    for cid, comm in enumerate(cpm_communities):
        for nid in comm:
            cpm_membership[nid].append(cid)

    overlapping_nodes = [nid for nid, comms in cpm_membership.items() if len(comms) > 1]

    # Package graph for visualization (nodes + edges)
    nodes_data = []
    for nid in G.nodes():
        node_info = G.nodes[nid]
        nodes_data.append({
            "id": nid,
            "title": node_info.get("title", f"Movie {nid}"),
            "genres": node_info.get("genres", ""),
            "degree": G.degree(nid),
            "gn_community": node_to_gn.get(nid, -1),
            "cpm_communities": cpm_membership.get(nid, []),
            "is_overlapping": nid in overlapping_nodes,
        })

    edges_data = []
    for u, v, d in G.edges(data=True):
        edges_data.append({
            "source": u,
            "target": v,
            "weight": d.get("weight", 0.2),
            "shared_users": d.get("shared_users", 0),
        })

    # Summary of discovered thematic clusters
    cluster_summaries = []
    for cid, comm in enumerate(gn_communities[:5]):
        sample_titles = [G.nodes[nid].get("title", str(nid)) for nid in comm if nid in G.nodes]
        cluster_summaries.append({
            "community_id": cid,
            "size": len(comm),
            "sample_movies": sample_titles[:4],
        })

    results = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "graph_density": round(nx.density(G), 4),
        "num_girvan_newman_communities": len(gn_communities),
        "girvan_newman_clusters": cluster_summaries,
        "num_cpm_communities": len(cpm_communities),
        "cpm_overlapping_nodes_count": len(overlapping_nodes),
        "overlapping_sample_movies": [G.nodes[nid].get("title") for nid in overlapping_nodes[:5] if nid in G.nodes],
        "nodes": nodes_data,
        "edges": edges_data,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    print("=" * 70)
    print("🕸️   BDA Experiment 13: Mining Social & Interaction Graphs Demonstration")
    print("=" * 70)
    res = execute_graph_analysis()

    print(f"Graph Nodes (Movies): {res['num_nodes']}")
    print(f"Graph Edges (High Co-Ratings): {res['num_edges']}")
    print(f"Graph Density: {res['graph_density']}")

    print(f"\n🧩  13a. Girvan-Newman Clusters Discovered: {res['num_girvan_newman_communities']}")
    for c in res["girvan_newman_clusters"]:
        print(f"   • Cluster #{c['community_id']} ({c['size']} films): {', '.join(c['sample_movies'])}")

    print(f"\n🔮  13b. Clique Percolation Method (k=3 CPM):")
    print(f"   • Discovered Overlapping Communities: {res['num_cpm_communities']}")
    print(f"   • Hybrid/Overlapping Films Found: {res['cpm_overlapping_nodes_count']}")
    if res['overlapping_sample_movies']:
        print(f"   • Examples: {', '.join(res['overlapping_sample_movies'])}")
    print("=" * 70)


if __name__ == "__main__":
    main()
