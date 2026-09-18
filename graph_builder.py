import networkx as nx
import osmnx as ox
from shapely.geometry import Polygon

def build_graph_from_bounds(bounds: tuple) -> nx.MultiDiGraph:
    """
    Fetches the OSM road network for a given bounding box (FR-NET-1).
    bounds format: (west, south, east, north) which matches (left, bottom, right, top)
    """
    print(f"[INFO] Fetching road network from OSMnx for bounds: {bounds}...")
    
    # Fetch driving network using the new OSMnx 2.0+ API
    G = ox.graph_from_bbox(bbox=bounds, network_type="drive")
    
    # Restrict to largest strongly connected component using NetworkX directly (FR-NET-3)
    # This prevents the 'utils_graph' AttributeError
    largest_cc = max(nx.strongly_connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    
    # Initialize base costs (travel time in seconds, or length if speed is missing)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    
    for u, v, key, data in G.edges(keys=True, data=True):
        # Base weight is travel time; fallback to length if missing (FR-NET-2)
        data["weight"] = data.get("travel_time", data.get("length", 1.0))
        data["base_weight"] = data["weight"]  # Store original for comparison
        data["status"] = "Clear"
        
    return G