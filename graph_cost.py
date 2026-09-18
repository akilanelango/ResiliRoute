import logging
import networkx as nx
from shapely.geometry import LineString, MultiPolygon, Polygon

# Configure logging for auditability (FR-COST-4)
logging.basicConfig(
    filename="cost_modifications.log", 
    level=logging.INFO, 
    format="%(asctime)s - %(message)s"
)

def apply_obstruction_costs(
    G: nx.MultiDiGraph, 
    obstruction_polygons: list[Polygon], 
    blocked_threshold: float = 0.3, 
    penalty_multiplier: float = 5.0
) -> nx.MultiDiGraph:
    """
    Intersects graph edges with obstruction polygons to modify traversal costs.
    """
    if not obstruction_polygons:
        return G
        
    combined_obstruction = MultiPolygon(obstruction_polygons)
    modifications = 0
    
    for u, v, key, data in G.edges(keys=True, data=True):
        # OSMnx edges might not have a geometry attribute if they are straight lines
        if "geometry" in data:
            edge_geom = data["geometry"]
        else:
            edge_geom = LineString([(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])])
            
        # Spatial intersection (FR-COST-1)
        if edge_geom.intersects(combined_obstruction):
            intersection = edge_geom.intersection(combined_obstruction)
            overlap_ratio = intersection.length / edge_geom.length
            
            # Classification (FR-COST-2 & FR-COST-3)
            if overlap_ratio >= blocked_threshold:
                data["status"] = "Blocked"
                data["weight"] = float("inf")
                logging.info(f"Edge ({u}, {v}) BLOCKED (Overlap: {overlap_ratio:.2f}). Weight: inf")
            else:
                data["status"] = "Degraded"
                data["weight"] = data["base_weight"] * penalty_multiplier
                logging.info(f"Edge ({u}, {v}) DEGRADED (Overlap: {overlap_ratio:.2f}). Weight: {data['weight']:.2f}")
            modifications += 1
            
    print(f"[INFO] Cost modification complete. {modifications} edges impacted.")
    return G