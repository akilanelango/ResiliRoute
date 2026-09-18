import networkx as nx
import osmnx as ox

class RoutePlanner:
    def __init__(self, graph: nx.MultiDiGraph):
        self.G = graph
        
    def find_path(self, start_coords: tuple, end_coords: tuple) -> list:
        """
        Computes the shortest path avoiding obstructions.
        Coordinates should be (latitude, longitude) (FR-ROUTE-1).
        """
        # Snap coordinates to the nearest graph nodes (OSMnx expects lon, lat)
        orig_node = ox.distance.nearest_nodes(self.G, X=start_coords[1], Y=start_coords[0])
        dest_node = ox.distance.nearest_nodes(self.G, X=end_coords[1], Y=end_coords[0])
        
        try:
            # Dijkstra's algorithm using the modified weight (FR-ROUTE-2)
            path = nx.shortest_path(self.G, source=orig_node, target=dest_node, weight="weight")
            
            # Validate the path does not contain blocked edges
            path_weight = nx.path_weight(self.G, path, weight="weight")
            if path_weight == float("inf"):
                raise nx.NetworkXNoPath("Optimal path contains blocked edges.")
                
            print(f"[SUCCESS] Route computed. Path is optimal under current edge weights (FR-ROUTE-3).")
            return path
            
        except nx.NetworkXNoPath:
            # Graceful failure handling (FR-ROUTE-4)
            print("[ERROR] No viable path exists between the start and end points.")
            return []