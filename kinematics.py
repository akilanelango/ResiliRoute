import math
import networkx as nx
from typing import List, Dict, Tuple

class KinematicSimulator:
    def __init__(self, max_speed_kmh: float = 90.0, max_accel_mps2: float = 3.0, dt: float = 1.0):
        """
        Initializes vehicle physics (FR-KIN-2).
        Defaults tuned for an ambulance response vehicle.
        """
        self.max_v = max_speed_kmh * (1000 / 3600)  # Convert km/h to m/s
        self.max_a = max_accel_mps2
        self.dt = dt  # Time step resolution in seconds (FR-KIN-1)

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2) -> float:
        R = 6371000.0  # Earth radius in meters
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    @staticmethod
    def bearing(lat1, lon1, lat2, lon2) -> float:
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        l1, l2 = math.radians(lon1), math.radians(lon2)
        y = math.sin(l2 - l1) * math.cos(phi2)
        x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(l2 - l1)
        return (math.degrees(math.atan2(y, x)) + 360) % 360

    def generate_trajectory(self, graph: nx.MultiDiGraph, path_nodes: List[int]) -> List[Dict]:
        """
        Simulates 1D motion along the route, computing speed and heading (FR-KIN-3, FR-KIN-4).
        """
        trajectory = []
        current_time = 0.0

        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i+1]
            lat1, lon1 = graph.nodes[u]['y'], graph.nodes[u]['x']
            lat2, lon2 = graph.nodes[v]['y'], graph.nodes[v]['x']
            
            segment_dist = self.haversine(lat1, lon1, lat2, lon2)
            segment_bearing = self.bearing(lat1, lon1, lat2, lon2)
            
            # Simple Kinematics: Time to accelerate/decelerate
            t_accel = self.max_v / self.max_a
            d_accel = 0.5 * self.max_a * (t_accel ** 2)
            
            # Check if segment is too short to reach max speed
            if 2 * d_accel > segment_dist:
                d_accel = segment_dist / 2
                peak_v = math.sqrt(2 * self.max_a * d_accel)
                t_accel = peak_v / self.max_a
                t_cruise = 0
            else:
                peak_v = self.max_v
                d_cruise = segment_dist - (2 * d_accel)
                t_cruise = d_cruise / peak_v
                
            segment_time = (2 * t_accel) + t_cruise
            
            # Step through the segment at resolution self.dt
            t = 0.0
            while t < segment_time:
                if t < t_accel:
                    current_v = self.max_a * t
                    dist_covered = 0.5 * self.max_a * (t ** 2)
                elif t < (t_accel + t_cruise):
                    current_v = peak_v
                    dist_covered = d_accel + peak_v * (t - t_accel)
                else:
                    t_dec = t - t_accel - t_cruise
                    current_v = peak_v - (self.max_a * t_dec)
                    dist_covered = d_accel + (peak_v * t_cruise) + (peak_v * t_dec - 0.5 * self.max_a * (t_dec ** 2))
                
                # Interpolate coordinate
                ratio = dist_covered / segment_dist if segment_dist > 0 else 0
                current_lat = lat1 + (lat2 - lat1) * ratio
                current_lon = lon1 + (lon2 - lon1) * ratio
                
                trajectory.append({
                    "time": round(current_time, 2),
                    "lat": current_lat,
                    "lon": current_lon,
                    "speed_mps": round(current_v, 2),
                    "heading": round(segment_bearing, 2)
                })
                
                t += self.dt
                current_time += self.dt
                
        return trajectory