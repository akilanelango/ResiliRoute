import csv
import simplekml
from shapely.geometry import Polygon
from typing import List, Dict

def export_trajectory_csv(trajectory: List[Dict], filepath: str = "trajectory.csv"):
    """Exports time-series trajectory to CSV (FR-EXP-2)."""
    if not trajectory:
        return
        
    keys = trajectory[0].keys()
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(trajectory)
    print(f"[SUCCESS] Trajectory CSV exported to {filepath}")

def export_resiliroute_kml(
    trajectory: List[Dict], 
    polygons: List[Polygon], 
    filepath: str = "resiliroute_output.kml"
):
    """
    Exports route, trajectory points, and flood polygons to Google Earth KML (FR-EXP-1).
    """
    kml = simplekml.Kml()
    
    # 1. Add Obstruction Polygons (Blue)
    poly_folder = kml.newfolder(name="Flood Obstructions")
    for idx, poly in enumerate(polygons):
        ext_coords = list(poly.exterior.coords)
        kml_poly = poly_folder.newpolygon(name=f"Obstruction_{idx}")
        # simplekml expects (lon, lat)
        kml_poly.outerboundaryis = [(lon, lat) for lon, lat in ext_coords]
        kml_poly.style.polystyle.color = simplekml.Color.changealphaint(150, simplekml.Color.blue)
        kml_poly.style.linestyle.color = simplekml.Color.darkblue
        kml_poly.style.linestyle.width = 2
        
    # 2. Add Continuous Route Line (Green)
    if trajectory:
        route_folder = kml.newfolder(name="Evacuation Route")
        route_coords = [(pt["lon"], pt["lat"]) for pt in trajectory]
        linestring = route_folder.newlinestring(name="Path", coords=route_coords)
        linestring.style.linestyle.color = simplekml.Color.green
        linestring.style.linestyle.width = 5
        
        # 3. Add Waypoints indicating simulation updates
        traj_folder = kml.newfolder(name="Kinematic Waypoints")
        for i, pt in enumerate(trajectory):
            # Only drop a pin every 5 seconds to prevent map clutter
            if i % 5 == 0:
                pnt = traj_folder.newpoint(name=f"t={pt['time']}s")
                pnt.coords = [(pt["lon"], pt["lat"])]
                pnt.description = f"Speed: {pt['speed_mps']} m/s\nHeading: {pt['heading']}°"
                pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
                pnt.style.iconstyle.scale = 0.5
                
    kml.save(filepath)
    print(f"[SUCCESS] 3D Google Earth Visualization exported to {filepath}")