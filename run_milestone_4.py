import argparse
from geometry import load_geotiff_for_sam, mask_to_polygons
from segmentation import ObstructionSegmenter
from graph_builder import build_graph_from_bounds
from graph_cost import apply_obstruction_costs
from router import RoutePlanner
from kinematics import KinematicSimulator
from export_kml import export_trajectory_csv, export_resiliroute_kml

def get_geotiff_bounds(image_path: str) -> tuple:
    import rasterio
    from rasterio.warp import transform_bounds
    with rasterio.open(image_path) as src:
        w, s, e, n = transform_bounds(src.crs, "EPSG:4326", src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        return (w - 0.01, s - 0.01, e + 0.01, n + 0.01)

def main():
    parser = argparse.ArgumentParser(description="ResiliRoute Milestone 4: Kinematics & Export")
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--start", type=float, nargs=2, required=True, metavar=("LAT", "LON"))
    parser.add_argument("--end", type=float, nargs=2, required=True, metavar=("LAT", "LON"))
    args = parser.parse_args()

    # M2: Vision & Geospatial
    image_rgb, transform, crs = load_geotiff_for_sam(args.image)
    segmenter = ObstructionSegmenter(checkpoint_path=args.weights)
    binary_mask = segmenter.segment_automatic(image_rgb)
    polygons = mask_to_polygons(binary_mask, transform=transform, src_crs=crs)

    # M3: Graph & Routing
    bounds = get_geotiff_bounds(args.image)
    G = build_graph_from_bounds(bounds)
    G_modified = apply_obstruction_costs(G, polygons)
    planner = RoutePlanner(G_modified)
    resilient_path = planner.find_path(tuple(args.start), tuple(args.end))

    # M4: Kinematics & Export
    if resilient_path:
        simulator = KinematicSimulator(max_speed_kmh=90, max_accel_mps2=3.0)
        trajectory = simulator.generate_trajectory(G_modified, resilient_path)
        
        export_trajectory_csv(trajectory, "resiliroute_trajectory.csv")
        export_resiliroute_kml(trajectory, polygons, "resiliroute_output.kml")
    else:
        print("[WARNING] Cannot run kinematics or export without a valid route.")

if __name__ == "__main__":
    main()