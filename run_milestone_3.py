import argparse
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import rasterio

from geometry import load_geotiff_for_sam, mask_to_polygons
from segmentation import ObstructionSegmenter
from graph_builder import build_graph_from_bounds
from graph_cost import apply_obstruction_costs
from router import RoutePlanner

from rasterio.warp import transform_bounds

def get_geotiff_bounds(image_path: str) -> tuple:
    with rasterio.open(image_path) as src:
        # Transform the native bounds into WGS84 (lon/lat) which OSMnx requires
        west, south, east, north = transform_bounds(
            src.crs, 
            "EPSG:4326", 
            src.bounds.left, 
            src.bounds.bottom, 
            src.bounds.right, 
            src.bounds.top
        )
        
        # Add a buffer (0.01 degrees ~ 1km) to guarantee we capture surrounding roads
        buffer = 0.01 
        buffered_bounds = (west - buffer, south - buffer, east + buffer, north + buffer)
        
        print(f"[DEBUG] Native CRS: {src.crs}")
        print(f"[DEBUG] Original Bounds (WGS84): ({west:.5f}, {south:.5f}, {east:.5f}, {north:.5f})")
        print(f"[DEBUG] Buffered Bounds for OSMnx: {buffered_bounds}")
        
        return buffered_bounds

def main():
    parser = argparse.ArgumentParser(description="ResiliRoute Milestone 3: Routing")
    parser.add_argument("--image", required=True, help="Path to GeoTIFF")
    parser.add_argument("--weights", required=True, help="Path to SAM weights")
    parser.add_argument("--start", type=float, nargs=2, required=True, metavar=("LAT", "LON"))
    parser.add_argument("--end", type=float, nargs=2, required=True, metavar=("LAT", "LON"))
    args = parser.parse_args()

    # 1. Run Pipeline up to M2
    image_rgb, transform, crs = load_geotiff_for_sam(args.image)
    segmenter = ObstructionSegmenter(checkpoint_path=args.weights)
    binary_mask = segmenter.segment_automatic(image_rgb)
    polygons = mask_to_polygons(binary_mask, transform=transform, src_crs=crs)

    # 2. Build Road Network for the image area
    bounds = get_geotiff_bounds(args.image)
    G = build_graph_from_bounds(bounds)

    # 3. Compute Naive Path (Before Obstructions)
    planner = RoutePlanner(G)
    naive_path = planner.find_path(tuple(args.start), tuple(args.end))

    # 4. Modify Graph Costs based on Obstructions
    G_modified = apply_obstruction_costs(G, polygons)
    
    # 5. Compute Resilient Path (After Obstructions)
    planner_modified = RoutePlanner(G_modified)
    resilient_path = planner_modified.find_path(tuple(args.start), tuple(args.end))

    # 6. Visualization
    fig, ax = ox.plot_graph_routes(
        G, 
        routes=[naive_path, resilient_path], 
        route_colors=["red", "green"], 
        route_linewidths=[4, 4],
        show=False, close=False,
        node_size=0
    )
    
    # Overlay flood polygons
    for poly in polygons:
        x, y = poly.exterior.xy
        ax.fill(x, y, color="blue", alpha=0.4)
        
    plt.title("Naive Shortest Path (Red) vs Obstruction-Aware Route (Green)")
    plt.savefig("milestone3_routing.png", dpi=300)
    print("[DONE] Visual comparison saved to milestone3_routing.png")

if __name__ == "__main__":
    main()