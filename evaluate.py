import time
import argparse
import networkx as nx

from geometry import load_geotiff_for_sam, mask_to_polygons
from segmentation import ObstructionSegmenter
from graph_builder import build_graph_from_bounds
from graph_cost import apply_obstruction_costs
from router import RoutePlanner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--start", type=float, nargs=2, required=True)
    parser.add_argument("--end", type=float, nargs=2, required=True)
    args = parser.parse_args()

    metrics = {}
    
    # 1. Vision Segmentation
    t0 = time.time()
    image_rgb, transform, crs = load_geotiff_for_sam(args.image)
    segmenter = ObstructionSegmenter(checkpoint_path=args.weights)
    binary_mask = segmenter.segment_automatic(image_rgb)
    polygons = mask_to_polygons(binary_mask, transform=transform, src_crs=crs)
    metrics["segmentation_time_sec"] = round(time.time() - t0, 2)
    metrics["obstructions_found"] = len(polygons)

    # 2. Graph Construction
    t1 = time.time()
    import rasterio
    from rasterio.warp import transform_bounds
    with rasterio.open(args.image) as src:
        w, s, e, n = transform_bounds(src.crs, "EPSG:4326", src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        bounds = (w-0.01, s-0.01, e+0.01, n+0.01)
    
    G = build_graph_from_bounds(bounds)
    G_modified = apply_obstruction_costs(G, polygons)
    metrics["graph_build_time_sec"] = round(time.time() - t1, 2)

    # 3. Routing & Route Validity Check
    t2 = time.time()
    planner = RoutePlanner(G_modified)
    path = planner.find_path(tuple(args.start), tuple(args.end))
    metrics["routing_time_sec"] = round(time.time() - t2, 2)
    
    if path:
        blocked_edges = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if G_modified[u][v][0].get('status') == 'Blocked':
                blocked_edges += 1
        
        metrics["route_validity_percent"] = 100.0 if blocked_edges == 0 else 0.0
    else:
        metrics["route_validity_percent"] = None

    print("\n=== RESILIROUTE QUANTITATIVE EVALUATION ===")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v}")

if __name__ == "__main__":
    main()