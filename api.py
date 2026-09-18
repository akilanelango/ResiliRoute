import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

# Import pipeline modules
from geometry import load_geotiff_for_sam, mask_to_polygons
from segmentation import ObstructionSegmenter
from graph_builder import build_graph_from_bounds
from graph_cost import apply_obstruction_costs
from router import RoutePlanner
from kinematics import KinematicSimulator

app = FastAPI(title="ResiliRoute API", version="1.0")

# Fixed model weights path for the microservice
WEIGHTS_PATH = "weights/sam_vit_b_01ec64.pth"

def get_geotiff_bounds(image_path: str) -> tuple:
    import rasterio
    from rasterio.warp import transform_bounds
    with rasterio.open(image_path) as src:
        w, s, e, n = transform_bounds(
            src.crs, "EPSG:4326", src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top
        )
        return (w - 0.01, s - 0.01, e + 0.01, n + 0.01)

@app.get("/health")
def health_check():
    """FR-API-2: Health-check endpoint"""
    return {"status": "healthy", "service": "ResiliRoute Engine"}

@app.post("/route")
async def compute_route(
    start_lat: float = Form(...),
    start_lon: float = Form(...),
    end_lat: float = Form(...),
    end_lon: float = Form(...),
    image: UploadFile = File(...)
):
    """FR-API-1: Accepts image + coords, returns JSON trajectory"""
    
    # FR-API-3: Meaningful error for invalid file types
    if not image.filename.endswith(('.tif', '.tiff')):
        raise HTTPException(status_code=400, detail="Only GeoTIFF (.tif) files are supported.")

    # Save uploaded image to a temporary file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        content = await image.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 1. Vision & Geospatial
        try:
            image_rgb, transform, crs = load_geotiff_for_sam(tmp_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image metadata error: {str(e)}")

        segmenter = ObstructionSegmenter(checkpoint_path=WEIGHTS_PATH, model_type="vit_b")
        binary_mask = segmenter.segment_automatic(image_rgb)
        polygons = mask_to_polygons(binary_mask, transform=transform, src_crs=crs)

        # 2. Graph Building & Bounding Box Validation
        try:
            bounds = get_geotiff_bounds(tmp_path)
            w, s, e, n = bounds
            
            # FR-API-3: Verify coordinates are within the playable map area
            if not (s <= start_lat <= n and w <= start_lon <= e):
                raise HTTPException(status_code=400, detail="Start point is outside the image bounds.")
            if not (s <= end_lat <= n and w <= end_lon <= e):
                raise HTTPException(status_code=400, detail="End point is outside the image bounds.")
                
            G = build_graph_from_bounds(bounds)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to build road network: {str(e)}")

        # 3. Cost Modification & Routing
        G_modified = apply_obstruction_costs(G, polygons)
        planner = RoutePlanner(G_modified)
        resilient_path = planner.find_path((start_lat, start_lon), (end_lat, end_lon))

        # FR-API-3: Handle the "No Path Exists" scenario
        if not resilient_path:
            raise HTTPException(
                status_code=404, 
                detail="No viable route exists between coordinates due to severe flooding."
            )

        # 4. Kinematics
        simulator = KinematicSimulator(max_speed_kmh=90, max_accel_mps2=3.0)
        trajectory = simulator.generate_trajectory(G_modified, resilient_path)

        return JSONResponse(content={
            "status": "success",
            "obstructions_detected": len(polygons),
            "route_node_count": len(resilient_path),
            "trajectory": trajectory
        })

    finally:
        # Ensure temporary file is deleted even if the pipeline crashes
        os.remove(tmp_path)