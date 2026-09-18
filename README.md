# ResiliRoute: End-to-End Emergency Routing Engine

ResiliRoute is a geospatial microservice designed to safely navigate emergency vehicles through post-disaster environments. It processes satellite imagery using Meta's Segment Anything Model (SAM) to dynamically identify flood obstructions, updates OpenStreetMap graph weights in real-time, and calculates kinematically viable traversal trajectories.

## 🏗️ System Architecture
1. **Vision Engine:** Ingests GeoTIFFs and extracts pixel-perfect flood masks using SAM (ViT-B) optimized for low-VRAM hardware.
2. **Geospatial Processing:** Reprojects raster masks to WGS84 Shapely polygons (`EPSG:4326`).
3. **Graph Theory Engine:** Fetches live driving networks via OSMnx and modifies Dijkstra edge weights based on spatial overlap with flood zones.
4. **Kinematic Simulator:** Simulates a trapezoidal velocity profile over the route to output time-indexed telemetry (Speed, Heading, Coordinates).
5. **FastAPI Microservice:** Wraps the pipeline into a RESTful API with automated file handling and error resolution.

## 🚀 Visualizations
*Naive shortest path (red) routing directly through flood zones vs. ResiliRoute's obstruction-aware path (green).*

![Routing Comparison](milestone3_routing.png)

## 📊 Performance Metrics (NVIDIA GTX 1050 Ti)

| Metric | Result | Target/Threshold |
| :--- | :--- | :--- |
| **Route Validity** | 100% | > 95% |
| **Segmentation Time** | ~4.5 sec | < 10 sec |
| **API Response Time** | ~8.0 sec | < 15 sec |

## ⚠️ Limitations & Future Work
* **VRAM Constraints:** Bound to `vit_b` (Base) weights; upgrading to `vit_h` (Huge) would improve edge detection accuracy at the cost of compute speed.
* **Topography:** Currently simulates 1D kinematics. Future iterations will integrate Digital Elevation Models (DEM) to account for vehicle grade resistance on inclines.