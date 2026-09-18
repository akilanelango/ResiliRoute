import argparse
import sys
import matplotlib.pyplot as plt
import torch

from geometry import load_geotiff_for_sam, mask_to_polygons
from segmentation import ObstructionSegmenter


def parse_args():
    parser = argparse.ArgumentParser(
        description="ResiliRoute Milestone 2: Process GeoTIFF satellite scenes with SAM to vector polygons."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input satellite GeoTIFF (.tif) file."
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the SAM checkpoint file (.pth)."
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
        help="SAM model architecture. Default is 'vit_b' for 4GB VRAM limits."
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=2500,
        help="Minimum pixel area threshold to discard minor segment artifacts."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="milestone2_verification.png",
        help="Output filepath for the visual comparison artifact."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Hardware detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using execution device: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[INFO] Hardware: {gpu_name} ({vram_gb:.2f} GB total VRAM)")

    # 1. Load GeoTIFF and affine spatial metadata
    print(f"[INFO] Ingesting GeoTIFF: {args.image}")
    try:
        image_rgb, transform, crs = load_geotiff_for_sam(args.image)
        print(f"[INFO] Image shape: {image_rgb.shape}, Detected CRS: {crs}")
    except Exception as exc:
        sys.exit(f"[ERROR] Failed to load GeoTIFF: {exc}")

    # 2. Segment flood obstructions using SAM
    segmenter = ObstructionSegmenter(
        checkpoint_path=args.weights,
        model_type=args.model_type,
        device=device
    )

    print("[INFO] Computing segmentation mask proposals...")
    with torch.inference_mode():
        if device == "cuda":
            with torch.cuda.amp.autocast():
                binary_mask = segmenter.segment_automatic(
                    image_rgb,
                    min_area_threshold=args.min_area
                )
        else:
            binary_mask = segmenter.segment_automatic(
                image_rgb,
                min_area_threshold=args.min_area
            )

    # 3. Transform mask to georeferenced WGS84 polygons
    print("[INFO] Converting binary mask to WGS84 Shapely polygons...")
    try:
        polygons = mask_to_polygons(binary_mask, transform=transform, src_crs=crs)
    except Exception as exc:
        sys.exit(f"[ERROR] Polygon extraction failed: {exc}")

    print(f"[SUCCESS] Extracted {len(polygons)} georeferenced polygon(s).")
    for idx, poly in enumerate(polygons[:3]):
        centroid = poly.centroid
        print(f"  - Polygon {idx}: Centroid = ({centroid.x:.6f}, {centroid.y:.6f}) [Lon, Lat]")

    # 4. Generate visual verification figure
    print(f"[INFO] Exporting verification figure to {args.output}...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Satellite scene + SAM mask overlay
    axes[0].imshow(image_rgb)
    axes[0].imshow(binary_mask, cmap="Blues", alpha=0.45)
    axes[0].set_title("GeoTIFF Scene + SAM Flood Overlay")
    axes[0].axis("off")

    # Georeferenced WGS84 vector coordinates
    for poly in polygons:
        x, y = poly.exterior.xy
        axes[1].plot(x, y, color="crimson", linewidth=1.2)
        axes[1].fill(x, y, color="red", alpha=0.35)

    axes[1].set_title("Georeferenced Obstructions (WGS84 Coordinates)")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"[DONE] Milestone 2 complete. Artifact saved to {args.output}.")


if __name__ == "__main__":
    main()