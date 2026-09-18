from typing import List, Tuple
import cv2
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.features import shapes
from rasterio.warp import transform_geom
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.validation import make_valid


def load_geotiff_for_sam(image_path: str) -> Tuple[np.ndarray, rasterio.Affine, CRS]:
    """
    Reads a GeoTIFF, extracts RGB channels, normalizes to 8-bit uint8,
    and returns image data with spatial transform metadata (FR-SEG-1, FR-GEO-2).
    """
    with rasterio.open(image_path) as src:
        transform = src.transform
        crs = src.crs

        # Read bands: read 3 channels if available, or repeat single channel to create pseudo-RGB
        if src.count >= 3:
            img = src.read([1, 2, 3])
        elif src.count == 1:
            band = src.read(1)
            img = np.stack([band, band, band], axis=0)
        else:
            bands = [src.read(i) for i in range(1, src.count + 1)]
            while len(bands) < 3:
                bands.append(bands[0])
            img = np.stack(bands[:3], axis=0)

        # Transpose from rasterio (Bands, Height, Width) to (Height, Width, Bands)
        img = np.transpose(img, (1, 2, 0))

        # Normalize 16-bit satellite imagery or float bands down to standard 8-bit RGB (0-255)
        if img.dtype == np.uint16:
            img = (img / 256.0).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        return img, transform, crs


def mask_to_polygons(
    binary_mask: np.ndarray,
    transform: rasterio.Affine,
    src_crs: CRS
) -> List[Polygon]:
    """
    Converts binary raster mask into cleaned Shapely Polygons reprojected
    to geographic WGS84 coordinates (FR-GEO-1, FR-GEO-2, FR-GEO-3).
    """
    mask_bool = (binary_mask > 0).astype(np.uint8)
    shape_generator = shapes(mask_bool, mask=mask_bool, transform=transform)

    valid_polygons = []
    target_crs = "EPSG:4326"

    for geom_dict, value in shape_generator:
        if value != 1:
            continue

        # If image CRS is projected (e.g., UTM), reproject geometry directly to WGS84 lat/lon
        if src_crs and src_crs.to_string() != target_crs:
            geom_dict = transform_geom(src_crs, target_crs, geom_dict)

        geom = shape(geom_dict)
        cleaned_geom = make_valid(geom)

        if cleaned_geom.is_empty:
            continue

        if isinstance(cleaned_geom, Polygon):
            valid_polygons.append(cleaned_geom)
        elif isinstance(cleaned_geom, MultiPolygon):
            valid_polygons.extend([poly for poly in cleaned_geom.geoms if not poly.is_empty])

    return valid_polygons