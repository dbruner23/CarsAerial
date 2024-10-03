import torch
import numpy as np
import supervision as sv
import cv2
import matplotlib.pyplot as plt
import math

from osgeo import gdal, ogr
import geopandas as gpd
from shapely.geometry import Point, Polygon
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from scipy.spatial import cKDTree
from tqdm.notebook import tqdm

from typing import List, Tuple

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pixel_to_geo(geotransform, x, y):
    geo_x = geotransform[0] + x * geotransform[1] + y * geotransform[2]
    geo_y = geotransform[3] + x * geotransform[4] + y * geotransform[5]
    return geo_x, geo_y

def convert_polygon_coordinates(polygon_pixels, geotransform):
    return [pixel_to_geo(geotransform, x, y) for x, y in polygon_pixels]

def load_geotif(image_path):
    dataset = gdal.Open(image_path)
    if dataset is None:
        print(f"Failed to open file '{image_path}'")
        return None, None, None

    geotransform = dataset.GetGeoTransform()

    projection = dataset.GetProjection()

    return dataset, geotransform, projection

def load_sam_model(checkpoint_path, device=DEVICE):
    print(f"Using device: {device}")
    sam = sam_model_registry["vit_h"](checkpoint_path).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator

def preprocess_for_sam(image_array):
    # Ensure the image is in RGB format
    if len(image_array.shape) == 2:  # If it's a single-band image
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif image_array.shape[2] == 4:  # If it's an RGBA image
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    elif image_array.shape[2] == 3:  # If it's already RGB
        image_rgb = image_array
    else:
        raise ValueError("Unsupported number of channels in the image")

    # Ensure the image is in uint8 format
    if image_rgb.dtype != np.uint8:
        image_rgb = (image_rgb / np.max(image_rgb) * 255).astype(np.uint8)

    return image_rgb

def process_tile(tile: np.ndarray, mask_generator: SamAutomaticMaskGenerator) -> List[dict]:
    tile_rgb = preprocess_for_sam(tile)
    sam_result = mask_generator.generate(tile_rgb)
    return sam_result

def process_geotif_with_sam_tiled(image_path, sam_checkpoint_path, shapefilename, tile_size: int = 1024, overlap: int = 100):
    dataset, geotransform, projection = load_geotif(image_path)
    if dataset is None:
        return

    mask_generator = load_sam_model(sam_checkpoint_path)

    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount
    print("Image size:", width, height)
    print("Number of bands:", bands)

    all_polygons = []
    all_car_polygons = []
    all_car_centroids = []

    num_tiles_x = math.ceil(width / (tile_size - overlap))
    num_tiles_y = math.ceil(height / (tile_size - overlap))
    total_tiles = num_tiles_x * num_tiles_y
    print(f"Number of tiles: {num_tiles_x} x {num_tiles_y} = {total_tiles}")

    pbar = tqdm(total=total_tiles, desc="Processing tiles", unit="tile")

    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            x = j * (tile_size - overlap)
            y = i * (tile_size - overlap)
            tile_width = min(tile_size, width - x)
            tile_height = min(tile_size, height - y)
            tile = dataset.ReadAsArray(x, y, tile_width, tile_height)

            if bands > 1:
                tile = np.transpose(tile, (1, 2, 0))

            sam_start_time = cv2.getTickCount()

            sam_result = process_tile(tile, mask_generator)

            sam_end_time = cv2.getTickCount()
            sam_time = (sam_end_time - sam_start_time) / cv2.getTickFrequency()
            print(f"Tile {i * num_tiles_x + j + 1}/{total_tiles} processed in {sam_time:.2f} seconds")

            detections = sv.Detections.from_sam(sam_result=sam_result)

            polygon_geometries = []

            for mask in detections.mask:
                mask_uint8 = mask.astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if len(contour) >= 3:
                        polygon = [tuple(point[0]) for point in contour]
                        # Adjust polygon coordinates to the tile's position in the full image
                        adjusted_polygon = [(p[0] + x, p[1] + y) for p in polygon]
                        geo_coords = convert_polygon_coordinates(adjusted_polygon, geotransform)
                        if len(geo_coords) >= 3:
                            polygon = Polygon(geo_coords)
                            if polygon.is_valid:
                                polygon_geometries.append(polygon)

            tile_car_polygons = filter_car_polygons_aerial(polygon_geometries)
            tile_car_centroids = [polygon.centroid for polygon in tile_car_polygons]

            all_polygons.extend(polygon_geometries)
            all_car_polygons.extend(tile_car_polygons)
            all_car_centroids.extend(tile_car_centroids)

            pbar.update(1)

    pbar.close()

    save_features_to_shapefile(all_polygons, projection, f"image_data/{shapefilename}_all_polygons")
    save_features_to_shapefile(all_car_polygons, projection, f"image_data/{shapefilename}_car_polygons")
    save_features_to_shapefile(all_car_centroids, projection, f"image_data/{shapefilename}_car_centroids")

    return all_car_polygons, all_car_centroids, dataset, projection


def process_geotif_single_tile(image_path, sam_checkpoint_path, shapefilename, tile_size: int = 1024):
    dataset, geotransform, projection = load_geotif(image_path)
    if dataset is None:
        return

    mask_generator = load_sam_model(sam_checkpoint_path)

    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount
    print("Image size:", width, height)
    print("Number of bands:", bands)

    tile = dataset.ReadAsArray(0, 0, tile_size, tile_size)

    if bands > 1:
        tile = np.transpose(tile, (1, 2, 0))

    sam_result = process_tile(tile, mask_generator)

    detections = sv.Detections.from_sam(sam_result=sam_result)

    polygon_geometries = []

    # for bbox in detections.xyxy:
    #     x1, y1, x2, y2 = bbox
    #     center_x = (x1 + x2) / 2
    #     center_y = (y1 + y2) / 2
    #     geo_x, geo_y = pixel_to_geo(geotransform, center_x, center_y)
    #     centroid_points.append(Point(geo_x, geo_y))

    for mask in detections.mask:
        mask_uint8 = mask.astype(np.uint8)
        # Find contours from binary mask
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) >= 3:
                polygon = [tuple(point[0]) for point in contour]
                geo_coords = convert_polygon_coordinates(polygon, geotransform)
                if len(geo_coords) >= 3:
                    polygon = Polygon(geo_coords)
                    if polygon.is_valid:
                        polygon_geometries.append(polygon)

    all_centroids = [polygon.centroid for polygon in polygon_geometries]

    car_polygons = filter_car_polygons_aerial(polygon_geometries)
    car_centroids = [polygon.centroid for polygon in car_polygons]

    save_features_to_shapefile(car_polygons, projection, f"image_data/{shapefilename}_polygons")
    save_features_to_shapefile(car_centroids, projection, f"image_data/{shapefilename}_centroids")

    return car_polygons, polygon_geometries, car_centroids, all_centroids, detections, tile, dataset, projection

def save_features_to_shapefile(features, projection, output_path):
    gdf = gpd.GeoDataFrame(geometry=features, crs=projection)
    gdf.to_file(output_path, driver="ESRI Shapefile")
    print(f"GeoDataFrame saved to {output_path}")

def filter_car_polygons_aerial(polygons, min_aspect_ratio=1, max_aspect_ratio=2.7, min_area=7, max_area=30, min_rectangularity=0.8, max_circularity=0.85):
    car_polygons = []
    for polygon in polygons:
        # Get the minimum rotated bounding rectangle
        rotated_bbox = polygon.minimum_rotated_rectangle

        # Extract the corners of the rectangle
        x, y = rotated_bbox.exterior.coords.xy
        bbox_points = list(zip(x, y))

        # Calculate the width and height of the rotated bounding box
        edge_lengths = [np.linalg.norm(np.array(bbox_points[i]) - np.array(bbox_points[i-1])) for i in range(1, 4)]
        longer_edge = max(edge_lengths)
        shorter_edge = min(edge_lengths)

        if shorter_edge == 0:  # To avoid division by zero
            continue

        aspect_ratio = longer_edge / shorter_edge
        polygon_area = polygon.area
        area = rotated_bbox.area
        rectangularity = polygon_area / area

        perimeter = polygon.length
        circularity = 4 * np.pi * polygon_area / (perimeter ** 2)

        if (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio and
            min_area <= area <= max_area and
            rectangularity >= min_rectangularity and
            circularity <= max_circularity):  # Adjust this threshold as needed
            car_polygons.append(polygon)

    return car_polygons

def visualize_detections_on_image(image_array, detections, save_path=None):
    image_rgb = preprocess_for_sam(image_array)
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated_image = mask_annotator.annotate(scene=image_rgb.copy(), detections=detections)

    sv.plot_images_grid(
        images=[image_rgb, annotated_image],
        grid_size=(1, 2),
        titles=["Original Image", "Annotated Image"],
    )

    if save_path:
        cv2.imwrite(save_path, annotated_image)
        print(f"Annotated image saved to {save_path}")

    return annotated_image

def visualize_filtered_cars(image_array, filtered_polygons):
    # Preprocess the image for display
    image_rgb = preprocess_for_sam(image_array)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Display the image
    ax.imshow(image_rgb)

    # Plot filtered polygons in green
    for poly in filtered_polygons:
        x, y = poly.exterior.xy
        ax.plot(x, y, color='green', alpha=0.8, linewidth=2, solid_capstyle='round', zorder=2)

    # Set title with current parameters
    ax.set_title(f"{len(filtered_polygons)} filtered car polygons")

    # Remove axis labels
    ax.set_axis_off()

    # Show the plot
    plt.tight_layout()
    plt.show()

def compare_geotif_outputs(
    polygons1: List[Polygon],
    centroids1: List[Point],
    polygons2: List[Polygon],
    centroids2: List[Point],
    distance_threshold: float = 1.0
) -> dict:
    """
    Compare two sets of polygon geometries and centroids from process_geotif_single_tile outputs.

    :param polygons1: List of Shapely Polygons from the first image
    :param centroids1: List of Shapely Points (centroids) from the first image
    :param polygons2: List of Shapely Polygons from the second image
    :param centroids2: List of Shapely Points (centroids) from the second image
    :param distance_threshold: Maximum distance to consider centroids as matching
    :return: Dictionary containing matched and unmatched polygons and centroids
    """
    # Convert centroids to numpy arrays for KDTree
    centroids1_array = np.array([(c.x, c.y) for c in centroids1])
    centroids2_array = np.array([(c.x, c.y) for c in centroids2])

    # Build KDTree for faster nearest neighbor search
    tree = cKDTree(centroids2_array)

    matched_indices = []
    unmatched_indices1 = []
    unmatched_indices2 = set(range(len(centroids2)))

    for i, centroid in enumerate(centroids1_array):
        distance, j = tree.query(centroid, k=1)
        if distance < distance_threshold:
            matched_indices.append((i, j))
            if j in unmatched_indices2:
                unmatched_indices2.remove(j)
        else:
            unmatched_indices1.append(i)

    # Prepare the results
    matched_polygons_1 = [polygons1[i] for i, _ in matched_indices]
    matched_polygons_2 = [polygons2[j] for _, j in matched_indices]
    matched_centroids_1 = [centroids1[i] for i, _ in matched_indices]
    matched_centroids_2 = [centroids2[j] for _, j in matched_indices]

    unmatched_polygons1 = [polygons1[i] for i in unmatched_indices1]
    unmatched_centroids1 = [centroids1[i] for i in unmatched_indices1]

    unmatched_polygons2 = [polygons2[i] for i in unmatched_indices2]
    unmatched_centroids2 = [centroids2[i] for i in unmatched_indices2]

    return {
        'matched_polygons_1': matched_polygons_1,
        'matched_centroids_1': matched_centroids_1,
        'matched_polygons_2': matched_polygons_2,
        'matched_centroids_2': matched_centroids_2,
        'unmatched_polygons1': unmatched_polygons1,
        'unmatched_centroids1': unmatched_centroids1,
        'unmatched_polygons2': unmatched_polygons2,
        'unmatched_centroids2': unmatched_centroids2
    }

# def create_df_from_detections(detections, geotransform, projection):
#     polygon_geometries = []

#     for mask in detections.mask:
#         mask_uint8 = mask.astype(np.uint8)

#         contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for contour in contours:
#             if len(contour) >= 3:
#                 polygon = [tuple(point[0]) for point in contour]
#                 geo_coords = convert_polygon_coordinates(polygon, geotransform)
#                 if len(geo_coords) >= 3:
#                     polygon = Polygon(geo_coords)
#                     if polygon.is_valid:
#                         polygon_geometries.append(polygon)

#     car_polygons = filter_car_polygons_aerial(polygon_geometries)

#     polygons_gdf = gpd.GeoDataFrame(geometry=polygon_geometries, crs=projection)
#     car_polygons_gdf = gpd.GeoDataFrame(geometry=car_polygons, crs=projection)

#     return polygons_gdf, car_polygons_gdf

# def save_annotated_geotiff_with_detections(
#     output_path: str,
#     original_dataset: gdal.Dataset,
#     combined_detections: sv.Detections,
#     annotated_image: np.ndarray
# ):
#     # Get geospatial information from the original dataset
#     geotransform = original_dataset.GetGeoTransform()
#     projection = original_dataset.GetProjection()

#     # Create a new GeoTIFF file
#     driver = gdal.GetDriverByName("GTiff")
#     out_dataset = driver.Create(
#         output_path,
#         original_dataset.RasterXSize,
#         original_dataset.RasterYSize,
#         3,  # 3 bands for RGB
#         gdal.GDT_Byte
#     )

#     # Set the geotransform and projection
#     out_dataset.SetGeoTransform(geotransform)
#     out_dataset.SetProjection(projection)

#     # Write the annotated image data
#     for i in range(3):  # RGB channels
#         out_dataset.GetRasterBand(i + 1).WriteArray(annotated_image[:, :, i])

#     # Create a vector layer for detections
#     mem_driver = ogr.GetDriverByName('Memory')
#     mem_ds = mem_driver.CreateDataSource('memory')
#     mem_layer = mem_ds.CreateLayer('detections', geom_type=ogr.wkbPolygon)

#     # Add fields for attributes
#     id_field = ogr.FieldDefn('id', ogr.OFTInteger)
#     mem_layer.CreateField(id_field)
#     conf_field = ogr.FieldDefn('confidence', ogr.OFTReal)
#     mem_layer.CreateField(conf_field)

#     # Create detections as polygons
#     for i, bbox in enumerate(combined_detections.xyxy):
#         x1, y1, x2, y2 = map(float, bbox)
#         ring = ogr.Geometry(ogr.wkbLinearRing)
#         ring.AddPoint(x1, y1)
#         ring.AddPoint(x2, y1)
#         ring.AddPoint(x2, y2)
#         ring.AddPoint(x1, y2)
#         ring.AddPoint(x1, y1)  # Close the polygon

#         poly = ogr.Geometry(ogr.wkbPolygon)
#         poly.AddGeometry(ring)

#         feature = ogr.Feature(mem_layer.GetLayerDefn())
#         feature.SetGeometry(poly)
#         feature.SetField('id', i)
#         # feature.SetField('confidence', float(combined_detections.confidence[i]))
#         mem_layer.CreateFeature(feature)

#     # Create a vector layer in the GeoTIFF
#     out_dataset.CreateLayer('detections', geom_type=ogr.wkbPolygon)

#     # Copy the memory layer to the GeoTIFF
#     gdal.VectorTranslate(out_dataset, mem_ds)

#     # Close datasets
#     out_dataset = None
#     mem_ds = None

#     print(f"Annotated GeoTIFF with detections saved to {output_path}")

