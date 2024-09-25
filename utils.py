import numpy as np
import supervision as sv
import cv2
import matplotlib.pyplot as plt

from osgeo import gdal, ogr
import geopandas as gpd
from shapely.geometry import Point, Polygon
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from typing import List, Tuple

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

def load_sam_model(checkpoint_path):
    sam = sam_model_registry["vit_h"](checkpoint_path)
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

def process_geotif_with_sam_tiled(image_path, sam_checkpoint_path, tile_size: int = 1024, overlap: int = 100):
    dataset, geotransform, projection = load_geotif(image_path)
    if dataset is None:
        return

    mask_generator = load_sam_model(sam_checkpoint_path)

    width = dataset.RasterXSize
    height = dataset.RasterYSize
    bands = dataset.RasterCount
    print("Image size:", width, height)
    print("Number of bands:", bands)

    all_detections = []
    all_geo_located_detections = []

    full_image = dataset.ReadAsArray()

    if bands > 1:
        full_image = np.transpose(full_image, (1, 2, 0))
    full_image_rgb = preprocess_for_sam(full_image)

    for y in range(0, height, tile_size - overlap):
        for x in range(0, width, tile_size - overlap):
            tile_width = min(tile_size, width - x)
            tile_height = min(tile_size, height - y)
            tile = dataset.ReadAsArray(x, y, tile_width, tile_height)

            if bands > 1:
                tile = np.transpose(tile, (1, 2, 0))

            sam_result = process_tile(tile, mask_generator)

            detections = sv.Detections.from_sam(sam_result=sam_result)

            detections.xyxy[:, [0, 2]] += x
            detections.xyxy[:, [1, 3]] += y

            all_detections.append(detections)

            for bbox in detections.xyxy:
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                geo_x, geo_y = pixel_to_geo(geotransform, center_x, center_y)
                all_geo_located_detections.append((geo_x, geo_y))

    combined_detections = sv.Detections.merge(all_detections)

    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated_image = mask_annotator.annotate(scene=full_image_rgb.copy(), detections=combined_detections)

    return combined_detections, all_geo_located_detections, dataset, annotated_image

def process_geotif_single_tile(image_path, sam_checkpoint_path, tile_size: int = 1024):
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

    centroid_points = []
    polygon_geometries = []

    for bbox in detections.xyxy:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        geo_x, geo_y = pixel_to_geo(geotransform, center_x, center_y)
        centroid_points.append(Point(geo_x, geo_y))

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

    centroids_gdf = gpd.GeoDataFrame(geometry=centroid_points, crs=projection)
    polygons_gdf = gpd.GeoDataFrame(geometry=polygon_geometries, crs=projection)

    return detections, centroids_gdf, polygons_gdf, tile, dataset, polygon_geometries, projection

def save_features_to_shapefile(features, projection, output_path):
    gdf = gpd.GeoDataFrame(geometry=features, crs=projection)
    gdf.to_file(output_path, driver="ESRI Shapefile")
    print(f"GeoDataFrame saved to {output_path}")

def filter_car_polygons_aerial(polygons, min_aspect_ratio=1.5, max_aspect_ratio=2.2, min_area=7, max_area=20, min_rectangularity=0.75):
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

        if (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio and
            min_area <= area <= max_area and
            rectangularity >= min_rectangularity):  # Adjust this threshold as needed
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

def visualize_filtered_cars(image_array, all_polygons, filtered_polygons, min_aspect_ratio, max_aspect_ratio, min_area, max_area):
    # Preprocess the image for display
    image_rgb = preprocess_for_sam(image_array)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Display the image
    ax.imshow(image_rgb)

    # Plot all polygons in red
    for poly in all_polygons:
        x, y = poly.exterior.xy
        ax.plot(x, y, color='red', alpha=0.6, linewidth=1, solid_capstyle='round', zorder=1)

    # Plot filtered polygons in green
    for poly in filtered_polygons:
        x, y = poly.exterior.xy
        ax.plot(x, y, color='green', alpha=0.8, linewidth=2, solid_capstyle='round', zorder=2)

    # Set title with current parameters
    ax.set_title(f"Filtered Cars\nAspect Ratio: {min_aspect_ratio}-{max_aspect_ratio}, Area: {min_area}-{max_area}")

    # Remove axis labels
    ax.set_axis_off()

    # Show the plot
    plt.tight_layout()
    plt.show()

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

