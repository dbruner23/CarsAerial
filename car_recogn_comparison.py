import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
from shapely.geometry import box
from sklearn.metrics.pairwise import cosine_similarity

def load_bounding_boxes(file_path):
    return gpd.read_file(file_path)

def find_close_boxes(gdf1, gdf2, distance_threshold=1.0):
    # Calculate centroids for both GeoDataFrames
    gdf1_centroids = gdf1.copy()
    gdf2_centroids = gdf2.copy()

    gdf1_centroids['geometry'] = gdf1_centroids.geometry.centroid
    gdf2_centroids['geometry'] = gdf2_centroids.geometry.centroid

    # Perform spatial join on centroids
    close_pairs = gpd.sjoin_nearest(
        gdf1_centroids,
        gdf2_centroids,
        max_distance=distance_threshold,
        distance_col="distance"
    )

    # Add original geometries back
    close_pairs['geometry_1'] = gdf1.loc[close_pairs.index, 'geometry']
    close_pairs['geometry_2'] = gdf2.loc[close_pairs['index_right'], 'geometry']

    return close_pairs

def extract_image_chip(raster_path, geometry):
    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, [geometry], crop=True)
    return out_image

def compare_image_chips(chip1, chip2):

    if chip1.shape != chip2.shape:
        chip2 = np.resize(chip2, chip1.shape)

    flat1 = chip1.flatten()
    flat2 = chip2.flatten()
    similarity = cosine_similarity(flat1.reshape(1, -1), flat2.reshape(1, -1))[0][0]

    return similarity

def compare_detections_main(raster_path1, raster_path2, boxes_path1, boxes_path2):
    boxes1 = load_bounding_boxes(boxes_path1)
    boxes2 = load_bounding_boxes(boxes_path2)

    close_pairs = find_close_boxes(boxes1, boxes2)

    results = []

    for i, row in close_pairs.iterrows():
        geometry1 = row['geometry']
        geometry2 = boxes2.loc[row['index_right']]['geometry']

        chip1 = extract_image_chip(raster_path1, geometry1)
        chip2 = extract_image_chip(raster_path2, geometry2)

        similarity = compare_image_chips(chip1, chip2)

        results.append({
            'box1_id': i,
            'box2_id': row['index_right'],
            'similarity': similarity,
        })

    return results