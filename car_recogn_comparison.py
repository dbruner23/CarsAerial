import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
from shapely.geometry import box
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import cKDTree

def load_bounding_boxes(file_path):
    return gpd.read_file(file_path)

def find_close_boxes(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, distance_threshold: float = 1.0) -> gpd.GeoDataFrame:
    """
    Find close pairs of bounding boxes between two GeoDataFrames using their centroids.

    :param gdf1: GeoDataFrame containing bounding boxes from the first image
    :param gdf2: GeoDataFrame containing bounding boxes from the second image
    :param distance_threshold: Maximum distance to consider centroids as matching
    :return: GeoDataFrame containing matched pairs of bounding boxes
    """
    # Calculate centroids
    gdf1['centroid'] = gdf1.geometry.centroid
    gdf2['centroid'] = gdf2.geometry.centroid

    # Convert centroids to numpy arrays for KDTree
    centroids1_array = np.array([(c.x, c.y) for c in gdf1['centroid']])
    centroids2_array = np.array([(c.x, c.y) for c in gdf2['centroid']])

    # Build KDTree for faster nearest neighbor search
    tree = cKDTree(centroids2_array)

    # Find nearest neighbors
    distances, indices = tree.query(centroids1_array, k=1, distance_upper_bound=distance_threshold)

    # Create a list of matched pairs
    matched_pairs = [(i, j) for i, (d, j) in enumerate(zip(distances, indices)) if d <= distance_threshold]

    # Create a new GeoDataFrame with matched pairs
    if matched_pairs:
        matched_df = gpd.GeoDataFrame({
            'index_1': [pair[0] for pair in matched_pairs],
            'index_2': [pair[1] for pair in matched_pairs],
            'geometry_1': gdf1.loc[[pair[0] for pair in matched_pairs], 'geometry'].values,
            'geometry_2': gdf2.loc[[pair[1] for pair in matched_pairs], 'geometry'].values,
            'centroid_1': gdf1.loc[[pair[0] for pair in matched_pairs], 'centroid'].values,
            'centroid_2': gdf2.loc[[pair[1] for pair in matched_pairs], 'centroid'].values,
            'distance': [distances[pair[0]] for pair in matched_pairs]
        })

        # Set the geometry to the centroid of the first image for spatial operations
        matched_df.set_geometry('centroid_1', inplace=True)

        # Set the CRS to match the input GeoDataFrames
        matched_df.set_crs(gdf1.crs, inplace=True)
    else:
        # If no matches found, return an empty GeoDataFrame with the correct structure
        matched_df = gpd.GeoDataFrame(columns=['index_1', 'index_2', 'geometry_1', 'geometry_2', 'centroid_1', 'centroid_2', 'distance'], geometry='centroid_1', crs=gdf1.crs)

    return matched_df

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

def compare_detections_main(raster_path1, raster_path2, boxes_path1, boxes_path2, high_similarity_threshold=0.9):
    boxes1 = gpd.read_file(boxes_path1)
    boxes2 = gpd.read_file(boxes_path2)

    close_pairs = find_close_boxes(boxes1, boxes2)

    results = []

    for i, row in close_pairs.iterrows():
        geometry1 = row['geometry_1']
        geometry2 = row['geometry_2']

        chip1 = extract_image_chip(raster_path1, geometry1)
        chip2 = extract_image_chip(raster_path2, geometry2)

        similarity = compare_image_chips(chip1, chip2)

        results.append((similarity, chip1, chip2))

    high_similarity_results = [(similarity, chip1, chip2) for similarity, chip1, chip2 in results if similarity > high_similarity_threshold]

    return results, high_similarity_results

def compare_detections_main_with_geoms(raster_path1, raster_path2, boxes_path1, boxes_path2, high_similarity_threshold=0.9):
    boxes1 = gpd.read_file(boxes_path1)
    boxes2 = gpd.read_file(boxes_path2)

    close_pairs = find_close_boxes(boxes1, boxes2)

    results = []

    for i, row in close_pairs.iterrows():
        geometry1 = row['geometry_1']
        geometry2 = row['geometry_2']

        chip1 = extract_image_chip(raster_path1, geometry1)
        chip2 = extract_image_chip(raster_path2, geometry2)

        similarity = compare_image_chips(chip1, chip2)

        results.append((similarity, chip1, chip2, geometry1, geometry2))

    high_similarity_results = []
    high_similarity_geometries1 = []
    high_similarity_geometries2 = []
    for similarity, chip1, chip2, geom1, geom2 in results:
        if similarity > high_similarity_threshold:
            high_similarity_results.append((similarity, chip1, chip2))
            high_similarity_geometries1.append(geom1)
            high_similarity_geometries2.append(geom2)

    return results, high_similarity_results, high_similarity_geometries1, high_similarity_geometries2