import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import supervision as sv
from osgeo import gdal, osr
from scipy.spatial import cKDTree
from typing import List, Tuple
from shapely.geometry import Point, Polygon

from utils import (
    process_geotif_single_tile,
    process_geotif_with_sam_tiled,
    filter_car_polygons_aerial,
    save_features_to_shapefile,
    visualize_filtered_cars,
    visualize_detections_on_image,
    compare_geotif_outputs,
)

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

HOME = os.getcwd()
CHECKPOINT_PATH = os.path.join(HOME, "sam_weights", "sam_vit_h_4b8939.pth")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam)

IMAGE_PATH = "image_data/auckland-0075m-urban-aerial-photos-2017.tif"
OUTPUT_PATH = "image_data/auckland_aerial_annotated.tif"

car_polygons, polygon_geometries, car_centroids, all_centroids, detections, tile, dataset, projection = process_geotif_single_tile(IMAGE_PATH, CHECKPOINT_PATH, "akl_2017")

# Visualize detections on the image
annotated_image = visualize_detections_on_image(image_array, detections, save_path="image_data/detections.jpg")








