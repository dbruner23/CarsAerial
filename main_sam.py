import os
import torch
import cv2
import supervision as sv
from osgeo import gdal, osr

from utils import (
    process_geotif_single_tile,
    visualize_detections_on_image,
)

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

HOME = os.getcwd()
CHECKPOINT_PATH = os.path.join(HOME, "sam_weights", "sam_vit_h_4b8939.pth")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam)

IMAGE_PATH = "image_data/auckland-0075m-urban-aerial-photos-2015-2016.tif"
OUTPUT_PATH = "image_data/auckland_aerial_annotated.tif"

# combined_detections, geo_located_detections, image_array = process_geotif_with_sam_tiled(image_path=IMAGE_PATH, sam_checkpoint_path=CHECKPOINT_PATH)

detections, centroid_gdf, polygons_gdf, tile, dataset, polygon_geometries, projection = process_geotif_single_tile(IMAGE_PATH, CHECKPOINT_PATH)

# Visualize detections on the image
annotated_image = visualize_detections_on_image(image_array, detections, save_path="image_data/detections.jpg")








