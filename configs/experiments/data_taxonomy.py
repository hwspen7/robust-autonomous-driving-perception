# Original class set used for BDD100K 2D object detection
# Keep class names and ordering fixed so COCO conversion MMDetection
# YOLO configs and per class AP statistics use the same definitions

SOURCE_TO_CANONICAL = {
    "person": "pedestrian",
    "rider": "rider",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "motor": "motorcycle",
    "bike": "bicycle",
    "traffic light": "traffic light",
    "traffic sign": "traffic sign",
}

DETECTION_CLASSES = (
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
)

# Known road debris classes will be collected into a separate dataset later
# At this stage keep only definitions and do not mix them with original BDD100K classes
KNOWN_OBSTACLE_CLASSES = (
    "box",
    "tire",
    "road_debris",
    "fallen_object",
    "construction_material",
    "traffic_cone",
    "barrier"
)

# These classes are excluded from training and reserved for later open set evaluation
# If a model sees these classes during training they are no longer considered unknown
UNKNOWN_OBSTACLE_EVAL_CLASSES = (
    "tree_branch",
    "rock",
    "animal",
    "vehicle_part",
    "suitcase",
    "unusual_cargo"
)

# Drivable area is a pixel level semantic segmentation task
# It should not be mixed into object detection classes
DRIVABLE_AREA_CLASSES = (
    "background",
    "ego_drivable_area",
    "other_drivable_area",
)

YOLO_CLASS_TO_ID = {
    class_name: class_id
    for class_id, class_name in enumerate(DETECTION_CLASSES)
}

COCO_CLASS_TO_ID = {
    class_name: class_id
    for class_id, class_name in enumerate(DETECTION_CLASSES, start=1)
}

STRUCTURE_CLASSES = (
    "lane",
    "drivable area",
)