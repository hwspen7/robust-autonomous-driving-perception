# BDD100K二维目标检测使用的原始类别。
# 保持类别名称和顺序固定，后续COCO转换、MMDetection、
# YOLO配置和分类别AP统计都必须使用同一套类别定义。

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

# 已知道路异物类别在后续阶段单独构建数据集，
# 当前阶段只保留定义，不与BDD100K原始类别混合。
KNOWN_OBSTACLE_CLASSES = (
    "box",
    "tire",
    "road_debris",
    "fallen_object",
    "construction_material",
    "traffic_cone",
    "barrier"
)

# 这些类别不参与训练，只用于后续开放集测试。
# 如果模型训练时已经看过这些类别，它们就不能再被称为未知类别。
UNKNOWN_OBSTACLE_EVAL_CLASSES = (
    "tree_branch",
    "rock",
    "animal",
    "vehicle_part",
    "suitcase",
    "unusual_cargo"
)

# 可行驶区域属于像素级语义分割任务，
# 不能直接混入目标检测类别。
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