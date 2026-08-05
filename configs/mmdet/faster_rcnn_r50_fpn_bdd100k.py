"""
BDD100K Faster R-CNN R50-FPN Baseline配置。

数据：
- COCO格式BDD100K；
- 70,000张训练图片；
- 10,000张验证图片；
- 10类矩形目标。

任务：
- 作为经典two-stage CNN检测器；
- 与YOLO11和RT-DETR进行精度、速度和鲁棒性比较。
"""

_base_ = "mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py"

classes = (
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

metainfo = {
    "classes": classes,
}

data_root = "datasets/coco/bdd100k/"

# Faster R-CNN的分类头必须从COCO的80类改为BDD100K的10类
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=len(classes),
        )
    )
)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type="DefaultSampler",
        shuffle=True,
    ),
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="annotations/instances_train.json",
        # COCO JSON里的file_name已经是images/train/xxx.jpg，
        # 因此这里不能再次添加images/train前缀。
        data_prefix=dict(img=""),
        metainfo=metainfo,
        filter_cfg=dict(
            filter_empty_gt=False,
            min_size=1
        )
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(
        type="DefaultSampler",
        shuffle=False,
    ),
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="annotations/instances_val.json",
        data_prefix=dict(img=""),
        metainfo=metainfo,
        test_mode=True,
    )
)
test_dataloader = val_dataloader
val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/instances_val.json",
    metric="bbox",
    classwise=True,
    format_only=False,
)

test_evaluator = val_evaluator

# MMDetection的1x训练策略为12个epoch。
train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=12,
    val_interval=1,
)

val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# 原始COCO配置常按较大的总batch设置学习率。
# 单GPU、batch=2时使用更保守的学习率。
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="SGD",
        lr=0.0025,
        momentum=0.9,
        weight_decay=0.0001,
    ),
    clip_grad=dict(
        max_norm=35,
        norm_type=2,
    )
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type="MultiStepLR",
        by_epoch=True,
        begin=0,
        end=12,
        milestones=[8, 11],
        gamma=0.1,
    )
]

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        save_best="coco/bbox_mAP",
        rule="greater",
        max_keep_ckpts=3,
    ),
    logger=dict(
        type="LoggerHook",
        interval=50,
    )
)

randomness = dict(
    seed=42,
    deterministic=True,
)

work_dir = "results/baselines/faster_rcnn/faster_rcnn_r50_fpn"
