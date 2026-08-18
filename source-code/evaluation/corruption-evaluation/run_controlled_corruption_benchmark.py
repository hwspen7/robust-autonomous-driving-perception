from __future__ import annotations 

import argparse 
import csv 
import hashlib 
import json 
import os 
import platform 
import shutil 
import subprocess 
import sys 
import time 
from datetime import datetime ,timezone 
from pathlib import Path 
from typing import Any 

import numpy as np 
from pycocotools .coco import COCO 
from pycocotools .cocoeval import COCOeval 






PROJECT_ROOT =Path (
"/root/rivermind-data/autodrive/code/robust-autonomous-driving-perception"
)

YOLO_EXPORTER =(
PROJECT_ROOT /"scripts/export_yolo11m_controlled_corruptions.py"
)

DFINE_EXPORTER =(
PROJECT_ROOT /"scripts/export_dfine_m_controlled_corruptions.py"
)

CONTROLLED_MODULE =(
PROJECT_ROOT /"scripts/controlled_corruptions.py"
)

CORRUPTION_PROTOCOL =Path (
"/root/rivermind-data/autodrive/results/evaluation/"
"controlled_corruption/corruption_protocol.json"
)

YOLO_MODEL =Path (
"/root/rivermind-data/autodrive/results/detection/"
"yolo11m_bdd100k_main/weights/best.pt"
)

DFINE_ROOT =PROJECT_ROOT /"third_party/D-FINE"

DFINE_CONFIG =(
DFINE_ROOT /"configs/dfine/dfine_hgnetv2_m_bdd100k.yml"
)

DFINE_CHECKPOINT =Path (
"/root/rivermind-data/autodrive/results/final/"
"dfine_m_bdd100k_60epoch_final/checkpoints/"
"dfine_m_bdd100k_best.pth"
)

DFINE_ENV =Path (
"/root/rivermind-data/autodrive/conda_envs/dfine"
)

DATASET_ROOT =Path (
"/root/rivermind-data/autodrive/datasets/datasets/bdd100k_final"
)

GT_PATH =(
DATASET_ROOT /"coco/annotations/instances_val.json"
)

DEFAULT_OUTPUT_ROOT =Path (
"/root/rivermind-data/autodrive/results/evaluation/"
"controlled_corruption/benchmark"
)

EXPECTED_IMAGES =10_000 
EXPECTED_ANNOTATIONS =185_523 

BDD_CATEGORIES ={
1 :"pedestrian",
2 :"rider",
3 :"car",
4 :"truck",
5 :"bus",
6 :"train",
7 :"motorcycle",
8 :"bicycle",
9 :"traffic light",
10 :"traffic sign",
}

VARIANTS =[
("low_light",1 ),("low_light",2 ),("low_light",3 ),
("fog",1 ),("fog",2 ),("fog",3 ),
("rain",1 ),("rain",2 ),("rain",3 ),
("blur",1 ),("blur",2 ),("blur",3 ),
("noise",1 ),("noise",2 ),("noise",3 ),
]

MODEL_KEYS ={
"yolo":"YOLO11m",
"dfine":"D-FINE-M",
}


CLEAN ={
"YOLO11m":{
"AP":0.3161 ,
"AP50":0.5543 ,
"AP75":0.2993 ,
"AP_small":0.1277 ,
"AP_medium":0.3748 ,
"AP_large":0.5686 ,
},
"D-FINE-M":{
"AP":0.3457 ,
"AP50":0.5976 ,
"AP75":0.3313 ,
"AP_small":0.1479 ,
"AP_medium":0.4030 ,
"AP_large":0.6111 ,
},
}

AP_FIELDS =[
"AP",
"AP50",
"AP75",
"AP_small",
"AP_medium",
"AP_large",
]

METRIC_FIELDS =[
"model","variant","corruption","severity",
"num_images","num_gt","num_predictions",
"AP","AP50","AP75",
"AP_small","AP_medium","AP_large",
"AR_1","AR_10","AR_100",
"AR_small","AR_medium","AR_large",
]

for field in AP_FIELDS :
    METRIC_FIELDS +=[
    f"{field }_absolute_drop",
    f"{field }_relative_drop",
    f"{field }_retention",
    ]

METRIC_FIELDS +=[
"prediction_sha256",
"inference_seconds",
"evaluation_seconds",
"completed_at_utc",
]

CLASS_FIELDS =[
"model","variant","corruption","severity",
"category_id","category_name","num_gt",
"AP","AP50","AP75",
]






def now_utc ()->str :
    return datetime .now (timezone .utc ).isoformat ()


def parse_args ()->argparse .Namespace :
    parser =argparse .ArgumentParser ()

    parser .add_argument (
    "--output-root",
    type =Path ,
    default =DEFAULT_OUTPUT_ROOT ,
    )

    parser .add_argument (
    "--only",
    nargs ="*",
    default =None ,
    help ="Example: --only fog_s1 fog_s2 fog_s3",
    )

    parser .add_argument (
    "--models",
    nargs ="+",
    choices =["yolo","dfine"],
    default =["yolo","dfine"],
    )

    parser .add_argument (
    "--yolo-batch",
    type =int ,
    default =16 ,
    )

    parser .add_argument (
    "--keep-temp",
    action ="store_true",
    )

    parser .add_argument (
    "--force-inference",
    action ="store_true",
    )

    return parser .parse_args ()


def vname (corruption :str ,severity :int )->str :
    return f"{corruption }_s{severity }"


def sha256_file (path :Path )->str :
    digest =hashlib .sha256 ()

    with path .open ("rb")as f :
        for block in iter (
        lambda :f .read (1024 *1024 ),
        b"",
        ):
            digest .update (block )

    return digest .hexdigest ()


def require_file (path :Path ,label :str )->None :
    if not path .is_file ():
        raise FileNotFoundError (f"{label }:\n{path }")


def require_dir (path :Path ,label :str )->None :
    if not path .is_dir ():
        raise FileNotFoundError (f"{label }:\n{path }")


def load_json (path :Path )->Any :
    with path .open ("r",encoding ="utf-8")as f :
        return json .load (f )


def atomic_json (path :Path ,data :Any )->None :
    path .parent .mkdir (parents =True ,exist_ok =True )

    tmp =path .with_suffix (path .suffix +".tmp")

    tmp .write_text (
    json .dumps (
    data ,
    indent =2 ,
    ensure_ascii =False ,
    allow_nan =False ,
    ),
    encoding ="utf-8",
    )

    os .replace (tmp ,path )


def atomic_csv (
path :Path ,
rows :list [dict [str ,Any ]],
fields :list [str ],
)->None :
    path .parent .mkdir (parents =True ,exist_ok =True )

    tmp =path .with_suffix (path .suffix +".tmp")

    with tmp .open (
    "w",
    encoding ="utf-8",
    newline ="",
    )as f :
        writer =csv .DictWriter (
        f ,
        fieldnames =fields ,
        extrasaction ="ignore",
        )
        writer .writeheader ()
        writer .writerows (rows )

    os .replace (tmp ,path )


def run_command (command :list [str ],label :str )->None :
    print ()
    print ("="*120 )
    print (label )
    print ("="*120 )

    subprocess .run (
    command ,
    cwd =str (PROJECT_ROOT ),
    check =True ,
    )


def choose_variants (
only :list [str ]|None ,
)->list [tuple [str ,int ]]:
    if not only :
        return list (VARIANTS )

    mapping ={
    vname (c ,s ):(c ,s )
    for c ,s in VARIANTS 
    }

    bad =[x for x in only if x not in mapping ]

    if bad :
        raise ValueError (f"Unknown variants: {bad }")

    return [mapping [x ]for x in only ]


def load_existing (path :Path )->list [dict [str ,Any ]]:
    if not path .is_file ():
        return []

    data =load_json (path )

    if not isinstance (data ,list ):
        raise TypeError (f"Expected list JSON: {path }")

    return data 


def replace_rows (
rows :list [dict [str ,Any ]],
model :str ,
variant :str ,
new_rows :list [dict [str ,Any ]],
)->list [dict [str ,Any ]]:
    rows =[
    row 
    for row in rows 
    if not (
    str (row ["model"])==model 
    and str (row ["variant"])==variant 
    )
    ]

    rows .extend (new_rows )
    return rows 


def has_metric (
rows :list [dict [str ,Any ]],
model :str ,
variant :str ,
)->bool :
    return any (
    str (row ["model"])==model 
    and str (row ["variant"])==variant 
    for row in rows 
    )






def validate_gt ()->tuple [
list [int ],
set [int ],
dict [int ,int ],
]:
    gt =load_json (GT_PATH )

    if len (gt ["images"])!=EXPECTED_IMAGES :
        raise RuntimeError (
        f"GT image count != {EXPECTED_IMAGES }"
        )

    if len (gt ["annotations"])!=EXPECTED_ANNOTATIONS :
        raise RuntimeError (
        f"GT annotation count != {EXPECTED_ANNOTATIONS }"
        )

    actual ={
    int (c ["id"]):str (c ["name"]).strip ().lower ()
    for c in gt ["categories"]
    }

    if actual !=BDD_CATEGORIES :
        raise RuntimeError (
        "BDD100K original 1~10 category mapping mismatch."
        )

    ids =sorted (int (x ["id"])for x in gt ["images"])

    counts ={
    category_id :0 
    for category_id in BDD_CATEGORIES 
    }

    for ann in gt ["annotations"]:
        counts [int (ann ["category_id"])]+=1 

    return ids ,set (ids ),counts 


def validate_summary (
path :Path ,
expected_ids :set [int ],
)->None :
    summary =load_json (path )

    if len (summary )!=EXPECTED_IMAGES :
        raise RuntimeError (
        f"image_summary count mismatch: {len (summary )}"
        )

    actual_ids ={
    int (item ["image_id"])
    for item in summary 
    }

    if actual_ids !=expected_ids :
        raise RuntimeError (
        "image_summary image IDs do not equal official val IDs."
        )






def safe_mean (values :np .ndarray )->float :
    valid =values [values >-1 ]

    if valid .size ==0 :
        return -1.0 

    return float (valid .mean ())


def evaluate (
*,
model :str ,
variant :str ,
corruption :str ,
severity :int ,
prediction_path :Path ,
gt_ids :list [int ],
gt_count_by_class :dict [int ,int ],
inference_seconds :float ,
)->tuple [
dict [str ,Any ],
list [dict [str ,Any ]],
]:
    eval_start =time .time ()

    print ()
    print (
    f"COCOeval | {model } | {variant }"
    )

    coco_gt =COCO (str (GT_PATH ))
    coco_dt =coco_gt .loadRes (str (prediction_path ))

    evaluator =COCOeval (
    coco_gt ,
    coco_dt ,
    "bbox",
    )

    evaluator .params .imgIds =gt_ids 

    evaluator .evaluate ()
    evaluator .accumulate ()
    evaluator .summarize ()

    stats =[float (x )for x in evaluator .stats ]

    row ={
    "model":model ,
    "variant":variant ,
    "corruption":corruption ,
    "severity":severity ,
    "num_images":EXPECTED_IMAGES ,
    "num_gt":EXPECTED_ANNOTATIONS ,
    "num_predictions":len (
    coco_dt .dataset .get (
    "annotations",
    [],
    )
    ),
    "AP":stats [0 ],
    "AP50":stats [1 ],
    "AP75":stats [2 ],
    "AP_small":stats [3 ],
    "AP_medium":stats [4 ],
    "AP_large":stats [5 ],
    "AR_1":stats [6 ],
    "AR_10":stats [7 ],
    "AR_100":stats [8 ],
    "AR_small":stats [9 ],
    "AR_medium":stats [10 ],
    "AR_large":stats [11 ],
    "prediction_sha256":sha256_file (
    prediction_path 
    ),
    "inference_seconds":inference_seconds ,
    "evaluation_seconds":round (
    time .time ()-eval_start ,
    3 ,
    ),
    "completed_at_utc":now_utc (),
    }

    clean =CLEAN [model ]

    for field in AP_FIELDS :
        current =float (row [field ])
        base =float (clean [field ])

        absolute_drop =base -current 

        row [f"{field }_absolute_drop"]=absolute_drop 
        row [f"{field }_relative_drop"]=(
        absolute_drop /base 
        if base >0 
        else -1.0 
        )
        row [f"{field }_retention"]=(
        current /base 
        if base >0 
        else -1.0 
        )

    precision =evaluator .eval ["precision"]

    class_rows =[]

    for k ,category_id in enumerate (
    evaluator .params .catIds 
    ):
        class_rows .append (
        {
        "model":model ,
        "variant":variant ,
        "corruption":corruption ,
        "severity":severity ,
        "category_id":int (category_id ),
        "category_name":BDD_CATEGORIES [
        int (category_id )
        ],
        "num_gt":gt_count_by_class [
        int (category_id )
        ],
        "AP":safe_mean (
        precision [:,:,k ,0 ,-1 ]
        ),
        "AP50":safe_mean (
        precision [0 ,:,k ,0 ,-1 ]
        ),
        "AP75":safe_mean (
        precision [5 ,:,k ,0 ,-1 ]
        ),
        }
        )

    del evaluator 
    del coco_dt 
    del coco_gt 

    return row ,class_rows 






def save_outputs (
output_root :Path ,
metrics :list [dict [str ,Any ]],
per_class :list [dict [str ,Any ]],
progress :dict [str ,Any ],
)->None :
    order ={
    "low_light":0 ,
    "fog":1 ,
    "rain":2 ,
    "blur":3 ,
    "noise":4 ,
    }

    metrics =sorted (
    metrics ,
    key =lambda x :(
    order [x ["corruption"]],
    int (x ["severity"]),
    x ["model"],
    ),
    )

    per_class =sorted (
    per_class ,
    key =lambda x :(
    order [x ["corruption"]],
    int (x ["severity"]),
    x ["model"],
    int (x ["category_id"]),
    ),
    )

    atomic_csv (
    output_root /"corruption_metrics.csv",
    metrics ,
    METRIC_FIELDS ,
    )

    atomic_json (
    output_root /"corruption_metrics.json",
    metrics ,
    )

    atomic_csv (
    output_root /"corruption_per_class_metrics.csv",
    per_class ,
    CLASS_FIELDS ,
    )

    atomic_json (
    output_root /"corruption_per_class_metrics.json",
    per_class ,
    )

    by_variant :dict [
    str ,
    dict [str ,dict [str ,Any ]]
    ]={}

    for row in metrics :
        by_variant .setdefault (
        row ["variant"],
        {},
        )[row ["model"]]=row 

    comparison =[]

    for variant ,pair in by_variant .items ():
        if (
        "YOLO11m"not in pair 
        or "D-FINE-M"not in pair 
        ):
            continue 

        y =pair ["YOLO11m"]
        d =pair ["D-FINE-M"]

        comparison .append (
        {
        "variant":variant ,
        "corruption":y ["corruption"],
        "severity":int (y ["severity"]),
        "yolo_AP":y ["AP"],
        "dfine_AP":d ["AP"],
        "dfine_minus_yolo_AP":(
        float (d ["AP"])-float (y ["AP"])
        ),
        "yolo_AP_drop":y ["AP_absolute_drop"],
        "dfine_AP_drop":d ["AP_absolute_drop"],
        "yolo_AP_retention":y ["AP_retention"],
        "dfine_AP_retention":d ["AP_retention"],
        "retention_gap_dfine_minus_yolo":(
        float (d ["AP_retention"])
        -float (y ["AP_retention"])
        ),
        }
        )

    comparison .sort (
    key =lambda x :(
    order [x ["corruption"]],
    int (x ["severity"]),
    )
    )

    atomic_csv (
    output_root /"model_comparison.csv",
    comparison ,
    [
    "variant",
    "corruption",
    "severity",
    "yolo_AP",
    "dfine_AP",
    "dfine_minus_yolo_AP",
    "yolo_AP_drop",
    "dfine_AP_drop",
    "yolo_AP_retention",
    "dfine_AP_retention",
    "retention_gap_dfine_minus_yolo",
    ],
    )

    groups :dict [
    tuple [str ,str ],
    list [dict [str ,Any ]]
    ]={}

    for row in metrics :
        groups .setdefault (
        (
        row ["model"],
        row ["corruption"],
        ),
        [],
        ).append (row )

    severity_summary =[]

    for (model ,corruption ),group in groups .items ():
        if len (group )!=3 :
            continue 

        group .sort (
        key =lambda x :int (x ["severity"])
        )

        worst =min (
        group ,
        key =lambda x :float (x ["AP"]),
        )

        severity_summary .append (
        {
        "model":model ,
        "corruption":corruption ,
        "mean_AP":float (
        np .mean (
        [float (x ["AP"])for x in group ]
        )
        ),
        "mean_AP_drop":float (
        np .mean (
        [
        float (x ["AP_absolute_drop"])
        for x in group 
        ]
        )
        ),
        "mean_AP_retention":float (
        np .mean (
        [
        float (x ["AP_retention"])
        for x in group 
        ]
        )
        ),
        "worst_variant":worst ["variant"],
        "worst_AP":worst ["AP"],
        "worst_AP_drop":worst ["AP_absolute_drop"],
        "worst_AP_retention":worst ["AP_retention"],
        }
        )

    severity_summary .sort (
    key =lambda x :(
    x ["model"],
    order [x ["corruption"]],
    )
    )

    atomic_csv (
    output_root /"severity_summary.csv",
    severity_summary ,
    [
    "model",
    "corruption",
    "mean_AP",
    "mean_AP_drop",
    "mean_AP_retention",
    "worst_variant",
    "worst_AP",
    "worst_AP_drop",
    "worst_AP_retention",
    ],
    )

    atomic_json (
    output_root /"progress.json",
    progress ,
    )






def run_yolo (
output_dir :Path ,
corruption :str ,
severity :int ,
batch :int ,
)->float :
    start =time .time ()

    run_command (
    [
    sys .executable ,
    str (YOLO_EXPORTER ),
    "--model",str (YOLO_MODEL ),
    "--dataset-root",str (DATASET_ROOT ),
    "--annotation",str (GT_PATH ),
    "--output-dir",str (output_dir ),
    "--corruption",corruption ,
    "--severity",str (severity ),
    "--device","0",
    "--imgsz","640",
    "--batch",str (batch ),
    "--conf","0.001",
    "--iou","0.7",
    "--max-det","300",
    "--limit","0",
    "--save-vis","0",
    "--overwrite",
    ],
    f"FORMAL YOLO11m | {vname (corruption ,severity )}",
    )

    return round (time .time ()-start ,3 )


def run_dfine (
conda :str ,
output_dir :Path ,
corruption :str ,
severity :int ,
)->float :
    start =time .time ()

    run_command (
    [
    conda ,
    "run",
    "--no-capture-output",
    "-p",
    str (DFINE_ENV ),
    "python",
    str (DFINE_EXPORTER ),
    "--dfine-root",str (DFINE_ROOT ),
    "--config",str (DFINE_CONFIG ),
    "--checkpoint",str (DFINE_CHECKPOINT ),
    "--dataset-root",str (DATASET_ROOT ),
    "--annotation",str (GT_PATH ),
    "--output-dir",str (output_dir ),
    "--corruption",corruption ,
    "--severity",str (severity ),
    "--device","cuda:0",
    "--limit","0",
    "--save-vis","0",
    "--vis-conf","0.25",
    "--overwrite",
    ],
    f"FORMAL D-FINE-M | {vname (corruption ,severity )}",
    )

    return round (time .time ()-start ,3 )






def main ()->None :
    args =parse_args ()

    try :
        import torch 
        import ultralytics 
    except ImportError as exc :
        raise RuntimeError (
        "Run this script from BASE:\n"
        "conda activate base"
        )from exc 

    if not torch .cuda .is_available ():
        raise RuntimeError ("CUDA is not available in BASE.")

    conda =shutil .which ("conda")

    if conda is None :
        raise RuntimeError ("conda executable not found.")

    required_files =[
    (YOLO_EXPORTER ,"YOLO controlled exporter"),
    (DFINE_EXPORTER ,"D-FINE controlled exporter"),
    (CONTROLLED_MODULE ,"controlled_corruptions.py"),
    (CORRUPTION_PROTOCOL ,"corruption_protocol.json"),
    (YOLO_MODEL ,"YOLO checkpoint"),
    (DFINE_CONFIG ,"D-FINE config"),
    (DFINE_CHECKPOINT ,"D-FINE checkpoint"),
    (GT_PATH ,"official BDD100K val GT"),
    ]

    for path ,label in required_files :
        require_file (path ,label )

    require_dir (DATASET_ROOT ,"BDD100K dataset root")
    require_dir (DFINE_ENV ,"D-FINE environment")

    gt_ids ,gt_id_set ,gt_counts =validate_gt ()

    variants =choose_variants (args .only )
    models =list (dict .fromkeys (args .models ))

    output_root =args .output_root .expanduser ().resolve ()
    temp_root =output_root /"temp"
    record_root =output_root /"variant_records"

    output_root .mkdir (parents =True ,exist_ok =True )
    temp_root .mkdir (parents =True ,exist_ok =True )
    record_root .mkdir (parents =True ,exist_ok =True )

    metric_json =output_root /"corruption_metrics.json"
    class_json =output_root /"corruption_per_class_metrics.json"

    metrics =load_existing (metric_json )
    per_class =load_existing (class_json )

    progress_path =output_root /"progress.json"

    if progress_path .is_file ():
        progress =load_json (progress_path )
    else :
        progress ={
        "stage":"formal_controlled_corruption_benchmark",
        "completed":{},
        }

    print ("="*120 )
    print ("FORMAL CONTROLLED CORRUPTION BENCHMARK")
    print ("="*120 )
    print (f"Images       : {EXPECTED_IMAGES }")
    print (f"GT objects   : {EXPECTED_ANNOTATIONS }")
    print (f"Variants     : {len (variants )}")
    print (f"Models       : {[MODEL_KEYS [x ]for x in models ]}")
    print (f"Output       : {output_root }")
    print (f"Keep temp    : {args .keep_temp }")
    print ("="*120 )

    run_start =time .time ()

    for variant_index ,(corruption ,severity )in enumerate (
    variants ,
    start =1 ,
    ):
        variant =vname (corruption ,severity )

        print ()
        print ("#"*120 )
        print (
        f"VARIANT {variant_index }/{len (variants )} | {variant }"
        )
        print ("#"*120 )

        for model_key in models :
            model =MODEL_KEYS [model_key ]

            if (
            has_metric (metrics ,model ,variant )
            and not args .force_inference 
            ):
                print (
                f"[SKIP FORMAL DONE] {model } | {variant }"
                )
                continue 

            temp_dir =(
            temp_root 
            /f"{model_key }_{variant }"
            )

            request_file =(
            temp_dir /"benchmark_request.json"
            )

            prediction_file =(
            temp_dir /"predictions.json"
            )

            summary_file =(
            temp_dir /"image_summary.json"
            )

            metadata_file =(
            temp_dir /"metadata.json"
            )

            reusable =False 

            if (
            not args .force_inference 
            and request_file .is_file ()
            and prediction_file .is_file ()
            and summary_file .is_file ()
            and metadata_file .is_file ()
            ):
                request =load_json (request_file )

                reusable =(
                request .get ("model")==model 
                and request .get ("variant")==variant 
                )

            if reusable :
                print (
                f"[RESUME TEMP] {model } | {variant }"
                )

                inference_seconds =float (
                load_json (request_file ).get (
                "inference_seconds",
                -1.0 ,
                )
                )

            else :
                if temp_dir .exists ():
                    shutil .rmtree (temp_dir )

                temp_dir .mkdir (
                parents =True ,
                exist_ok =True ,
                )

                atomic_json (
                request_file ,
                {
                "model":model ,
                "variant":variant ,
                "corruption":corruption ,
                "severity":severity ,
                "created_at_utc":now_utc (),
                },
                )

                progress ["completed"][
                f"{model }::{variant }"
                ]={
                "status":"inference_started",
                "updated_at_utc":now_utc (),
                }

                save_outputs (
                output_root ,
                metrics ,
                per_class ,
                progress ,
                )

                if model_key =="yolo":
                    inference_seconds =run_yolo (
                    temp_dir ,
                    corruption ,
                    severity ,
                    args .yolo_batch ,
                    )
                else :
                    inference_seconds =run_dfine (
                    conda ,
                    temp_dir ,
                    corruption ,
                    severity ,
                    )

                request =load_json (request_file )
                request ["inference_seconds"]=inference_seconds 
                atomic_json (request_file ,request )

            require_file (prediction_file ,"predictions.json")
            require_file (summary_file ,"image_summary.json")
            require_file (metadata_file ,"metadata.json")

            validate_summary (
            summary_file ,
            gt_id_set ,
            )

            row ,class_rows =evaluate (
            model =model ,
            variant =variant ,
            corruption =corruption ,
            severity =severity ,
            prediction_path =prediction_file ,
            gt_ids =gt_ids ,
            gt_count_by_class =gt_counts ,
            inference_seconds =inference_seconds ,
            )

            metrics =replace_rows (
            metrics ,
            model ,
            variant ,
            [row ],
            )

            per_class =replace_rows (
            per_class ,
            model ,
            variant ,
            class_rows ,
            )

            record ={
            "model":model ,
            "variant":variant ,
            "metric":row ,
            "exporter_metadata":load_json (metadata_file ),
            "prediction_deleted_after_metric_save":(
            not args .keep_temp 
            ),
            "saved_at_utc":now_utc (),
            }

            atomic_json (
            record_root 
            /f"{model_key }_{variant }.json",
            record ,
            )

            progress ["completed"][
            f"{model }::{variant }"
            ]={
            "status":"done",
            "AP":row ["AP"],
            "AP_drop":row ["AP_absolute_drop"],
            "AP_retention":row ["AP_retention"],
            "updated_at_utc":now_utc (),
            }

            progress ["last_updated_utc"]=now_utc ()


            save_outputs (
            output_root ,
            metrics ,
            per_class ,
            progress ,
            )

            if not args .keep_temp :
                shutil .rmtree (temp_dir )

            print (
            f"[FORMAL DONE] {model :<10} {variant :<18} "
            f"AP={row ['AP']:.4f} "
            f"drop={row ['AP_absolute_drop']:+.4f} "
            f"retention={row ['AP_retention']:.4f}"
            )

    metadata ={
    "stage":"formal_controlled_corruption_benchmark",
    "dataset":"BDD100K official val",
    "num_images":EXPECTED_IMAGES ,
    "num_annotations":EXPECTED_ANNOTATIONS ,
    "models":[MODEL_KEYS [x ]for x in models ],
    "variants":[vname (c ,s )for c ,s in variants ],
    "clean_baseline":CLEAN ,
    "clean_baseline_source":(
    "/root/rivermind-data/autodrive/results/evaluation/"
    "unified_detection"
    ),
    "protocol":{
    "on_the_fly":True ,
    "before_model_preprocessing":True ,
    "geometry_changed":False ,
    "gt_modified":False ,
    "full_corrupted_images_saved":False ,
    "validation_used_for_training":False ,
    },
    "disk_policy":{
    "one_temporary_prediction_at_a_time":True ,
    "delete_temp_after_metric_persistence":(
    not args .keep_temp 
    ),
    },
    "sha256":{
    "gt":sha256_file (GT_PATH ),
    "yolo_checkpoint":sha256_file (YOLO_MODEL ),
    "dfine_checkpoint":sha256_file (DFINE_CHECKPOINT ),
    "dfine_config":sha256_file (DFINE_CONFIG ),
    "controlled_corruptions":sha256_file (
    CONTROLLED_MODULE 
    ),
    "corruption_protocol":sha256_file (
    CORRUPTION_PROTOCOL 
    ),
    "yolo_exporter":sha256_file (
    YOLO_EXPORTER 
    ),
    "dfine_exporter":sha256_file (
    DFINE_EXPORTER 
    ),
    },
    "software":{
    "python":platform .python_version (),
    "numpy":np .__version__ ,
    "torch":torch .__version__ ,
    "ultralytics":ultralytics .__version__ ,
    },
    "elapsed_hours":round (
    (time .time ()-run_start )/3600.0 ,
    4 ,
    ),
    "finished_at_utc":now_utc (),
    }

    atomic_json (
    output_root /"benchmark_metadata.json",
    metadata ,
    )

    missing =[]

    for corruption ,severity in variants :
        variant =vname (corruption ,severity )

        for model_key in models :
            model =MODEL_KEYS [model_key ]

            if not has_metric (metrics ,model ,variant ):
                missing .append (
                (model ,variant )
                )

    print ()
    print ("="*120 )

    if missing :
        print (
        "FORMAL CONTROLLED CORRUPTION BENCHMARK PARTIAL"
        )
        print (f"Missing: {missing }")
    else :
        print (
        "FORMAL CONTROLLED CORRUPTION BENCHMARK COMPLETE"
        )

    print ("="*120 )
    print (
    f"Metrics     : {output_root /'corruption_metrics.csv'}"
    )
    print (
    f"Per-class   : {output_root /'corruption_per_class_metrics.csv'}"
    )
    print (
    f"Comparison  : {output_root /'model_comparison.csv'}"
    )
    print (
    f"Severity    : {output_root /'severity_summary.csv'}"
    )
    print (
    f"Progress    : {output_root /'progress.json'}"
    )
    print (
    f"Metadata    : {output_root /'benchmark_metadata.json'}"
    )
    print ("="*120 )


if __name__ =="__main__":
    main ()
