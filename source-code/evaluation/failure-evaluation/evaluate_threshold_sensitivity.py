from __future__ import annotations 

import argparse 
import csv 
import gc 
import hashlib 
import json 
import math 
import os 
import shutil 
import subprocess 
import sys 
import time 
from collections import Counter ,defaultdict 
from concurrent .futures import ThreadPoolExecutor ,as_completed 
from pathlib import Path 

import numpy as np 


ROOT =Path (
"/root/rivermind-data/autodrive/results/evaluation/"
"final_analysis_v1"
)

PROJECT =Path (
"/root/rivermind-data/autodrive/code/"
"robust-autonomous-driving-perception"
)

PROTOCOL =(
ROOT /"frozen_protocol/final_testing_protocol_v1.json"
)
CANONICAL_MANIFEST =(
ROOT /"canonical_clean_v1/artifact_manifest.json"
)
CENTRAL_MATCHES =(
ROOT /"official_val_failure_matches_v1"
)
CENTRAL_MATCH_MANIFEST =(
CENTRAL_MATCHES /"artifact_manifest.json"
)
CENTRAL_SUBSETS =(
ROOT /"official_val_failure_subsets_v1"
)
CENTRAL_SUBSET_MANIFEST =(
CENTRAL_SUBSETS /"artifact_manifest.json"
)
CENTRAL_FAILURE_TABLE =(
CENTRAL_SUBSETS /"failure_object_table.csv"
)

GT =Path (
"/root/rivermind-data/autodrive/datasets/datasets/"
"bdd100k_final/coco/annotations/instances_val.json"
)

CALIBRATION =Path (
"/root/rivermind-data/autodrive/results/evaluation/"
"train_risk_labels/operating_point_calibration"
)

OPERATING_POINTS =CALIBRATION /"operating_points.json"

MATCHER =PROJECT /"scripts/build_train_detection_matches.py"

BASELINE_PREDICTIONS ={
"YOLO11m":Path (
"/root/rivermind-data/autodrive/results/evaluation/"
"unified_detection/canonical_predictions/YOLO11m.json"
),
"D-FINE-M":Path (
"/root/rivermind-data/autodrive/results/evaluation/"
"unified_detection/canonical_predictions/D-FINE-M.json"
),
}

GATE_PREDICTIONS ={
key :ROOT /f"canonical_clean_v1/predictions/{key }/predictions.json"
for key in (
"gate_uniform",
"gate_scalar",
"gate_typed",
"gate_typed_v2",
)
}

SENSITIVITY_FILES ={
"YOLO11m":CALIBRATION /"YOLO11m_threshold_sensitivity.csv",
"D-FINE-M":CALIBRATION /"D-FINE-M_threshold_sensitivity.csv",
}

OUTPUT =ROOT /"threshold_sensitivity_v1"

EXPECTED_SHA256 ={
PROTOCOL :
"693c39375cc1eefc4f3ff9f957a8355fb947552a19dfef228cae6cfc21e3dcfe",
CANONICAL_MANIFEST :
"88d2bace7a2f69e26c34e7eed4e2814b12e8317980602968b407d1b94282338b",
CENTRAL_MATCH_MANIFEST :
"a60485d5242bcf66b6b86be4ffe58bfe27ad32bb4fb6c30811fa6a917d4ee7ea",
CENTRAL_SUBSET_MANIFEST :
"79029af794f214d45bdecb6cb00d9b64155aa8ace04df97ca6524e1f40f2c1dc",
GT :
"7be8a02e147743123a4a62b2c92ed2e4a5712d261c8868223e6325bad6e346d6",
OPERATING_POINTS :
"f13868c935ea16091fd0bae592bb9817d77bd58d2f45597dc7d2add8e2fa44ae",
MATCHER :
"3a3b5aa6efd5c956b429769fcdf45f0def85ec174c8164e5f3c217d62e6337c2",
BASELINE_PREDICTIONS ["YOLO11m"]:
"14ff8491f7849077cb7ead3ce7d9e618d2f92ee04bef0a8c68efdb1b836a54b0",
BASELINE_PREDICTIONS ["D-FINE-M"]:
"10b4159d0f52e412714ac9ae076ac574ebbb74321ddcebab03e2bdbad55e3ebd",
SENSITIVITY_FILES ["YOLO11m"]:
"e8a77870d1f27d889bb6c465c5259256452273a4f7824c0a74e319c56fd5df22",
SENSITIVITY_FILES ["D-FINE-M"]:
"d47c36fd350b4c6b5318787403f39da07a056712db86fe75d08e8638211c997d",
}

SUBSETS =(
"small_jointly_missed",
"dfine_supported_yolo_failure",
"yolo_low_confidence",
"yolo_localization_failure",
"yolo_classification_failure",
"small_high_risk",
)

FAILURE_MACRO_SUBSETS =(
"small_jointly_missed",
"dfine_supported_yolo_failure",
"yolo_low_confidence",
"yolo_localization_failure",
"yolo_classification_failure",
)

COMPARISONS ={
"gate_scalar_minus_gate_uniform":
("gate_scalar","gate_uniform"),
"gate_typed_minus_gate_uniform":
("gate_typed","gate_uniform"),
"gate_typed_minus_gate_scalar":
("gate_typed","gate_scalar"),
"gate_typed_v2_minus_gate_uniform":
("gate_typed_v2","gate_uniform"),
}

EXPECTED_OBJECTS =185_523 
EXPECTED_IMAGES =10_000 
MATCH_IOU =0.50 
LOCALIZATION_IOU =0.10 
CONFIDENCE_FLOOR =0.001 
MAX_DET =100 
CLASS_PRIOR_STRENGTH =500.0 
NONDETECTION_MULTIPLIER =0.5 


def parse_args ():
    parser =argparse .ArgumentParser ()
    group =parser .add_mutually_exclusive_group (required =True )
    group .add_argument ("--preflight-only",action ="store_true")
    group .add_argument ("--execute",action ="store_true")
    return parser .parse_args ()


def sha256_file (path :Path )->str :
    digest =hashlib .sha256 ()
    with path .open ("rb")as handle :
        while True :
            block =handle .read (8 *1024 *1024 )
            if not block :
                break 
            digest .update (block )
    return digest .hexdigest ()


def write_json (path :Path ,value )->None :
    path .write_text (
    json .dumps (
    value ,
    indent =2 ,
    ensure_ascii =False ,
    sort_keys =True ,
    )+"\n"
    )


def verify (path :Path ,expected :str )->None :
    if not path .is_file ():
        raise FileNotFoundError (path )
    actual =sha256_file (path )
    print (path )
    print (" expected:",expected )
    print (" actual  :",actual )
    if actual !=expected :
        raise AssertionError (path )


def parse_plateau (model :str )->dict :
    path =SENSITIVITY_FILES [model ]

    with path .open (newline ="")as handle :
        rows =list (csv .DictReader (handle ))

    parsed =[
    {
    "threshold":float (row ["threshold"]),
    "macro_f1":float (row ["macro_f1"]),
    }
    for row in rows 
    ]
    parsed .sort (key =lambda row :row ["threshold"])

    maximum =max (row ["macro_f1"]for row in parsed )
    minimum =0.95 *maximum 

    plateau =[
    row for row in parsed 
    if row ["macro_f1"]>=minimum 
    ]

    central =max (
    parsed ,
    key =lambda row :row ["macro_f1"],
    )

    points ={
    "low":plateau [0 ],
    "central":central ,
    "high":plateau [-1 ],
    }

    if not (
    points ["low"]["threshold"]
    <=points ["central"]["threshold"]
    <=points ["high"]["threshold"]
    ):
        raise AssertionError (model )

    return {
    "maximum_macro_f1":maximum ,
    "plateau_minimum":minimum ,
    "plateau_rows":len (plateau ),
    "points":points ,
    }


def preflight ():
    print ("=== FINAL THRESHOLD-SENSITIVITY PREFLIGHT ===")

    for path ,expected in EXPECTED_SHA256 .items ():
        verify (path ,expected )

    for path in GATE_PREDICTIONS .values ():
        if not path .is_file ():
            raise FileNotFoundError (path )
        print (
        "PASS prediction:",
        path ,
        f"{path .stat ().st_size /1024 **2 :.3f} MiB",
        )

    protocol =json .loads (PROTOCOL .read_text ())
    section =protocol ["threshold_sensitivity"]

    assert section ["enabled"]is True 
    assert section ["plateau_rule"]==(
    "macro_f1 >= 0.95 * maximum_macro_f1"
    )

    plateaus ={
    model :parse_plateau (model )
    for model in ("YOLO11m","D-FINE-M")
    }

    expected_thresholds ={
    "YOLO11m":{
    "low":0.13 ,
    "central":0.22527144849300385 ,
    "high":0.37 ,
    },
    "D-FINE-M":{
    "low":0.41 ,
    "central":0.4963708519935608 ,
    "high":0.60 ,
    },
    }

    for model ,expected_points in expected_thresholds .items ():
        for point ,expected in expected_points .items ():
            actual =(
            plateaus [model ]["points"][point ]["threshold"]
            )
            if not math .isclose (
            actual ,
            expected ,
            rel_tol =0.0 ,
            abs_tol =1e-12 ,
            ):
                raise AssertionError (
                (model ,point ,actual ,expected )
                )

    for model ,data in plateaus .items ():
        print ("\n",model )
        for point ,row in data ["points"].items ():
            print (
            point ,
            "threshold=",
            row ["threshold"],
            "macro_f1=",
            row ["macro_f1"],
            )

    central_rows =0 
    with CENTRAL_FAILURE_TABLE .open (newline ="")as handle :
        for central_rows ,_ in enumerate (
        csv .DictReader (handle ),
        start =1 ,
        ):
            pass 

    assert central_rows ==EXPECTED_OBJECTS 

    if OUTPUT .exists ():
        raise FileExistsError (OUTPUT )

    incomplete =sorted (
    OUTPUT .parent .glob (OUTPUT .name +".incomplete-*")
    )
    if incomplete :
        raise AssertionError (incomplete )

    free_gib =shutil .disk_usage (OUTPUT .parent ).free /1024 **3 
    print ("\nfree_GiB:",round (free_gib ,3 ))
    if free_gib <3.0 :
        raise RuntimeError ("At least 3 GiB free space is required.")

    print ("PASS: all Stage-3 inputs and rules are frozen")
    return protocol ,plateaus 


def expected_prediction_count (model :str )->int :
    metadata_path =CENTRAL_MATCHES /"metadata.json"
    metadata =json .loads (metadata_path .read_text ())

    model_record =metadata ["models"][model ]
    audit =model_record .get ("prediction_audit",{})

    value =audit .get (
    "prediction_count",
    model_record .get ("prediction_count"),
    )

    if value is None :
        raise KeyError (
        f"Missing prediction count for {model }"
        )

    value =int (value )

    if value <=0 :
        raise ValueError (
        f"Invalid prediction count for {model }: {value }"
        )

    return value 


def run_match_combination (
job_name :str ,
yolo_threshold :float ,
dfine_threshold :float ,
staging :Path ,
prediction_counts :dict [str ,int ],
)->Path :
    output_dir =(
    staging /"teacher_matches"/job_name 
    )

    command =[
    sys .executable ,
    "-u",
    str (MATCHER ),
    "--gt",
    str (GT ),
    "--calibration",
    str (OPERATING_POINTS ),
    "--expected-calibration-sha256",
    EXPECTED_SHA256 [OPERATING_POINTS ],
    "--prediction",
    f"YOLO11m={BASELINE_PREDICTIONS ['YOLO11m']}",
    "--prediction",
    f"D-FINE-M={BASELINE_PREDICTIONS ['D-FINE-M']}",
    "--threshold",
    f"YOLO11m={format (yolo_threshold ,'.17g')}",
    "--threshold",
    f"D-FINE-M={format (dfine_threshold ,'.17g')}",
    "--expected-predictions",
    f"YOLO11m={prediction_counts ['YOLO11m']}",
    "--expected-predictions",
    f"D-FINE-M={prediction_counts ['D-FINE-M']}",
    "--prediction-sha256",
    (
    "YOLO11m="
    +EXPECTED_SHA256 [
    BASELINE_PREDICTIONS ["YOLO11m"]
    ]
    ),
    "--prediction-sha256",
    (
    "D-FINE-M="
    +EXPECTED_SHA256 [
    BASELINE_PREDICTIONS ["D-FINE-M"]
    ]
    ),
    "--expected-images",
    str (EXPECTED_IMAGES ),
    "--expected-annotations",
    str (EXPECTED_OBJECTS ),
    "--match-iou",
    str (MATCH_IOU ),
    "--localization-iou",
    str (LOCALIZATION_IOU ),
    "--output-dir",
    str (output_dir ),
    ]

    print ("\n"+"="*100 ,flush =True )
    print ("MATCHING JOB:",job_name ,flush =True )
    print (
    "YOLO11m threshold:",
    yolo_threshold ,
    flush =True ,
    )
    print (
    "D-FINE-M threshold:",
    dfine_threshold ,
    flush =True ,
    )
    print ("="*100 ,flush =True )

    environment =os .environ .copy ()
    environment ["OMP_NUM_THREADS"]="4"
    environment ["OPENBLAS_NUM_THREADS"]="4"
    environment ["MKL_NUM_THREADS"]="4"
    environment ["NUMEXPR_NUM_THREADS"]="4"

    subprocess .run (
    command ,
    cwd =PROJECT ,
    check =True ,
    env =environment ,
    )

    for model in ("YOLO11m","D-FINE-M"):
        result =output_dir /model /"object_matches.csv"

        if not result .is_file ():
            raise FileNotFoundError (result )

    return output_dir 

def optional_float (value :str )->float :
    if value in ("",None ):
        return 0.0 
    result =float (value )
    if not math .isfinite (result ):
        raise ValueError (value )
    return result 


def load_match_table (path :Path )->dict :
    fields =defaultdict (list )

    with path .open (newline ="")as handle :
        reader =csv .DictReader (handle )

        for row in reader :
            fields ["annotation_id"].append (
            int (row ["annotation_id"])
            )
            fields ["image_id"].append (
            int (row ["image_id"])
            )
            fields ["category_id"].append (
            int (row ["category_id"])
            )
            fields ["class_name"].append (
            row ["category"]
            )
            fields ["scale"].append (row ["scale"])
            fields ["status"].append (row ["status"])
            fields ["detected"].append (
            int (row ["detected"])
            )
            fields ["matched_iou"].append (
            optional_float (row ["matched_iou"])
            )
            fields ["matched_score"].append (
            optional_float (row ["matched_score"])
            )
            fields ["best_iou"].append (
            optional_float (
            row ["best_same_all_iou"]
            )
            )
            fields ["best_score"].append (
            optional_float (
            row ["best_same_all_score"]
            )
            )

    if len (fields ["annotation_id"])!=EXPECTED_OBJECTS :
        raise AssertionError (
        (path ,len (fields ["annotation_id"]))
        )

    return {
    "annotation_id":np .asarray (
    fields ["annotation_id"],
    dtype =np .int64 ,
    ),
    "image_id":np .asarray (
    fields ["image_id"],
    dtype =np .int64 ,
    ),
    "category_id":np .asarray (
    fields ["category_id"],
    dtype =np .int16 ,
    ),
    "class_name":np .asarray (
    fields ["class_name"],
    dtype =object ,
    ),
    "scale":np .asarray (
    fields ["scale"],
    dtype =object ,
    ),
    "status":np .asarray (
    fields ["status"],
    dtype =object ,
    ),
    "detected":np .asarray (
    fields ["detected"],
    dtype =np .uint8 ,
    ),
    "matched_iou":np .asarray (
    fields ["matched_iou"],
    dtype =np .float32 ,
    ),
    "matched_score":np .asarray (
    fields ["matched_score"],
    dtype =np .float32 ,
    ),
    "best_iou":np .asarray (
    fields ["best_iou"],
    dtype =np .float32 ,
    ),
    "best_score":np .asarray (
    fields ["best_score"],
    dtype =np .float32 ,
    ),
    }


def confidence_support (
scores :np .ndarray ,
threshold :float ,
)->np .ndarray :
    scores =np .clip (scores ,0.0 ,1.0 )
    result =np .empty_like (scores ,dtype =np .float32 )

    lower =scores <=threshold 

    result [lower ]=(
    0.5 *scores [lower ]/threshold 
    if threshold >0.0 
    else 0.5 
    )

    result [~lower ]=(
    0.5 
    +0.5 
    *(scores [~lower ]-threshold )
    /(1.0 -threshold )
    if threshold <1.0 
    else 0.5 
    )

    return result 


def detector_quality (
table :dict ,
threshold :float ,
)->np .ndarray :
    detected =table ["detected"].astype (bool )

    score =np .where (
    detected ,
    table ["matched_score"],
    table ["best_score"],
    ).astype (np .float32 )

    iou =np .where (
    detected ,
    table ["matched_iou"],
    table ["best_iou"],
    ).astype (np .float32 )

    multiplier =np .where (
    detected ,
    1.0 ,
    NONDETECTION_MULTIPLIER ,
    ).astype (np .float32 )

    quality =multiplier *np .sqrt (
    confidence_support (score ,threshold )
    *np .clip (iou ,0.0 ,1.0 )
    )

    return np .clip (
    quality ,
    0.0 ,
    1.0 ,
    ).astype (np .float32 )


def percentile_rank (values :np .ndarray )->np .ndarray :
    count =len (values )

    if count ==1 :
        return np .asarray ([0.5 ],dtype =np .float32 )

    order =np .argsort (values ,kind ="mergesort")
    sorted_values =values [order ]

    _ ,starts ,counts =np .unique (
    sorted_values ,
    return_index =True ,
    return_counts =True ,
    )

    positions =(
    starts .astype (np .float64 )
    +(counts .astype (np .float64 )-1.0 )/2.0 
    )/float (count -1 )

    sorted_ranks =np .repeat (
    positions ,
    counts ,
    ).astype (np .float32 )

    ranks =np .empty (count ,dtype =np .float32 )
    ranks [order ]=sorted_ranks 
    return ranks 


def make_subsets (
yolo :dict ,
dfine :dict ,
yolo_threshold :float ,
dfine_threshold :float ,
yolo_macro_f1 :float ,
dfine_macro_f1 :float ,
)->dict [str ,np .ndarray ]:
    if not np .array_equal (
    yolo ["annotation_id"],
    dfine ["annotation_id"],
    ):
        raise AssertionError (
        "YOLO and D-FINE annotation ordering differs."
        )

    yolo_quality =detector_quality (
    yolo ,
    yolo_threshold ,
    )
    dfine_quality =detector_quality (
    dfine ,
    dfine_threshold ,
    )

    yolo_risk =1.0 -yolo_quality 
    dfine_risk =1.0 -dfine_quality 

    reliability_sum =(
    yolo_macro_f1 +dfine_macro_f1 
    )

    consensus =(
    yolo_macro_f1 /reliability_sum *yolo_risk 
    +dfine_macro_f1 /reliability_sum *dfine_risk 
    ).astype (np .float32 )

    disagreement =np .abs (
    yolo_risk -dfine_risk 
    ).astype (np .float32 )

    joint =(
    1.0 
    -(1.0 -consensus )
    *(1.0 -disagreement )
    ).astype (np .float32 )

    global_percentile =percentile_rank (joint )

    category_ids =yolo ["category_id"]
    class_percentile =np .empty (
    EXPECTED_OBJECTS ,
    dtype =np .float32 ,
    )
    class_reliability =np .empty (
    EXPECTED_OBJECTS ,
    dtype =np .float32 ,
    )

    for category_id in sorted (set (category_ids .tolist ())):
        positions =np .flatnonzero (
        category_ids ==category_id 
        )
        class_percentile [positions ]=percentile_rank (
        joint [positions ]
        )
        count =len (positions )
        class_reliability [positions ]=(
        count /(count +CLASS_PRIOR_STRENGTH )
        )

    risk_score =(
    class_reliability *class_percentile 
    +(1.0 -class_reliability )
    *global_percentile 
    ).astype (np .float32 )

    small =yolo ["scale"]=="small"
    yolo_detected =yolo ["detected"].astype (bool )
    dfine_detected =dfine ["detected"].astype (bool )

    return {
    "small_jointly_missed":(
    small 
    &~yolo_detected 
    &~dfine_detected 
    ),
    "dfine_supported_yolo_failure":(
    dfine_detected 
    &~yolo_detected 
    ),
    "yolo_low_confidence":(
    yolo ["status"]
    =="low_confidence_candidate"
    ),
    "yolo_localization_failure":(
    yolo ["status"]
    =="localization_error_candidate"
    ),
    "yolo_classification_failure":(
    yolo ["status"]
    =="classification_error_candidate"
    ),
    "small_high_risk":(
    small &(risk_score >=0.75 )
    ),
    }


def load_central_sets ()->dict [str ,set [int ]]:
    result ={
    subset :set ()
    for subset in SUBSETS 
    }

    with CENTRAL_FAILURE_TABLE .open (newline ="")as handle :
        reader =csv .DictReader (handle )

        for row in reader :
            annotation_id =int (row ["annotation_id"])

            for subset in SUBSETS :
                if int (row [f"in_{subset }"]):
                    result [subset ].add (annotation_id )

    return result 


def iou_matrix (
detections :np .ndarray ,
ground_truth :np .ndarray ,
)->np .ndarray :
    if len (detections )==0 or len (ground_truth )==0 :
        return np .zeros (
        (len (detections ),len (ground_truth )),
        dtype =np .float32 ,
        )

    dx1 =detections [:,0 :1 ]
    dy1 =detections [:,1 :2 ]
    dx2 =dx1 +detections [:,2 :3 ]
    dy2 =dy1 +detections [:,3 :4 ]

    gx1 =ground_truth [:,0 ][None ,:]
    gy1 =ground_truth [:,1 ][None ,:]
    gx2 =gx1 +ground_truth [:,2 ][None ,:]
    gy2 =gy1 +ground_truth [:,3 ][None ,:]

    ix1 =np .maximum (dx1 ,gx1 )
    iy1 =np .maximum (dy1 ,gy1 )
    ix2 =np .minimum (dx2 ,gx2 )
    iy2 =np .minimum (dy2 ,gy2 )

    intersection =(
    np .maximum (0.0 ,ix2 -ix1 )
    *np .maximum (0.0 ,iy2 -iy1 )
    )

    detection_area =np .maximum (
    0.0 ,
    (dx2 -dx1 )*(dy2 -dy1 ),
    )
    gt_area =np .maximum (
    0.0 ,
    (gx2 -gx1 )*(gy2 -gy1 ),
    )

    union =detection_area +gt_area -intersection 

    return np .divide (
    intersection ,
    union ,
    out =np .zeros_like (intersection ),
    where =union >0 ,
    ).astype (np .float32 )


def load_gate_predictions (path :Path ):
    payload =json .loads (path .read_text ())
    groups =defaultdict (list )

    for prediction in payload :
        score =float (prediction ["score"])
        bbox =prediction ["bbox"]

        if score <CONFIDENCE_FLOOR :
            continue 
        if float (bbox [2 ])<=0 or float (bbox [3 ])<=0 :
            continue 

        key =(
        int (prediction ["image_id"]),
        int (prediction ["category_id"]),
        )

        groups [key ].append (
        (
        score ,
        [
        float (bbox [0 ]),
        float (bbox [1 ]),
        float (bbox [2 ]),
        float (bbox [3 ]),
        ],
        )
        )

    compact ={}

    for key ,rows in groups .items ():
        rows .sort (
        key =lambda row :row [0 ],
        reverse =True ,
        )
        compact [key ]=np .asarray (
        [row [1 ]for row in rows [:MAX_DET ]],
        dtype =np .float32 ,
        )

    del payload 
    del groups 
    gc .collect ()
    return compact 


def load_gt_annotations ():
    data =json .loads (GT .read_text ())

    annotations ={}
    class_names ={
    int (category ["id"]):category ["name"]
    for category in data ["categories"]
    }

    for annotation in data ["annotations"]:
        annotations [int (annotation ["id"])]={
        "image_id":int (annotation ["image_id"]),
        "category_id":int (annotation ["category_id"]),
        "bbox":[
        float (value )
        for value in annotation ["bbox"]
        ],
        }

    assert len (annotations )==EXPECTED_OBJECTS 
    assert len (data ["images"])==EXPECTED_IMAGES 
    return annotations ,class_names 


def subset_recovery_metrics (
annotation_ids :set [int ],
gt_annotations :dict ,
prediction_groups :dict ,
)->dict :
    gt_groups =defaultdict (list )
    category_totals =Counter ()

    for annotation_id in annotation_ids :
        annotation =gt_annotations [annotation_id ]
        key =(
        annotation ["image_id"],
        annotation ["category_id"],
        )
        gt_groups [key ].append (annotation ["bbox"])
        category_totals [annotation ["category_id"]]+=1 

    thresholds =np .arange (
    0.50 ,
    0.96 ,
    0.05 ,
    dtype =np .float32 ,
    )

    tp_by_threshold_category ={
    float (threshold ):Counter ()
    for threshold in thresholds 
    }

    total_tp_50 =0 

    for key ,boxes in gt_groups .items ():
        category_id =key [1 ]
        gt_boxes =np .asarray (
        boxes ,
        dtype =np .float32 ,
        )
        detections =prediction_groups .get (
        key ,
        np .empty ((0 ,4 ),dtype =np .float32 ),
        )

        ious =iou_matrix (
        detections ,
        gt_boxes ,
        )

        for threshold in thresholds :
            matched =np .zeros (
            len (gt_boxes ),
            dtype =bool ,
            )
            true_positives =0 

            for detection_index in range (len (detections )):
                candidates =np .flatnonzero (
                (~matched )
                &(
                ious [detection_index ]
                >=float (threshold )
                )
                )

                if len (candidates )==0 :
                    continue 

                best =candidates [
                np .argmax (
                ious [detection_index ,candidates ]
                )
                ]
                matched [best ]=True 
                true_positives +=1 

            tp_by_threshold_category [
            float (threshold )
            ][category_id ]+=true_positives 

            if math .isclose (
            float (threshold ),
            0.50 ,
            abs_tol =1e-6 ,
            ):
                total_tp_50 +=true_positives 

    category_ar =[]

    for category_id ,total in category_totals .items ():
        recalls =[]

        for threshold in thresholds :
            recalls .append (
            tp_by_threshold_category [
            float (threshold )
            ][category_id ]/total 
            )

        category_ar .append (float (np .mean (recalls )))

    return {
    "objects":len (annotation_ids ),
    "images":len ({
    gt_annotations [annotation_id ]["image_id"]
    for annotation_id in annotation_ids 
    }),
    "AR100":(
    float (np .mean (category_ar ))
    if category_ar 
    else 0.0 
    ),
    "recall50":(
    total_tp_50 /len (annotation_ids )
    if annotation_ids 
    else 0.0 
    ),
    }


def sign (value :float ,tolerance :float =1e-12 )->int :
    if value >tolerance :
        return 1 
    if value <-tolerance :
        return -1 
    return 0 


def build_manifest (directory :Path ):
    artifacts =[]

    for path in sorted (directory .rglob ("*")):
        if not path .is_file ():
            continue 
        if path .name =="artifact_manifest.json":
            continue 

        artifacts .append ({
        "path":str (path .relative_to (directory )),
        "bytes":path .stat ().st_size ,
        "sha256":sha256_file (path ),
        })

    manifest ={
    "created_utc":time .strftime (
    "%Y-%m-%dT%H:%M:%SZ",
    time .gmtime (),
    ),
    "artifacts":artifacts ,
    }

    manifest_path =directory /"artifact_manifest.json"
    write_json (manifest_path ,manifest )
    return manifest_path ,sha256_file (manifest_path )


def execute (protocol :dict ,plateaus :dict ):
    started =time .time ()

    staging =OUTPUT .with_name (
    OUTPUT .name 
    +f".incomplete-{int (time .time ())}-{os .getpid ()}"
    )
    staging .mkdir (parents =True )

    match_paths ={
    "YOLO11m":{
    "central":
    CENTRAL_MATCHES 
    /"YOLO11m/object_matches.csv"
    },
    "D-FINE-M":{
    "central":
    CENTRAL_MATCHES 
    /"D-FINE-M/object_matches.csv"
    },
    }

    prediction_counts ={
    model :expected_prediction_count (model )
    for model in ("YOLO11m","D-FINE-M")
    }

    print ("prediction_counts:",prediction_counts )

    yolo_low =(
    plateaus ["YOLO11m"]["points"]["low"]["threshold"]
    )
    yolo_central =(
    plateaus ["YOLO11m"]["points"]["central"]["threshold"]
    )
    yolo_high =(
    plateaus ["YOLO11m"]["points"]["high"]["threshold"]
    )

    dfine_low =(
    plateaus ["D-FINE-M"]["points"]["low"]["threshold"]
    )
    dfine_central =(
    plateaus ["D-FINE-M"]["points"]["central"]["threshold"]
    )
    dfine_high =(
    plateaus ["D-FINE-M"]["points"]["high"]["threshold"]
    )

    job_specs ={
    "yolo_low":(
    yolo_low ,
    dfine_central ,
    ),
    "yolo_high":(
    yolo_high ,
    dfine_central ,
    ),
    "dfine_low":(
    yolo_central ,
    dfine_low ,
    ),
    "dfine_high":(
    yolo_central ,
    dfine_high ,
    ),
    }

    print ("\n=== RUNNING FOUR MATCH JOBS IN PARALLEL ===")
    print ("parallel_workers: 4")

    job_outputs ={}

    with ThreadPoolExecutor (max_workers =4 )as executor :
        futures ={
        executor .submit (
        run_match_combination ,
        job_name ,
        thresholds [0 ],
        thresholds [1 ],
        staging ,
        prediction_counts ,
        ):job_name 
        for job_name ,thresholds in job_specs .items ()
        }

        for future in as_completed (futures ):
            job_name =futures [future ]
            job_outputs [job_name ]=future .result ()
            print (
            "PASS MATCH JOB:",
            job_name ,
            flush =True ,
            )

    match_paths ["YOLO11m"]["low"]=(
    job_outputs ["yolo_low"]
    /"YOLO11m/object_matches.csv"
    )
    match_paths ["YOLO11m"]["high"]=(
    job_outputs ["yolo_high"]
    /"YOLO11m/object_matches.csv"
    )
    match_paths ["D-FINE-M"]["low"]=(
    job_outputs ["dfine_low"]
    /"D-FINE-M/object_matches.csv"
    )
    match_paths ["D-FINE-M"]["high"]=(
    job_outputs ["dfine_high"]
    /"D-FINE-M/object_matches.csv"
    )

    print ("\n=== LOADING SIX TEACHER MATCH TABLES ===")

    match_tables ={
    model :{
    point :load_match_table (path )
    for point ,path in paths .items ()
    }
    for model ,paths in match_paths .items ()
    }

    central_sets =load_central_sets ()
    combination_sets ={}
    subset_rows =[]
    class_rows =[]
    scale_rows =[]

    print ("\n=== BUILDING 3 x 3 FAILURE CONDITIONS ===")

    for yolo_point in ("low","central","high"):
        for dfine_point in ("low","central","high"):
            combination =(
            f"yolo_{yolo_point }"
            f"__dfine_{dfine_point }"
            )

            yolo_point_data =(
            plateaus ["YOLO11m"]["points"][yolo_point ]
            )
            dfine_point_data =(
            plateaus ["D-FINE-M"]["points"][dfine_point ]
            )

            yolo_table =(
            match_tables ["YOLO11m"][yolo_point ]
            )
            dfine_table =(
            match_tables ["D-FINE-M"][dfine_point ]
            )

            masks =make_subsets (
            yolo_table ,
            dfine_table ,
            yolo_point_data ["threshold"],
            dfine_point_data ["threshold"],
            yolo_point_data ["macro_f1"],
            dfine_point_data ["macro_f1"],
            )

            combination_sets [combination ]={}

            for subset ,mask in masks .items ():
                ids =set (
                yolo_table ["annotation_id"][mask ].tolist ()
                )
                combination_sets [combination ][subset ]=ids 

                central_ids =central_sets [subset ]
                union =ids |central_ids 
                intersection =ids &central_ids 

                subset_rows .append ({
                "combination":combination ,
                "yolo_point":yolo_point ,
                "yolo_threshold":
                yolo_point_data ["threshold"],
                "dfine_point":dfine_point ,
                "dfine_threshold":
                dfine_point_data ["threshold"],
                "subset":subset ,
                "objects":len (ids ),
                "images":len (set (
                yolo_table ["image_id"][mask ].tolist ()
                )),
                "central_objects":len (central_ids ),
                "intersection_with_central":
                len (intersection ),
                "union_with_central":len (union ),
                "jaccard_against_central":
                len (intersection )/len (union )
                if union else 1.0 ,
                })

                categories =Counter (
                yolo_table ["category_id"][mask ].tolist ()
                )
                scales =Counter (
                yolo_table ["scale"][mask ].tolist ()
                )

                for category_id in sorted (
                set (
                yolo_table [
                "category_id"
                ].tolist ()
                )
                ):
                    class_rows .append ({
                    "combination":combination ,
                    "subset":subset ,
                    "category_id":category_id ,
                    "class_name":str (
                    yolo_table ["class_name"][
                    np .flatnonzero (
                    yolo_table ["category_id"]
                    ==category_id 
                    )[0 ]
                    ]
                    ),
                    "objects":
                    categories [category_id ],
                    "fraction":
                    categories [category_id ]/len (ids )
                    if ids else 0.0 ,
                    })

                for scale in ("small","medium","large"):
                    scale_rows .append ({
                    "combination":combination ,
                    "subset":subset ,
                    "scale":scale ,
                    "objects":scales [scale ],
                    "fraction":
                    scales [scale ]/len (ids )
                    if ids else 0.0 ,
                    })

            print ("PASS:",combination )

    central_combination =(
    "yolo_central__dfine_central"
    )

    for subset in SUBSETS :
        if (
        combination_sets [
        central_combination 
        ][subset ]
        !=central_sets [subset ]
        ):
            raise AssertionError (
            f"Central subset mismatch: {subset }"
            )

    print ("PASS: central memberships reproduce Stage 2 exactly")

    ids_dir =staging /"subset_annotation_ids"
    ids_dir .mkdir ()

    for combination ,subsets in combination_sets .items ():
        write_json (
        ids_dir /f"{combination }.json",
        {
        subset :sorted (ids )
        for subset ,ids in subsets .items ()
        },
        )

    gt_annotations ,class_names =load_gt_annotations ()

    recovery ={}
    print ("\n=== EVALUATING PRIMARY EFFECT DIRECTIONS ===")

    for model ,prediction_path in GATE_PREDICTIONS .items ():
        print ("\nLOADING:",model )
        prediction_groups =load_gate_predictions (
        prediction_path 
        )
        recovery [model ]={}

        for combination ,subsets in combination_sets .items ():
            subset_metrics ={}

            for subset in FAILURE_MACRO_SUBSETS :
                subset_metrics [subset ]=(
                subset_recovery_metrics (
                subsets [subset ],
                gt_annotations ,
                prediction_groups ,
                )
                )

            failure_macro =float (np .mean ([
            subset_metrics [subset ]["AR100"]
            for subset in FAILURE_MACRO_SUBSETS 
            ]))

            recovery [model ][combination ]={
            "failure_macro_AR100":failure_macro ,
            "small_jointly_missed_recall50":
            subset_metrics [
            "small_jointly_missed"
            ]["recall50"],
            "subset_metrics":subset_metrics ,
            }

            print (
            model ,
            combination ,
            "macro_AR100=",
            round (failure_macro ,6 ),
            "small_joint_recall50=",
            round (
            recovery [model ][combination ][
            "small_jointly_missed_recall50"
            ],
            6 ,
            ),
            )

        del prediction_groups 
        gc .collect ()

    direction_rows =[]

    for comparison ,(
    positive_model ,
    reference_model ,
    )in COMPARISONS .items ():
        for metric in (
        "failure_macro_AR100",
        "small_jointly_missed_recall50",
        ):
            central_delta =(
            recovery [positive_model ][
            central_combination 
            ][metric ]
            -recovery [reference_model ][
            central_combination 
            ][metric ]
            )
            central_sign =sign (central_delta )

            for combination in sorted (combination_sets ):
                delta =(
                recovery [positive_model ][
                combination 
                ][metric ]
                -recovery [reference_model ][
                combination 
                ][metric ]
                )

                direction_rows .append ({
                "comparison":comparison ,
                "metric":metric ,
                "combination":combination ,
                "positive_model":positive_model ,
                "reference_model":reference_model ,
                "positive_value":
                recovery [positive_model ][
                combination 
                ][metric ],
                "reference_value":
                recovery [reference_model ][
                combination 
                ][metric ],
                "delta":delta ,
                "sign":sign (delta ),
                "central_delta":central_delta ,
                "central_sign":central_sign ,
                "same_direction_as_central":
                sign (delta )==central_sign ,
                })

    def write_csv (path ,rows ):
        if not rows :
            raise ValueError (path )

        with path .open (
        "w",
        newline ="",
        encoding ="utf-8",
        )as handle :
            writer =csv .DictWriter (
            handle ,
            fieldnames =list (rows [0 ].keys ()),
            )
            writer .writeheader ()
            writer .writerows (rows )

    write_csv (
    staging /"subset_stability.csv",
    subset_rows ,
    )
    write_csv (
    staging /"class_composition.csv",
    class_rows ,
    )
    write_csv (
    staging /"scale_composition.csv",
    scale_rows ,
    )
    write_csv (
    staging /"primary_effect_direction_stability.csv",
    direction_rows ,
    )

    write_json (
    staging /"primary_recovery_metrics.json",
    recovery ,
    )

    stability_summary ={}

    for comparison in COMPARISONS :
        stability_summary [comparison ]={}

        for metric in (
        "failure_macro_AR100",
        "small_jointly_missed_recall50",
        ):
            selected =[
            row for row in direction_rows 
            if row ["comparison"]==comparison 
            and row ["metric"]==metric 
            ]

            stability_summary [comparison ][metric ]={
            "central_delta":
            selected [0 ]["central_delta"],
            "stable_combinations":sum (
            bool (row ["same_direction_as_central"])
            for row in selected 
            ),
            "total_combinations":len (selected ),
            "all_directions_stable":all (
            bool (row ["same_direction_as_central"])
            for row in selected 
            ),
            "minimum_delta":min (
            row ["delta"]for row in selected 
            ),
            "maximum_delta":max (
            row ["delta"]for row in selected 
            ),
            }

    summary ={
    "version":"final_threshold_sensitivity_v1",
    "status":"complete",
    "protocol_sha256":sha256_file (PROTOCOL ),
    "central_failure_manifest_sha256":
    sha256_file (CENTRAL_SUBSET_MANIFEST ),
    "plateau_rule":
    protocol ["threshold_sensitivity"][
    "plateau_rule"
    ],
    "threshold_points":{
    model :data ["points"]
    for model ,data in plateaus .items ()
    },
    "combinations":sorted (combination_sets ),
    "subset_names":list (SUBSETS ),
    "targeted_recovery_comparisons":
    list (COMPARISONS ),
    "effect_direction_stability":
    stability_summary ,
    "architecture_intervention":{
    "comparison":
    "p2_e20_minus_standard_e20",
    "threshold_dependency":False ,
    "reason":(
    "AP_small and AR_small are canonical "
    "COCO metrics and do not depend on the "
    "baseline failure operating thresholds."
    ),
    },
    "interpretation":(
    "Threshold stability supports robustness "
    "of the empirical failure analysis but does "
    "not convert association into causal proof."
    ),
    "elapsed_minutes":
    (time .time ()-started )/60.0 ,
    }

    write_json (
    staging /"summary.json",
    summary ,
    )

    metadata ={
    "version":"final_threshold_sensitivity_v1",
    "status":"complete",
    "created_utc":time .strftime (
    "%Y-%m-%dT%H:%M:%SZ",
    time .gmtime (),
    ),
    "script":str (Path (__file__ ).resolve ()),
    "script_sha256":
    sha256_file (Path (__file__ ).resolve ()),
    "gt":str (GT ),
    "gt_sha256":sha256_file (GT ),
    "match_iou":MATCH_IOU ,
    "localization_iou":LOCALIZATION_IOU ,
    "prediction_confidence_floor":
    CONFIDENCE_FLOOR ,
    "max_detections_per_image_category":
    MAX_DET ,
    "central_membership_reproduced":True ,
    "new_training":False ,
    "new_gpu_inference":False ,
    }

    write_json (
    staging /"metadata.json",
    metadata ,
    )

    manifest_path ,manifest_sha256 =build_manifest (
    staging 
    )

    os .replace (staging ,OUTPUT )

    print ("\n"+"="*100 )
    print ("FINAL THRESHOLD-SENSITIVITY ANALYSIS COMPLETE")
    print ("="*100 )

    for subset in SUBSETS :
        rows =[
        row for row in subset_rows 
        if row ["subset"]==subset 
        ]
        print (
        subset ,
        "count_range=",
        (
        min (row ["objects"]for row in rows ),
        max (row ["objects"]for row in rows ),
        ),
        "jaccard_range=",
        (
        round (
        min (
        row ["jaccard_against_central"]
        for row in rows 
        ),
        6 ,
        ),
        round (
        max (
        row ["jaccard_against_central"]
        for row in rows 
        ),
        6 ,
        ),
        ),
        )

    print (
    json .dumps (
    stability_summary ,
    indent =2 ,
    ensure_ascii =False ,
    sort_keys =True ,
    )
    )

    print ("output:",OUTPUT )
    print (
    "manifest:",
    OUTPUT /manifest_path .name ,
    )
    print ("manifest_sha256:",manifest_sha256 )
    print ("NOTHING TRAINED OR GPU-INFERRED")
    print ("NEXT: training-dynamics attribution")


def main ():
    args =parse_args ()
    protocol ,plateaus =preflight ()

    if args .preflight_only :
        print ("\nPASS: Stage-3 executor preflight only")
        print ("NOTHING CREATED, MATCHED OR EVALUATED")
        return 

    execute (protocol ,plateaus )


if __name__ =="__main__":
    main ()
