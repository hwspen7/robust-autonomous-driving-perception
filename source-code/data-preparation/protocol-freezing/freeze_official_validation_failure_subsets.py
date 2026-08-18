from __future__ import annotations 

import argparse 
import csv 
import hashlib 
import json 
import math 
import os 
import shutil 
import sys 
import time 
from collections import Counter 
from datetime import datetime ,timezone 
from itertools import zip_longest 
from pathlib import Path 

import numpy as np 


RESULTS =Path (
"/root/rivermind-data/autodrive/results/evaluation"
)
PROJECT =Path (
"/root/rivermind-data/autodrive/code/"
"robust-autonomous-driving-perception"
)

MATCHES_DIR =(
RESULTS 
/"final_analysis_v1"
/"official_val_failure_matches_v1"
)
OUTPUT_DIR =(
RESULTS 
/"final_analysis_v1"
/"official_val_failure_subsets_v1"
)

PROTOCOL =(
RESULTS 
/"final_analysis_v1/frozen_protocol/"
"final_testing_protocol_v1.json"
)
CANONICAL_MANIFEST =(
RESULTS 
/"final_analysis_v1/canonical_clean_v1/"
"artifact_manifest.json"
)
GT =Path (
"/root/rivermind-data/autodrive/datasets/"
"datasets/bdd100k_final/coco/annotations/"
"instances_val.json"
)

EXPECTED ={
PROTOCOL :
"693c39375cc1eefc4f3ff9f957a8355fb947552a19dfef228cae6cfc21e3dcfe",

CANONICAL_MANIFEST :
"88d2bace7a2f69e26c34e7eed4e2814b12e8317980602968b407d1b94282338b",

GT :
"7be8a02e147743123a4a62b2c92ed2e4a5712d261c8868223e6325bad6e346d6",

PROJECT /"scripts/build_train_detection_matches.py":
"3a3b5aa6efd5c956b429769fcdf45f0def85ec174c8164e5f3c217d62e6337c2",

PROJECT /"scripts/build_train_risk_table.py":
"e2f8f029da6cb7316f7be9ae49d73f55b11fade9cac925127f5530bba83aa44c",

RESULTS 
/"train_risk_labels/operating_point_calibration/"
"operating_points.json":
"f13868c935ea16091fd0bae592bb9817d77bd58d2f45597dc7d2add8e2fa44ae",
}

EXPECTED_OBJECTS =185_523 
EXPECTED_IMAGES =10_000 
CLASS_PRIOR_STRENGTH =500.0 
NONDETECTION_QUALITY_MULTIPLIER =0.5 

THRESHOLDS ={
"YOLO11m":0.22527144849300385 ,
"D-FINE-M":0.4963708519935608 ,
}
CALIBRATION_MACRO_F1 ={
"YOLO11m":0.6945657943717607 ,
"D-FINE-M":0.7012245282327485 ,
}
CATEGORY_NAMES ={
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
VALID_STATUSES ={
"detected",
"assignment_conflict_candidate",
"low_confidence_candidate",
"classification_error_candidate",
"localization_error_candidate",
"missed",
}
SUBSET_DEFINITIONS ={
"small_jointly_missed":(
"scale == small and YOLO11m detected == 0 "
"and D-FINE-M detected == 0"
),
"dfine_supported_yolo_failure":(
"D-FINE-M detected == 1 and YOLO11m detected == 0"
),
"yolo_low_confidence":(
"YOLO11m status == low_confidence_candidate"
),
"yolo_localization_failure":(
"YOLO11m status == localization_error_candidate"
),
"yolo_classification_failure":(
"YOLO11m status == classification_error_candidate"
),
"small_high_risk":(
"scale == small and frozen official-val "
"risk_score >= 0.75"
),
}


def sha256 (path :Path )->str :
    digest =hashlib .sha256 ()
    with path .open ("rb")as handle :
        while True :
            block =handle .read (8 *1024 *1024 )
            if not block :
                break 
            digest .update (block )
    return digest .hexdigest ()


def verify_file (path :Path ,expected :str )->None :
    if not path .is_file ():
        raise FileNotFoundError (path )

    actual =sha256 (path )

    if actual !=expected :
        raise AssertionError (
        f"SHA256 mismatch\n"
        f"path: {path }\n"
        f"expected: {expected }\n"
        f"actual:   {actual }"
        )


def write_json (path :Path ,payload )->None :
    path .parent .mkdir (parents =True ,exist_ok =True )
    temporary =path .with_suffix (path .suffix +".tmp")

    temporary .write_text (
    json .dumps (
    payload ,
    indent =2 ,
    ensure_ascii =False ,
    sort_keys =True ,
    )+"\n",
    encoding ="utf-8",
    )

    os .replace (temporary ,path )


def optional_float (value :str |None )->float :
    if value is None or value =="":
        return 0.0 

    result =float (value )

    if not math .isfinite (result ):
        raise ValueError (value )

    return result 


def confidence_support (
score :float ,
threshold :float ,
)->float :
    score =min (1.0 ,max (0.0 ,score ))

    if score <=threshold :
        if threshold <=0 :
            return 0.5 
        return 0.5 *score /threshold 

    if threshold >=1 :
        return 0.5 

    return (
    0.5 
    +0.5 
    *(score -threshold )
    /(1.0 -threshold )
    )


def detector_quality (
row :dict [str ,str ],
threshold :float ,
)->float :
    detected =int (row ["detected"])

    if detected :
        score =optional_float (
        row ["matched_score"]
        )
        iou =optional_float (
        row ["matched_iou"]
        )
        multiplier =1.0 
    else :
        score =optional_float (
        row ["best_same_all_score"]
        )
        iou =optional_float (
        row ["best_same_all_iou"]
        )
        multiplier =(
        NONDETECTION_QUALITY_MULTIPLIER 
        )

    quality =multiplier *math .sqrt (
    confidence_support (
    score ,
    threshold ,
    )
    *min (1.0 ,max (0.0 ,iou ))
    )

    return min (1.0 ,max (0.0 ,quality ))


def percentile_rank (
values :np .ndarray ,
)->np .ndarray :
    count =len (values )

    if count ==0 :
        return np .empty (
        0 ,
        dtype =np .float32 ,
        )

    if count ==1 :
        return np .asarray (
        [0.5 ],
        dtype =np .float32 ,
        )

    order =np .argsort (
    values ,
    kind ="mergesort",
    )
    sorted_values =values [order ]

    _ ,starts ,counts =np .unique (
    sorted_values ,
    return_index =True ,
    return_counts =True ,
    )

    average_positions =(
    starts .astype (np .float64 )
    +(
    counts .astype (np .float64 )
    -1.0 
    )
    /2.0 
    )/float (count -1 )

    sorted_ranks =np .repeat (
    average_positions ,
    counts ,
    ).astype (np .float32 )

    ranks =np .empty (
    count ,
    dtype =np .float32 ,
    )
    ranks [order ]=sorted_ranks 

    return ranks 


def risk_tier (score :float )->str :
    if score >=0.90 :
        return "critical"
    if score >=0.75 :
        return "high"
    if score >=0.50 :
        return "medium"
    return "standard"


def validate_match_manifest ()->tuple [dict ,str ]:
    metadata_path =MATCHES_DIR /"metadata.json"
    manifest_path =(
    MATCHES_DIR /"artifact_manifest.json"
    )

    if not metadata_path .is_file ():
        raise FileNotFoundError (metadata_path )

    if not manifest_path .is_file ():
        raise FileNotFoundError (manifest_path )

    metadata =json .loads (
    metadata_path .read_text ()
    )
    manifest =json .loads (
    manifest_path .read_text ()
    )

    if metadata ["gt_sha256"]!=EXPECTED [GT ]:
        raise AssertionError (
        metadata ["gt_sha256"]
        )

    if int (metadata ["num_images"])!=EXPECTED_IMAGES :
        raise AssertionError (
        metadata ["num_images"]
        )

    if (
    int (metadata ["num_gt_objects"])
    !=EXPECTED_OBJECTS 
    ):
        raise AssertionError (
        metadata ["num_gt_objects"]
        )

    if (
    float (metadata ["match_iou"])
    !=0.5 
    ):
        raise AssertionError (
        metadata ["match_iou"]
        )

    if (
    float (metadata ["localization_iou"])
    !=0.1 
    ):
        raise AssertionError (
        metadata ["localization_iou"]
        )

    for model ,threshold in THRESHOLDS .items ():
        actual =float (
        metadata ["thresholds"][model ]
        )

        if not math .isclose (
        actual ,
        threshold ,
        rel_tol =0.0 ,
        abs_tol =1e-15 ,
        ):
            raise AssertionError (
            (model ,actual ,threshold )
            )

    if (
    metadata ["script_sha256"]
    !=EXPECTED [
    PROJECT 
    /"scripts/build_train_detection_matches.py"
    ]
    ):
        raise AssertionError (
        metadata ["script_sha256"]
        )

    artifacts =manifest .get (
    "artifacts",
    []
    )

    if not artifacts :
        raise AssertionError (
        "Empty match artifact manifest"
        )

    for record in artifacts :
        relative =Path (record ["path"])

        if relative .is_absolute ():
            path =relative 
        else :
            path =MATCHES_DIR /relative 

        if not path .is_file ():
            raise FileNotFoundError (path )

        actual_hash =sha256 (path )

        if (
        actual_hash 
        !=record ["sha256"]
        ):
            raise AssertionError (
            (
            path ,
            record ["sha256"],
            actual_hash ,
            )
            )

        if (
        int (record ["bytes"])
        !=path .stat ().st_size 
        ):
            raise AssertionError (
            path 
            )

    return metadata ,sha256 (manifest_path )


def csv_row_count (path :Path )->int :
    with path .open (
    "r",
    newline ="",
    encoding ="utf-8",
    )as handle :
        return sum (1 for _ in handle )-1 


def preflight ()->tuple [dict ,str ]:
    print (
    "=== OFFICIAL-VAL FAILURE SUBSET PREFLIGHT ==="
    )

    if OUTPUT_DIR .exists ():
        raise FileExistsError (
        f"Refusing to overwrite: {OUTPUT_DIR }"
        )

    for path ,expected in EXPECTED .items ():
        verify_file (
        path ,
        expected ,
        )
        print ("PASS:",path )

    metadata ,manifest_hash =(
    validate_match_manifest ()
    )

    yolo_path =(
    MATCHES_DIR 
    /"YOLO11m"
    /"object_matches.csv"
    )
    dfine_path =(
    MATCHES_DIR 
    /"D-FINE-M"
    /"object_matches.csv"
    )

    required_columns ={
    "annotation_id",
    "image_id",
    "category_id",
    "category",
    "gt_x",
    "gt_y",
    "gt_width",
    "gt_height",
    "gt_area",
    "scale",
    "timeofday",
    "weather",
    "scene",
    "status",
    "detected",
    "matched_iou",
    "matched_score",
    "best_same_all_iou",
    "best_same_all_score",
    }

    for path in (yolo_path ,dfine_path ):
        if not path .is_file ():
            raise FileNotFoundError (path )

        with path .open (
        "r",
        newline ="",
        encoding ="utf-8",
        )as handle :
            header =set (
            next (csv .reader (handle ))
            )

        missing =(
        required_columns -header 
        )

        if missing :
            raise AssertionError (
            (path ,sorted (missing ))
            )

        rows =csv_row_count (path )

        print (
        path ,
        "rows=",
        rows ,
        )

        if rows !=EXPECTED_OBJECTS :
            raise AssertionError (
            (path ,rows )
            )

    usage =shutil .disk_usage (
    "/root/rivermind-data"
    )

    print (
    "free_GiB:",
    round (
    usage .free /1024 **3 ,
    3 ,
    ),
    )
    print (
    "match_manifest_sha256:",
    manifest_hash ,
    )
    print (
    "PASS: official-val matches are valid"
    )

    return metadata ,manifest_hash 


def distribution (
values :np .ndarray ,
)->dict :
    if len (values )==0 :
        return {"count":0 }

    return {
    "count":int (len (values )),
    "mean":float (np .mean (values )),
    "std":float (np .std (values )),
    "min":float (np .min (values )),
    "p25":float (
    np .quantile (values ,0.25 )
    ),
    "p50":float (
    np .quantile (values ,0.50 )
    ),
    "p75":float (
    np .quantile (values ,0.75 )
    ),
    "p90":float (
    np .quantile (values ,0.90 )
    ),
    "max":float (np .max (values )),
    }


def freeze ()->None :
    match_metadata ,match_manifest_hash =(
    preflight ()
    )

    started =time .time ()

    yolo_path =(
    MATCHES_DIR 
    /"YOLO11m"
    /"object_matches.csv"
    )
    dfine_path =(
    MATCHES_DIR 
    /"D-FINE-M"
    /"object_matches.csv"
    )

    reliability_sum =sum (
    CALIBRATION_MACRO_F1 .values ()
    )
    reliability_weights ={
    model :(
    CALIBRATION_MACRO_F1 [model ]
    /reliability_sum 
    )
    for model in CALIBRATION_MACRO_F1 
    }

    annotation_ids =np .empty (
    EXPECTED_OBJECTS ,
    dtype =np .int64 ,
    )
    image_ids =np .empty (
    EXPECTED_OBJECTS ,
    dtype =np .int64 ,
    )
    category_ids =np .empty (
    EXPECTED_OBJECTS ,
    dtype =np .uint8 ,
    )
    scales =np .empty (
    EXPECTED_OBJECTS ,
    dtype ="U6",
    )
    yolo_detected =np .empty (
    EXPECTED_OBJECTS ,
    dtype =np .uint8 ,
    )
    dfine_detected =np .empty (
    EXPECTED_OBJECTS ,
    dtype =np .uint8 ,
    )
    yolo_status =np .empty (
    EXPECTED_OBJECTS ,
    dtype ="U40",
    )
    dfine_status =np .empty (
    EXPECTED_OBJECTS ,
    dtype ="U40",
    )
    yolo_quality =np .empty (
    EXPECTED_OBJECTS ,
    dtype =np .float32 ,
    )
    dfine_quality =np .empty (
    EXPECTED_OBJECTS ,
    dtype =np .float32 ,
    )
    joint_raw_risk =np .empty (
    EXPECTED_OBJECTS ,
    dtype =np .float32 ,
    )

    print (
    "\n=== FUSING OFFICIAL-VAL OBJECTS ==="
    )

    with (
    yolo_path .open (
    "r",
    newline ="",
    encoding ="utf-8",
    )as yolo_handle ,
    dfine_path .open (
    "r",
    newline ="",
    encoding ="utf-8",
    )as dfine_handle ,
    ):
        yolo_reader =csv .DictReader (
        yolo_handle 
        )
        dfine_reader =csv .DictReader (
        dfine_handle 
        )

        count =0 

        for index ,pair in enumerate (
        zip_longest (
        yolo_reader ,
        dfine_reader ,
        )
        ):
            yolo_row ,dfine_row =pair 

            if (
            yolo_row is None 
            or dfine_row is None 
            ):
                raise RuntimeError (
                "Teacher CSV lengths differ"
                )

            for field in (
            "annotation_id",
            "image_id",
            "category_id",
            "category",
            "scale",
            "gt_area",
            ):
                if (
                yolo_row [field ]
                !=dfine_row [field ]
                ):
                    raise RuntimeError (
                    (
                    index ,
                    field ,
                    yolo_row [field ],
                    dfine_row [field ],
                    )
                    )

            y_status =yolo_row ["status"]
            d_status =dfine_row ["status"]

            if y_status not in VALID_STATUSES :
                raise ValueError (y_status )

            if d_status not in VALID_STATUSES :
                raise ValueError (d_status )

            category_id =int (
            yolo_row ["category_id"]
            )

            if category_id not in CATEGORY_NAMES :
                raise ValueError (category_id )

            scale =yolo_row ["scale"]

            if scale not in {
            "small",
            "medium",
            "large",
            }:
                raise ValueError (scale )

            y_quality =detector_quality (
            yolo_row ,
            THRESHOLDS ["YOLO11m"],
            )
            d_quality =detector_quality (
            dfine_row ,
            THRESHOLDS ["D-FINE-M"],
            )

            y_risk =1.0 -y_quality 
            d_risk =1.0 -d_quality 

            consensus =(
            reliability_weights ["YOLO11m"]
            *y_risk 
            +reliability_weights ["D-FINE-M"]
            *d_risk 
            )
            disagreement =abs (
            y_risk -d_risk 
            )
            joint =(
            1.0 
            -(1.0 -consensus )
            *(1.0 -disagreement )
            )

            annotation_ids [index ]=int (
            yolo_row ["annotation_id"]
            )
            image_ids [index ]=int (
            yolo_row ["image_id"]
            )
            category_ids [index ]=category_id 
            scales [index ]=scale 
            yolo_detected [index ]=int (
            yolo_row ["detected"]
            )
            dfine_detected [index ]=int (
            dfine_row ["detected"]
            )
            yolo_status [index ]=y_status 
            dfine_status [index ]=d_status 
            yolo_quality [index ]=y_quality 
            dfine_quality [index ]=d_quality 
            joint_raw_risk [index ]=joint 

            count +=1 

        if count !=EXPECTED_OBJECTS :
            raise AssertionError (count )

    if (
    len (np .unique (annotation_ids ))
    !=EXPECTED_OBJECTS 
    ):
        raise AssertionError (
        "Duplicate annotation IDs"
        )

    print (
    "objects:",
    EXPECTED_OBJECTS ,
    )

    global_percentile =percentile_rank (
    joint_raw_risk 
    )
    class_percentile =np .empty (
    EXPECTED_OBJECTS ,
    dtype =np .float32 ,
    )
    class_reliability =np .empty (
    EXPECTED_OBJECTS ,
    dtype =np .float32 ,
    )

    for category_id in CATEGORY_NAMES :
        positions =np .flatnonzero (
        category_ids ==category_id 
        )
        class_count =len (positions )

        class_percentile [positions ]=(
        percentile_rank (
        joint_raw_risk [positions ]
        )
        )
        class_reliability [positions ]=(
        class_count 
        /(
        class_count 
        +CLASS_PRIOR_STRENGTH 
        )
        )

    risk_score =(
    class_reliability 
    *class_percentile 
    +(
    1.0 -class_reliability 
    )
    *global_percentile 
    ).astype (np .float32 )

    masks ={
    "small_jointly_missed":(
    (scales =="small")
    &(yolo_detected ==0 )
    &(dfine_detected ==0 )
    ),
    "dfine_supported_yolo_failure":(
    (dfine_detected ==1 )
    &(yolo_detected ==0 )
    ),
    "yolo_low_confidence":(
    yolo_status 
    =="low_confidence_candidate"
    ),
    "yolo_localization_failure":(
    yolo_status 
    =="localization_error_candidate"
    ),
    "yolo_classification_failure":(
    yolo_status 
    =="classification_error_candidate"
    ),
    "small_high_risk":(
    (scales =="small")
    &(risk_score >=0.75 )
    ),
    }

    OUTPUT_DIR .parent .mkdir (
    parents =True ,
    exist_ok =True ,
    )
    staging =OUTPUT_DIR .with_name (
    OUTPUT_DIR .name 
    +f".incomplete-{os .getpid ()}"
    )

    if staging .exists ():
        raise FileExistsError (staging )

    staging .mkdir ()

    try :
        created_utc =datetime .now (
        timezone .utc 
        ).isoformat ()

        summary_rows =[]
        subset_payloads ={}

        print (
        "\n=== FREEZING FAILURE SUBSETS ==="
        )

        for name ,mask in masks .items ():
            positions =np .flatnonzero (mask )
            subset_annotation_ids =sorted (
            int (value )
            for value in annotation_ids [
            positions 
            ]
            )
            subset_image_ids =sorted (
            {
            int (value )
            for value in image_ids [
            positions 
            ]
            }
            )

            class_counts =Counter (
            int (value )
            for value in category_ids [
            positions 
            ]
            )
            scale_counts =Counter (
            str (value )
            for value in scales [
            positions 
            ]
            )

            payload ={
            "version":(
            "official_val_failure_subset_v1"
            ),
            "status":"frozen",
            "created_utc":created_utc ,
            "name":name ,
            "definition":(
            SUBSET_DEFINITIONS [name ]
            ),
            "source_split":(
            "BDD100K official validation"
            ),
            "num_objects":len (positions ),
            "num_images":len (
            subset_image_ids 
            ),
            "annotation_ids":(
            subset_annotation_ids 
            ),
            "image_ids":subset_image_ids ,
            "class_counts":{
            str (category_id ):{
            "class_name":(
            CATEGORY_NAMES [
            category_id 
            ]
            ),
            "count":int (
            class_counts .get (
            category_id ,
            0 ,
            )
            ),
            }
            for category_id 
            in CATEGORY_NAMES 
            },
            "scale_counts":{
            scale :int (
            scale_counts .get (
            scale ,
            0 ,
            )
            )
            for scale in (
            "small",
            "medium",
            "large",
            )
            },
            "risk_distribution":(
            distribution (
            risk_score [positions ]
            )
            ),
            "protocol_sha256":(
            EXPECTED [PROTOCOL ]
            ),
            "match_manifest_sha256":(
            match_manifest_hash 
            ),
            }

            subset_payloads [name ]=payload 

            write_json (
            staging 
            /"subsets"
            /f"{name }.json",
            payload ,
            )

            summary_rows .append (
            {
            "subset":name ,
            "num_objects":len (
            positions 
            ),
            "num_images":len (
            subset_image_ids 
            ),
            "mean_risk":(
            float (
            np .mean (
            risk_score [
            positions 
            ]
            )
            )
            if len (positions )
            else float ("nan")
            ),
            }
            )

            print (
            f"{name :35s} "
            f"objects={len (positions ):7d} "
            f"images={len (subset_image_ids ):5d}"
            )

        overlap_rows =[]
        names =list (masks )

        for left in names :
            for right in names :
                intersection =int (
                np .count_nonzero (
                masks [left ]
                &masks [right ]
                )
                )
                union =int (
                np .count_nonzero (
                masks [left ]
                |masks [right ]
                )
                )

                overlap_rows .append (
                {
                "left_subset":left ,
                "right_subset":right ,
                "intersection":(
                intersection 
                ),
                "union":union ,
                "jaccard":(
                intersection /union 
                if union 
                else 1.0 
                ),
                }
                )

        summary_path =(
        staging /"subset_summary.csv"
        )

        with summary_path .open (
        "w",
        newline ="",
        encoding ="utf-8",
        )as handle :
            writer =csv .DictWriter (
            handle ,
            fieldnames =[
            "subset",
            "num_objects",
            "num_images",
            "mean_risk",
            ],
            )
            writer .writeheader ()
            writer .writerows (
            summary_rows 
            )

        overlap_path =(
        staging /"subset_overlap.csv"
        )

        with overlap_path .open (
        "w",
        newline ="",
        encoding ="utf-8",
        )as handle :
            writer =csv .DictWriter (
            handle ,
            fieldnames =[
            "left_subset",
            "right_subset",
            "intersection",
            "union",
            "jaccard",
            ],
            )
            writer .writeheader ()
            writer .writerows (
            overlap_rows 
            )

        table_path =(
        staging /"failure_object_table.csv"
        )

        table_fields =[
        "annotation_id",
        "image_id",
        "category_id",
        "class_name",
        "scale",
        "gt_x",
        "gt_y",
        "gt_width",
        "gt_height",
        "gt_area",
        "timeofday",
        "weather",
        "scene",
        "yolo_status",
        "yolo_detected",
        "yolo_quality",
        "yolo_risk",
        "dfine_status",
        "dfine_detected",
        "dfine_quality",
        "dfine_risk",
        "consensus_risk",
        "teacher_disagreement",
        "joint_raw_risk",
        "global_risk_percentile",
        "class_risk_percentile",
        "class_reliability",
        "risk_score",
        "risk_tier",
        *[
        f"in_{name }"
        for name in names 
        ],
        ]

        with (
        yolo_path .open (
        "r",
        newline ="",
        encoding ="utf-8",
        )as yolo_handle ,
        dfine_path .open (
        "r",
        newline ="",
        encoding ="utf-8",
        )as dfine_handle ,
        table_path .open (
        "w",
        newline ="",
        encoding ="utf-8",
        )as output_handle ,
        ):
            yolo_reader =csv .DictReader (
            yolo_handle 
            )
            dfine_reader =csv .DictReader (
            dfine_handle 
            )
            writer =csv .DictWriter (
            output_handle ,
            fieldnames =table_fields ,
            )
            writer .writeheader ()

            for index ,pair in enumerate (
            zip_longest (
            yolo_reader ,
            dfine_reader ,
            )
            ):
                yolo_row ,dfine_row =pair 

                y_quality =float (
                yolo_quality [index ]
                )
                d_quality =float (
                dfine_quality [index ]
                )
                y_risk =1.0 -y_quality 
                d_risk =1.0 -d_quality 
                consensus =(
                reliability_weights [
                "YOLO11m"
                ]
                *y_risk 
                +reliability_weights [
                "D-FINE-M"
                ]
                *d_risk 
                )
                disagreement =abs (
                y_risk -d_risk 
                )

                row ={
                "annotation_id":(
                yolo_row [
                "annotation_id"
                ]
                ),
                "image_id":(
                yolo_row ["image_id"]
                ),
                "category_id":(
                yolo_row [
                "category_id"
                ]
                ),
                "class_name":(
                yolo_row ["category"]
                ),
                "scale":(
                yolo_row ["scale"]
                ),
                "gt_x":yolo_row ["gt_x"],
                "gt_y":yolo_row ["gt_y"],
                "gt_width":(
                yolo_row ["gt_width"]
                ),
                "gt_height":(
                yolo_row ["gt_height"]
                ),
                "gt_area":(
                yolo_row ["gt_area"]
                ),
                "timeofday":(
                yolo_row ["timeofday"]
                ),
                "weather":(
                yolo_row ["weather"]
                ),
                "scene":(
                yolo_row ["scene"]
                ),
                "yolo_status":(
                yolo_row ["status"]
                ),
                "yolo_detected":(
                yolo_row ["detected"]
                ),
                "yolo_quality":(
                f"{y_quality :.10g}"
                ),
                "yolo_risk":(
                f"{y_risk :.10g}"
                ),
                "dfine_status":(
                dfine_row ["status"]
                ),
                "dfine_detected":(
                dfine_row ["detected"]
                ),
                "dfine_quality":(
                f"{d_quality :.10g}"
                ),
                "dfine_risk":(
                f"{d_risk :.10g}"
                ),
                "consensus_risk":(
                f"{consensus :.10g}"
                ),
                "teacher_disagreement":(
                f"{disagreement :.10g}"
                ),
                "joint_raw_risk":(
                f"{float (joint_raw_risk [index ]):.10g}"
                ),
                "global_risk_percentile":(
                f"{float (global_percentile [index ]):.10g}"
                ),
                "class_risk_percentile":(
                f"{float (class_percentile [index ]):.10g}"
                ),
                "class_reliability":(
                f"{float (class_reliability [index ]):.10g}"
                ),
                "risk_score":(
                f"{float (risk_score [index ]):.10g}"
                ),
                "risk_tier":(
                risk_tier (
                float (
                risk_score [
                index 
                ]
                )
                )
                ),
                }

                for name in names :
                    row [f"in_{name }"]=int (
                    masks [name ][index ]
                    )

                writer .writerow (row )

        metadata ={
        "version":(
        "official_val_failure_subsets_v1"
        ),
        "status":"frozen",
        "created_utc":created_utc ,
        "source_split":(
        "BDD100K official validation"
        ),
        "num_images":EXPECTED_IMAGES ,
        "num_objects":EXPECTED_OBJECTS ,
        "protocol_sha256":(
        EXPECTED [PROTOCOL ]
        ),
        "canonical_manifest_sha256":(
        EXPECTED [
        CANONICAL_MANIFEST 
        ]
        ),
        "matches_dir":str (
        MATCHES_DIR 
        ),
        "matches_manifest_sha256":(
        match_manifest_hash 
        ),
        "matches_metadata_sha256":(
        sha256 (
        MATCHES_DIR 
        /"metadata.json"
        )
        ),
        "matching":{
        "match_iou":0.5 ,
        "localization_iou":0.1 ,
        "thresholds":THRESHOLDS ,
        },
        "risk_method":{
        "method":(
        "CAFR object-risk fusion v1"
        ),
        "formula_source_sha256":(
        EXPECTED [
        PROJECT 
        /"scripts/"
        "build_train_risk_table.py"
        ]
        ),
        "calibration_macro_f1":(
        CALIBRATION_MACRO_F1 
        ),
        "teacher_reliability_weights":(
        reliability_weights 
        ),
        "class_prior_strength":(
        CLASS_PRIOR_STRENGTH 
        ),
        "non_detection_quality_multiplier":(
        NONDETECTION_QUALITY_MULTIPLIER 
        ),
        "risk_tiers":{
        "standard":"[0.00, 0.50)",
        "medium":"[0.50, 0.75)",
        "high":"[0.75, 0.90)",
        "critical":"[0.90, 1.00]",
        },
        },
        "subset_definitions":(
        SUBSET_DEFINITIONS 
        ),
        "subset_counts":{
        name :{
        "objects":int (
        np .count_nonzero (
        mask 
        )
        ),
        "images":len (
        {
        int (value )
        for value 
        in image_ids [mask ]
        }
        ),
        }
        for name ,mask 
        in masks .items ()
        },
        "elapsed_minutes":(
        time .time ()-started 
        )/60.0 ,
        }

        write_json (
        staging /"metadata.json",
        metadata ,
        )

        files ={}

        for path in sorted (
        staging .rglob ("*")
        ):
            if (
            path .is_file ()
            and path .name 
            !="artifact_manifest.json"
            ):
                relative =str (
                path .relative_to (
                staging 
                )
                )
                files [relative ]={
                "bytes":(
                path .stat ().st_size 
                ),
                "sha256":sha256 (path ),
                }

        manifest ={
        "version":(
        "official_val_failure_subsets_"
        "artifact_manifest_v1"
        ),
        "status":"frozen",
        "created_utc":created_utc ,
        "protocol_sha256":(
        EXPECTED [PROTOCOL ]
        ),
        "files":files ,
        }

        write_json (
        staging 
        /"artifact_manifest.json",
        manifest ,
        )

        os .replace (
        staging ,
        OUTPUT_DIR ,
        )

    except Exception :
        if staging .exists ():
            shutil .rmtree (staging )
        raise 

    print ("\n"+"="*100 )
    print (
    "OFFICIAL-VAL FAILURE SUBSETS FROZEN"
    )
    print ("="*100 )

    for row in summary_rows :
        print (
        f"{row ['subset']:35s} "
        f"objects={row ['num_objects']:7d} "
        f"images={row ['num_images']:5d} "
        f"mean_risk={row ['mean_risk']:.6f}"
        )

    manifest_path =(
    OUTPUT_DIR /"artifact_manifest.json"
    )

    print ("output:",OUTPUT_DIR )
    print (
    "manifest:",
    manifest_path ,
    )
    print (
    "manifest_sha256:",
    sha256 (manifest_path ),
    )
    print (
    "PASS: all six official-val "
    "failure subsets are immutable"
    )
    print ("NOTHING TRAINED OR INFERRED")
    print (
    "NEXT: threshold-sensitivity analysis"
    )
    print ("="*100 )


def main ()->None :
    parser =argparse .ArgumentParser ()

    group =(
    parser .add_mutually_exclusive_group (
    required =True 
    )
    )
    group .add_argument (
    "--preflight-only",
    action ="store_true",
    )
    group .add_argument (
    "--freeze",
    action ="store_true",
    )

    args =parser .parse_args ()

    if args .preflight_only :
        preflight ()
        print (
        "NOTHING CREATED, MODIFIED, "
        "FUSED OR TRAINED"
        )
        return 

    freeze ()


if __name__ =="__main__":
    main ()
