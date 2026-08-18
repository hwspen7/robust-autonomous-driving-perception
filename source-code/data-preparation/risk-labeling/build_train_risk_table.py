from __future__ import annotations 

import argparse 
import csv 
import hashlib 
import json 
import math 
import os 
import platform 
import sys 
import time 
from collections import Counter 
from datetime import datetime ,timezone 
from itertools import zip_longest 
from pathlib import Path 

import numpy as np 


METHOD_NAME ="CAFR object-risk fusion v1"
MODELS =("YOLO11m","D-FINE-M")
EXPECTED_THRESHOLDS ={
"YOLO11m":0.22527144849300385 ,
"D-FINE-M":0.4963708519935608 ,
}
CALIBRATION_MACRO_F1 ={
"YOLO11m":0.6945657943717607 ,
"D-FINE-M":0.7012245282327485 ,
}
EXPECTED_CALIBRATION_SHA256 =(
"f13868c935ea16091fd0bae592bb9817d77bd58d2f45597dc7d2add8e2fa44ae"
)
EXPECTED_SPLIT_HASHES ={
"train_dev_image_ids.json":(
"961ddf514525223aa08599774c167ffcf81752551cdc6619fc6febb1184ada8c"
),
"train_core_image_ids.json":(
"9996d854b06ddefd12dde1949ecae30c6d08230eb10003e7ec1863a3386ee1a3"
),
"split_metadata.json":(
"cfd4c2cc92dd0206d106431dc0c4a9b682b86b88e9ab5ce2862559a0add5fa89"
),
}
EXPECTED_IMAGES =70_000 
EXPECTED_OBJECTS =1_286_852 
EXPECTED_DEV_IMAGES =5_000 
EXPECTED_CORE_IMAGES =65_000 
NONDETECTION_QUALITY_MULTIPLIER =0.5 
CLASS_PRIOR_STRENGTH =500.0 

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
SCALE_TO_CODE ={"small":0 ,"medium":1 ,"large":2 }
CODE_TO_SCALE ={value :key for key ,value in SCALE_TO_CODE .items ()}
SPLIT_TO_CODE ={"train_core":0 ,"train_dev":1 }
CODE_TO_SPLIT ={value :key for key ,value in SPLIT_TO_CODE .items ()}

STATUS_NAMES =(
"detected",
"assignment_conflict_candidate",
"low_confidence_candidate",
"classification_error_candidate",
"localization_error_candidate",
"missed",
)
STATUS_TO_CODE ={name :index for index ,name in enumerate (STATUS_NAMES )}


def parse_args ():
    parser =argparse .ArgumentParser (
    description =(
    "Fuse audited YOLO11m and D-FINE-M train matches into a frozen "
    "object-risk table for CAFR sampling."
    )
    )
    parser .add_argument ("--matches-dir",required =True )
    parser .add_argument ("--split-dir",required =True )
    parser .add_argument ("--output-dir",required =True )
    return parser .parse_args ()


def sha256_file (path :Path ,chunk_size :int =8 *1024 *1024 )->str :
    digest =hashlib .sha256 ()
    with path .open ("rb")as handle :
        while True :
            block =handle .read (chunk_size )
            if not block :
                break 
            digest .update (block )
    return digest .hexdigest ()


def write_json (path :Path ,payload )->None :
    temporary =path .with_suffix (path .suffix +".tmp")
    with temporary .open ("w",encoding ="utf-8")as handle :
        json .dump (
        payload ,
        handle ,
        ensure_ascii =False ,
        indent =2 ,
        sort_keys =True ,
        )
        handle .write ("\n")
    os .replace (temporary ,path )


def validate_input_manifest (matches_dir :Path )->tuple [dict ,str ]:
    metadata_path =matches_dir /"metadata.json"
    manifest_path =matches_dir /"artifact_manifest.json"

    if not metadata_path .is_file ():
        raise FileNotFoundError (metadata_path )
    if not manifest_path .is_file ():
        raise FileNotFoundError (manifest_path )

    metadata =json .loads (metadata_path .read_text (encoding ="utf-8"))
    manifest =json .loads (manifest_path .read_text (encoding ="utf-8"))

    if metadata .get ("dataset")!="BDD100K":
        raise ValueError ("Unexpected match dataset.")
    if metadata .get ("split")!="official_training":
        raise ValueError ("Expected official_training matches.")
    if int (metadata .get ("num_images",-1 ))!=EXPECTED_IMAGES :
        raise ValueError ("Unexpected match image count.")
    if int (metadata .get ("num_gt_objects",-1 ))!=EXPECTED_OBJECTS :
        raise ValueError ("Unexpected match object count.")
    if metadata .get ("calibration_sha256")!=EXPECTED_CALIBRATION_SHA256 :
        raise ValueError ("Unexpected operating-point calibration hash.")

    for model in MODELS :
        threshold =float (metadata ["thresholds"][model ])
        if abs (threshold -EXPECTED_THRESHOLDS [model ])>1e-12 :
            raise ValueError (f"Unexpected threshold for {model }.")

    print ("=== VERIFYING MATCH ARTIFACT MANIFEST ===",flush =True )
    artifact_paths =set ()

    for item in manifest .get ("artifacts",[]):
        relative_path =str (item ["path"])
        artifact_path =matches_dir /relative_path 
        artifact_paths .add (relative_path )

        if not artifact_path .is_file ():
            raise FileNotFoundError (artifact_path )
        if artifact_path .stat ().st_size !=int (item ["bytes"]):
            raise ValueError (f"Size mismatch: {relative_path }")
        if sha256_file (artifact_path )!=item ["sha256"]:
            raise ValueError (f"SHA256 mismatch: {relative_path }")

    required ={
    "metadata.json",
    "YOLO11m/object_matches.csv",
    "YOLO11m/false_positives.csv",
    "YOLO11m/image_summary.csv",
    "YOLO11m/summary.json",
    "D-FINE-M/object_matches.csv",
    "D-FINE-M/false_positives.csv",
    "D-FINE-M/image_summary.csv",
    "D-FINE-M/summary.json",
    }
    missing =required -artifact_paths 
    if missing :
        raise ValueError (f"Required match artifacts absent: {sorted (missing )}")

    manifest_sha256 =sha256_file (manifest_path )
    print ("PASS: match artifact manifest",flush =True )
    print ("match_manifest_sha256:",manifest_sha256 ,flush =True )
    return metadata ,manifest_sha256 


def extract_image_ids (payload ,path :Path )->set [int ]:
    if isinstance (payload ,list ):
        values =payload 
    elif isinstance (payload ,dict ):
        preferred_keys =(
        "image_ids",
        "ids",
        "train_dev_image_ids",
        "train_core_image_ids",
        )
        values =None 
        for key in preferred_keys :
            candidate =payload .get (key )
            if isinstance (candidate ,list ):
                values =candidate 
                break 

        if values is None :
            candidates =[
            value for value in payload .values ()if isinstance (value ,list )
            ]
            if len (candidates )!=1 :
                raise ValueError (f"Cannot identify image ID list in {path }")
            values =candidates [0 ]
    else :
        raise TypeError (f"Unsupported split JSON in {path }")

    result ={int (value )for value in values }
    if len (result )!=len (values ):
        raise ValueError (f"Duplicate image IDs in {path }")
    return result 


def load_frozen_split (split_dir :Path )->tuple [set [int ],set [int ],dict ]:
    print ("=== VERIFYING FROZEN CALIBRATION SPLIT ===",flush =True )

    for filename ,expected_hash in EXPECTED_SPLIT_HASHES .items ():
        path =split_dir /filename 
        if not path .is_file ():
            raise FileNotFoundError (path )
        actual_hash =sha256_file (path )
        if actual_hash !=expected_hash :
            raise ValueError (
            f"Frozen split SHA256 mismatch for {filename }: "
            f"{actual_hash }"
            )

    dev_path =split_dir /"train_dev_image_ids.json"
    core_path =split_dir /"train_core_image_ids.json"
    metadata_path =split_dir /"split_metadata.json"

    dev_ids =extract_image_ids (
    json .loads (dev_path .read_text (encoding ="utf-8")),
    dev_path ,
    )
    core_ids =extract_image_ids (
    json .loads (core_path .read_text (encoding ="utf-8")),
    core_path ,
    )
    split_metadata =json .loads (metadata_path .read_text (encoding ="utf-8"))

    if len (dev_ids )!=EXPECTED_DEV_IMAGES :
        raise ValueError (f"Unexpected train_dev size: {len (dev_ids )}")
    if len (core_ids )!=EXPECTED_CORE_IMAGES :
        raise ValueError (f"Unexpected train_core size: {len (core_ids )}")
    if dev_ids &core_ids :
        raise ValueError ("train_dev and train_core overlap.")
    if len (dev_ids |core_ids )!=EXPECTED_IMAGES :
        raise ValueError ("Frozen split does not cover 70k images.")

    print ("train_dev:",len (dev_ids ),flush =True )
    print ("train_core:",len (core_ids ),flush =True )
    print ("PASS: frozen split hashes and membership",flush =True )
    return dev_ids ,core_ids ,split_metadata 


def optional_float (value :str )->float :
    if value is None or value =="":
        return 0.0 
    result =float (value )
    if not math .isfinite (result ):
        raise ValueError (f"Non-finite numeric value: {value }")
    return result 


def calibrated_confidence_support (score :float ,threshold :float )->float :
    score =min (1.0 ,max (0.0 ,score ))
    if score <=threshold :
        if threshold <=0.0 :
            return 0.5 
        return 0.5 *score /threshold 

    if threshold >=1.0 :
        return 0.5 
    return 0.5 +0.5 *(score -threshold )/(1.0 -threshold )


def detector_quality (row :dict [str ,str ],threshold :float )->float :
    detected =int (row ["detected"])

    if detected :
        score =optional_float (row ["matched_score"])
        iou =optional_float (row ["matched_iou"])
        multiplier =1.0 
    else :
        score =optional_float (row ["best_same_all_score"])
        iou =optional_float (row ["best_same_all_iou"])
        multiplier =NONDETECTION_QUALITY_MULTIPLIER 

    confidence_support =calibrated_confidence_support (score ,threshold )
    localization_support =min (1.0 ,max (0.0 ,iou ))
    quality =multiplier *math .sqrt (
    confidence_support *localization_support 
    )
    return min (1.0 ,max (0.0 ,quality ))


def percentile_rank (values :np .ndarray )->np .ndarray :
    count =len (values )
    if count ==0 :
        return np .empty (0 ,dtype =np .float32 )
    if count ==1 :
        return np .asarray ([0.5 ],dtype =np .float32 )

    order =np .argsort (values ,kind ="mergesort")
    sorted_values =values [order ]
    _ ,starts ,counts =np .unique (
    sorted_values ,
    return_index =True ,
    return_counts =True ,
    )
    average_positions =(
    starts .astype (np .float64 )
    +(counts .astype (np .float64 )-1.0 )/2.0 
    )/float (count -1 )
    sorted_ranks =np .repeat (average_positions ,counts ).astype (np .float32 )

    ranks =np .empty (count ,dtype =np .float32 )
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


def distribution_summary (values :np .ndarray )->dict :
    if len (values )==0 :
        return {"count":0 }

    quantiles =np .quantile (
    values ,
    [0.0 ,0.25 ,0.5 ,0.75 ,0.9 ,0.95 ,0.99 ,1.0 ],
    )
    labels =("min","p25","p50","p75","p90","p95","p99","max")

    result ={
    "count":int (len (values )),
    "mean":float (np .mean (values )),
    "std":float (np .std (values )),
    }
    result .update (
    {label :float (value )for label ,value in zip (labels ,quantiles )}
    )
    return result 


RISK_FIELDS =[
"annotation_id",
"image_id",
"split",
"category_id",
"category",
"scale",
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
]

IMAGE_FIELDS =[
"image_id",
"split",
"num_objects",
"num_small_objects",
"risk_mean",
"risk_max",
"risk_p90",
"small_risk_mean",
"teacher_disagreement_mean",
"high_or_critical_fraction",
]


def main ()->None :
    args =parse_args ()
    matches_dir =Path (args .matches_dir ).expanduser ().resolve ()
    split_dir =Path (args .split_dir ).expanduser ().resolve ()
    output_dir =Path (args .output_dir ).expanduser ().resolve ()

    if not matches_dir .is_dir ():
        raise FileNotFoundError (matches_dir )
    if not split_dir .is_dir ():
        raise FileNotFoundError (split_dir )
    if output_dir .exists ():
        raise FileExistsError (
        f"Refusing to overwrite output directory: {output_dir }"
        )

    start_time =time .time ()

    match_metadata ,match_manifest_sha256 =validate_input_manifest (
    matches_dir 
    )
    dev_ids ,core_ids ,split_metadata =load_frozen_split (split_dir )

    reliability_sum =sum (CALIBRATION_MACRO_F1 .values ())
    reliability_weights ={
    model :CALIBRATION_MACRO_F1 [model ]/reliability_sum 
    for model in MODELS 
    }

    annotation_ids =np .empty (EXPECTED_OBJECTS ,dtype =np .int64 )
    image_ids =np .empty (EXPECTED_OBJECTS ,dtype =np .int64 )
    category_ids =np .empty (EXPECTED_OBJECTS ,dtype =np .uint8 )
    scale_codes =np .empty (EXPECTED_OBJECTS ,dtype =np .uint8 )
    split_codes =np .empty (EXPECTED_OBJECTS ,dtype =np .uint8 )
    yolo_status_codes =np .empty (EXPECTED_OBJECTS ,dtype =np .uint8 )
    dfine_status_codes =np .empty (EXPECTED_OBJECTS ,dtype =np .uint8 )
    yolo_detected =np .empty (EXPECTED_OBJECTS ,dtype =np .uint8 )
    dfine_detected =np .empty (EXPECTED_OBJECTS ,dtype =np .uint8 )
    yolo_quality =np .empty (EXPECTED_OBJECTS ,dtype =np .float32 )
    dfine_quality =np .empty (EXPECTED_OBJECTS ,dtype =np .float32 )
    yolo_risk =np .empty (EXPECTED_OBJECTS ,dtype =np .float32 )
    dfine_risk =np .empty (EXPECTED_OBJECTS ,dtype =np .float32 )
    consensus_risk =np .empty (EXPECTED_OBJECTS ,dtype =np .float32 )
    disagreement =np .empty (EXPECTED_OBJECTS ,dtype =np .float32 )
    joint_raw_risk =np .empty (EXPECTED_OBJECTS ,dtype =np .float32 )

    yolo_path =matches_dir /"YOLO11m"/"object_matches.csv"
    dfine_path =matches_dir /"D-FINE-M"/"object_matches.csv"

    status_pairs =Counter ()
    detection_consensus =Counter ()
    observed_image_ids =set ()

    print ("=== FUSING OBJECT MATCHES ===",flush =True )

    with (
    yolo_path .open ("r",encoding ="utf-8",newline ="")as yolo_file ,
    dfine_path .open ("r",encoding ="utf-8",newline ="")as dfine_file ,
    ):
        yolo_reader =csv .DictReader (yolo_file )
        dfine_reader =csv .DictReader (dfine_file )

        object_count =0 
        for index ,pair in enumerate (
        zip_longest (yolo_reader ,dfine_reader ),
        ):
            yolo_row ,dfine_row =pair 
            if yolo_row is None or dfine_row is None :
                raise RuntimeError ("Teacher object CSV lengths differ.")
            if index >=EXPECTED_OBJECTS :
                raise RuntimeError ("More object rows than expected.")

            common_fields =(
            "annotation_id",
            "image_id",
            "category_id",
            "category",
            "scale",
            "gt_area",
            )
            for field in common_fields :
                if yolo_row [field ]!=dfine_row [field ]:
                    raise RuntimeError (
                    f"Teacher row mismatch at index={index }, field={field }"
                    )

            annotation_id =int (yolo_row ["annotation_id"])
            image_id =int (yolo_row ["image_id"])
            category_id =int (yolo_row ["category_id"])
            scale =yolo_row ["scale"]
            y_status =yolo_row ["status"]
            d_status =dfine_row ["status"]

            if category_id not in CATEGORY_NAMES :
                raise ValueError (f"Unknown category_id={category_id }")
            if scale not in SCALE_TO_CODE :
                raise ValueError (f"Unknown scale={scale }")
            if y_status not in STATUS_TO_CODE :
                raise ValueError (f"Unknown YOLO status={y_status }")
            if d_status not in STATUS_TO_CODE :
                raise ValueError (f"Unknown D-FINE status={d_status }")

            if image_id in dev_ids :
                split_name ="train_dev"
            elif image_id in core_ids :
                split_name ="train_core"
            else :
                raise ValueError (f"Image absent from frozen split: {image_id }")

            y_detected =int (yolo_row ["detected"])
            d_detected =int (dfine_row ["detected"])
            y_quality =detector_quality (
            yolo_row ,
            EXPECTED_THRESHOLDS ["YOLO11m"],
            )
            d_quality =detector_quality (
            dfine_row ,
            EXPECTED_THRESHOLDS ["D-FINE-M"],
            )
            y_risk =1.0 -y_quality 
            d_risk =1.0 -d_quality 

            consensus =(
            reliability_weights ["YOLO11m"]*y_risk 
            +reliability_weights ["D-FINE-M"]*d_risk 
            )
            difference =abs (y_risk -d_risk )
            joint =1.0 -(1.0 -consensus )*(1.0 -difference )

            annotation_ids [index ]=annotation_id 
            image_ids [index ]=image_id 
            category_ids [index ]=category_id 
            scale_codes [index ]=SCALE_TO_CODE [scale ]
            split_codes [index ]=SPLIT_TO_CODE [split_name ]
            yolo_status_codes [index ]=STATUS_TO_CODE [y_status ]
            dfine_status_codes [index ]=STATUS_TO_CODE [d_status ]
            yolo_detected [index ]=y_detected 
            dfine_detected [index ]=d_detected 
            yolo_quality [index ]=y_quality 
            dfine_quality [index ]=d_quality 
            yolo_risk [index ]=y_risk 
            dfine_risk [index ]=d_risk 
            consensus_risk [index ]=consensus 
            disagreement [index ]=difference 
            joint_raw_risk [index ]=joint 

            status_pairs [(y_status ,d_status )]+=1 
            observed_image_ids .add (image_id )

            if y_detected and d_detected :
                detection_consensus ["both_detected"]+=1 
            elif y_detected :
                detection_consensus ["yolo_only"]+=1 
            elif d_detected :
                detection_consensus ["dfine_only"]+=1 
            else :
                detection_consensus ["neither_detected"]+=1 

            object_count +=1 

            if object_count %200_000 ==0 :
                print (
                f"fused={object_count }/{EXPECTED_OBJECTS }",
                flush =True ,
                )

    if object_count !=EXPECTED_OBJECTS :
        raise RuntimeError (
        f"Unexpected fused object count: {object_count }"
        )
    if len (np .unique (annotation_ids ))!=EXPECTED_OBJECTS :
        raise RuntimeError ("Duplicate annotation IDs in fused table.")
    frozen_image_ids =dev_ids |core_ids 
    if not observed_image_ids <=frozen_image_ids :
        raise RuntimeError ("Fused objects contain images outside frozen split.")
    missing_object_image_ids =sorted (frozen_image_ids -observed_image_ids )
    print (
    "images_with_gt_objects:",
    len (observed_image_ids ),
    flush =True ,
    )
    print (
    "images_without_gt_objects:",
    len (missing_object_image_ids ),
    flush =True ,
    )

    print ("=== COMPUTING SHRUNK RISK RANKS ===",flush =True )
    global_percentile =percentile_rank (joint_raw_risk )
    class_percentile =np .empty (EXPECTED_OBJECTS ,dtype =np .float32 )
    class_reliability =np .empty (EXPECTED_OBJECTS ,dtype =np .float32 )

    class_counts =Counter ()
    for category_id in sorted (CATEGORY_NAMES ):
        positions =np .flatnonzero (category_ids ==category_id )
        count =len (positions )
        class_counts [category_id ]=count 
        if count ==0 :
            raise RuntimeError (f"No objects for category_id={category_id }")

        class_percentile [positions ]=percentile_rank (
        joint_raw_risk [positions ]
        )
        reliability =count /(count +CLASS_PRIOR_STRENGTH )
        class_reliability [positions ]=reliability 

    risk_score =(
    class_reliability *class_percentile 
    +(1.0 -class_reliability )*global_percentile 
    ).astype (np .float32 )

    output_dir .parent .mkdir (parents =True ,exist_ok =True )
    staging_dir =output_dir .with_name (
    output_dir .name +f".incomplete-{int (time .time ())}-{os .getpid ()}"
    )
    if staging_dir .exists ():
        raise FileExistsError (staging_dir )
    staging_dir .mkdir ()

    risk_table_path =staging_dir /"object_risk_table.csv"
    image_summary_path =staging_dir /"image_risk_summary.csv"

    print ("=== WRITING RISK TABLES ===",flush =True )

    tier_counts =Counter ()

    with (
    yolo_path .open ("r",encoding ="utf-8",newline ="")as yolo_file ,
    dfine_path .open ("r",encoding ="utf-8",newline ="")as dfine_file ,
    risk_table_path .open ("w",encoding ="utf-8",newline ="")as risk_file ,
    image_summary_path .open (
    "w",encoding ="utf-8",newline =""
    )as image_file ,
    ):
        yolo_reader =csv .DictReader (yolo_file )
        dfine_reader =csv .DictReader (dfine_file )
        risk_writer =csv .DictWriter (risk_file ,fieldnames =RISK_FIELDS )
        image_writer =csv .DictWriter (image_file ,fieldnames =IMAGE_FIELDS )
        risk_writer .writeheader ()
        image_writer .writeheader ()

        current_image_id =None 
        current_split =None 
        current_risks =[]
        current_disagreements =[]
        current_small_risks =[]
        written_images =0 

        def flush_image_summary ():
            nonlocal written_images 
            if current_image_id is None :
                return 

            risks =np .asarray (current_risks ,dtype =np .float32 )
            disagreements =np .asarray (
            current_disagreements ,
            dtype =np .float32 ,
            )
            small =np .asarray (current_small_risks ,dtype =np .float32 )

            image_writer .writerow (
            {
            "image_id":current_image_id ,
            "split":current_split ,
            "num_objects":len (risks ),
            "num_small_objects":len (small ),
            "risk_mean":format (float (np .mean (risks )),".10g"),
            "risk_max":format (float (np .max (risks )),".10g"),
            "risk_p90":format (
            float (np .quantile (risks ,0.90 )),".10g"
            ),
            "small_risk_mean":(
            format (float (np .mean (small )),".10g")
            if len (small )
            else ""
            ),
            "teacher_disagreement_mean":format (
            float (np .mean (disagreements )),".10g"
            ),
            "high_or_critical_fraction":format (
            float (np .mean (risks >=0.75 )),".10g"
            ),
            }
            )
            written_images +=1 

        for index ,pair in enumerate (
        zip_longest (yolo_reader ,dfine_reader )
        ):
            yolo_row ,dfine_row =pair 
            if yolo_row is None or dfine_row is None :
                raise RuntimeError ("Teacher CSV lengths changed during write.")

            annotation_id =int (yolo_row ["annotation_id"])
            image_id =int (yolo_row ["image_id"])
            if annotation_id !=int (annotation_ids [index ]):
                raise RuntimeError ("Annotation order changed during write.")
            if image_id !=int (image_ids [index ]):
                raise RuntimeError ("Image order changed during write.")

            split_name =CODE_TO_SPLIT [int (split_codes [index ])]
            category_id =int (category_ids [index ])
            scale =CODE_TO_SCALE [int (scale_codes [index ])]
            score =float (risk_score [index ])
            tier =risk_tier (score )
            tier_counts [tier ]+=1 

            risk_writer .writerow (
            {
            "annotation_id":annotation_id ,
            "image_id":image_id ,
            "split":split_name ,
            "category_id":category_id ,
            "category":yolo_row ["category"],
            "scale":scale ,
            "gt_area":yolo_row ["gt_area"],
            "timeofday":yolo_row ["timeofday"],
            "weather":yolo_row ["weather"],
            "scene":yolo_row ["scene"],
            "yolo_status":yolo_row ["status"],
            "yolo_detected":int (yolo_detected [index ]),
            "yolo_quality":format (
            float (yolo_quality [index ]),".10g"
            ),
            "yolo_risk":format (float (yolo_risk [index ]),".10g"),
            "dfine_status":dfine_row ["status"],
            "dfine_detected":int (dfine_detected [index ]),
            "dfine_quality":format (
            float (dfine_quality [index ]),".10g"
            ),
            "dfine_risk":format (
            float (dfine_risk [index ]),".10g"
            ),
            "consensus_risk":format (
            float (consensus_risk [index ]),".10g"
            ),
            "teacher_disagreement":format (
            float (disagreement [index ]),".10g"
            ),
            "joint_raw_risk":format (
            float (joint_raw_risk [index ]),".10g"
            ),
            "global_risk_percentile":format (
            float (global_percentile [index ]),".10g"
            ),
            "class_risk_percentile":format (
            float (class_percentile [index ]),".10g"
            ),
            "class_reliability":format (
            float (class_reliability [index ]),".10g"
            ),
            "risk_score":format (score ,".10g"),
            "risk_tier":tier ,
            }
            )

            if current_image_id is None :
                current_image_id =image_id 
                current_split =split_name 
            elif image_id !=current_image_id :
                flush_image_summary ()
                current_image_id =image_id 
                current_split =split_name 
                current_risks =[]
                current_disagreements =[]
                current_small_risks =[]

            current_risks .append (score )
            current_disagreements .append (float (disagreement [index ]))
            if scale =="small":
                current_small_risks .append (score )

            if (index +1 )%200_000 ==0 :
                print (
                f"written={index +1 }/{EXPECTED_OBJECTS }",
                flush =True ,
                )

        flush_image_summary ()

        for image_id in missing_object_image_ids :
            split_name =(
            "train_dev"if image_id in dev_ids else "train_core"
            )
            image_writer .writerow (
            {
            "image_id":image_id ,
            "split":split_name ,
            "num_objects":0 ,
            "num_small_objects":0 ,
            "risk_mean":"",
            "risk_max":"",
            "risk_p90":"",
            "small_risk_mean":"",
            "teacher_disagreement_mean":"",
            "high_or_critical_fraction":"",
            }
            )
            written_images +=1 

        if written_images !=EXPECTED_IMAGES :
            raise RuntimeError (
            f"Unexpected image summary count: {written_images }"
            )

    split_summary ={
    CODE_TO_SPLIT [code ]:distribution_summary (
    risk_score [split_codes ==code ]
    )
    for code in sorted (CODE_TO_SPLIT )
    }
    category_summary ={
    str (category_id ):{
    "category":CATEGORY_NAMES [category_id ],
    "class_reliability":float (
    class_counts [category_id ]
    /(class_counts [category_id ]+CLASS_PRIOR_STRENGTH )
    ),
    "risk":distribution_summary (
    risk_score [category_ids ==category_id ]
    ),
    "joint_raw_risk":distribution_summary (
    joint_raw_risk [category_ids ==category_id ]
    ),
    }
    for category_id in sorted (CATEGORY_NAMES )
    }
    scale_summary ={
    scale :distribution_summary (risk_score [scale_codes ==code ])
    for scale ,code in SCALE_TO_CODE .items ()
    }

    status_pair_payload ={
    f"{yolo_status } | {dfine_status }":count 
    for (yolo_status ,dfine_status ),count in sorted (
    status_pairs .items ()
    )
    }

    risk_summary ={
    "method":METHOD_NAME ,
    "num_objects":EXPECTED_OBJECTS ,
    "num_images":EXPECTED_IMAGES ,
    "overall_risk":distribution_summary (risk_score ),
    "overall_joint_raw_risk":distribution_summary (joint_raw_risk ),
    "overall_teacher_disagreement":distribution_summary (disagreement ),
    "risk_tier_counts":dict (sorted (tier_counts .items ())),
    "detection_consensus":dict (sorted (detection_consensus .items ())),
    "teacher_status_pairs":status_pair_payload ,
    "by_split":split_summary ,
    "by_category":category_summary ,
    "by_scale":scale_summary ,
    }
    write_json (staging_dir /"risk_summary.json",risk_summary )

    metadata ={
    "created_utc":datetime .now (timezone .utc ).isoformat (),
    "method":METHOD_NAME ,
    "dataset":"BDD100K",
    "split":"official_training",
    "num_images":EXPECTED_IMAGES ,
    "num_objects":EXPECTED_OBJECTS ,
    "matches_dir":str (matches_dir ),
    "matches_metadata_sha256":sha256_file (
    matches_dir /"metadata.json"
    ),
    "matches_manifest_sha256":match_manifest_sha256 ,
    "calibration_sha256":EXPECTED_CALIBRATION_SHA256 ,
    "calibration_thresholds":EXPECTED_THRESHOLDS ,
    "teacher_calibration_macro_f1":CALIBRATION_MACRO_F1 ,
    "teacher_reliability_weights":reliability_weights ,
    "split_dir":str (split_dir ),
    "split_hashes":EXPECTED_SPLIT_HASHES ,
    "split_metadata":split_metadata ,
    "risk_definition":{
    "confidence_support":(
    "Piecewise-linear model-specific score transform with "
    "the frozen operating threshold mapped to 0.5."
    ),
    "teacher_quality":(
    "sqrt(confidence_support * same-class IoU); multiply by "
    "0.5 when the object is not greedily detected."
    ),
    "teacher_risk":"1 - teacher_quality",
    "consensus_risk":(
    "Macro-F1 reliability-weighted mean of teacher risks."
    ),
    "teacher_disagreement":(
    "Absolute difference between the two teacher risks."
    ),
    "joint_raw_risk":(
    "1 - (1 - consensus_risk) * (1 - teacher_disagreement)"
    ),
    "risk_score":(
    "Empirical-Bayes blend of within-class and global "
    "joint-risk percentile ranks."
    ),
    "class_reliability":(
    "class_count / (class_count + 500)"
    ),
    "risk_tiers":{
    "standard":"risk_score < 0.50",
    "medium":"0.50 <= risk_score < 0.75",
    "high":"0.75 <= risk_score < 0.90",
    "critical":"risk_score >= 0.90",
    },
    },
    "non_detection_quality_multiplier":(
    NONDETECTION_QUALITY_MULTIPLIER 
    ),
    "class_prior_strength":CLASS_PRIOR_STRENGTH ,
    "input_match_metadata":match_metadata ,
    "script":str (Path (__file__ ).resolve ()),
    "script_sha256":sha256_file (Path (__file__ ).resolve ()),
    "python":sys .version ,
    "platform":platform .platform (),
    "numpy":np .__version__ ,
    "elapsed_minutes":(time .time ()-start_time )/60.0 ,
    }
    write_json (staging_dir /"metadata.json",metadata )

    artifacts =[]
    for path in sorted (staging_dir .rglob ("*")):
        if path .is_file ():
            artifacts .append (
            {
            "path":str (path .relative_to (staging_dir )),
            "bytes":path .stat ().st_size ,
            "sha256":sha256_file (path ),
            }
            )

    write_json (
    staging_dir /"artifact_manifest.json",
    {
    "created_utc":datetime .now (timezone .utc ).isoformat (),
    "artifacts":artifacts ,
    },
    )

    os .replace (staging_dir ,output_dir )

    print ("\n"+"="*80 )
    print ("TRAIN RISK TABLE COMPLETE")
    print ("objects:",EXPECTED_OBJECTS )
    print ("images:",EXPECTED_IMAGES )
    print ("risk_tiers:",dict (sorted (tier_counts .items ())))
    print ("detection_consensus:",dict (sorted (detection_consensus .items ())))
    print ("output:",output_dir )
    print ("metadata:",output_dir /"metadata.json")
    print ("manifest:",output_dir /"artifact_manifest.json")
    print ("elapsed_minutes:",round ((time .time ()-start_time )/60.0 ,3 ))
    print ("="*80 )


if __name__ =="__main__":
    main ()
