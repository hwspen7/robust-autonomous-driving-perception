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
from collections import Counter 
from datetime import datetime ,timezone 
from pathlib import Path 

import numpy as np 
from pycocotools .coco import COCO 
from pycocotools .cocoeval import COCOeval 


PROJECT =Path (
"/root/rivermind-data/autodrive/code/"
"robust-autonomous-driving-perception"
)
RESULTS =Path (
"/root/rivermind-data/autodrive/results/evaluation"
)
DATASET =Path (
"/root/rivermind-data/autodrive/datasets/datasets/"
"bdd100k_final"
)
GT =DATASET /"coco/annotations/instances_val.json"
EXPORTER =PROJECT /"scripts/export_yolo11m_predictions.py"

FINAL_OUTPUT =(
RESULTS 
/"final_analysis_v1"
/"canonical_clean_v1"
)
WORK_OUTPUT =FINAL_OUTPUT .parent /(
FINAL_OUTPUT .name 
+".incomplete-693c39375cc1"
)

PROTOCOL_FILES ={
RESULTS 
/"final_analysis_v1/frozen_protocol/"
"final_testing_protocol_v1.json":
"693c39375cc1eefc4f3ff9f957a8355fb947552a19dfef228cae6cfc21e3dcfe",

RESULTS 
/"final_analysis_v1/frozen_protocol/"
"input_inventory.json":
"47354efac7fa634a8e6c713de8adb188ee40be265c25202f35bbb657ba23d16d",

RESULTS 
/"final_analysis_v1/frozen_protocol/"
"artifact_manifest.json":
"bc6bac887694d7e4e09210e6996df22d1cf7006b8c44957011bd4fcf9112adce",
}

GT_SHA256 =(
"7be8a02e147743123a4a62b2c92ed2e4"
"a5712d261c8868223e6325bad6e346d6"
)
EXPORTER_SHA256 =(
"974efcdb71dbbe4fff980cae27ebb7cf8"
"26a630bb9d5c2afd654ed3b36f23166"
)

REUSABLE ={
"baseline_yolo":{
"display_name":"YOLO11m",
"prediction":(
RESULTS 
/"unified_detection/canonical_predictions/"
"YOLO11m.json"
),
"prediction_sha256":(
"14ff8491f7849077cb7ead3ce7d9e618d"
"2f92ee04bef0a8c68efdb1b836a54b0"
),
},
"baseline_dfine":{
"display_name":"D-FINE-M",
"prediction":(
RESULTS 
/"unified_detection/canonical_predictions/"
"D-FINE-M.json"
),
"prediction_sha256":(
"10b4159d0f52e412714ac9ae076ac574e"
"bbb74321ddcebab03e2bdbad55e3ebd"
),
},
"standard_e20":{
"display_name":"Standard-960-E20",
"prediction":(
RESULTS 
/"architecture_gate/"
"borderline_adjudication_v5_1/"
"predictions/standard_epoch20/"
"predictions.json"
),
"prediction_sha256":(
"422bf84b636d422c46a186c2ef8a6f7e"
"585b94cd82c8ef97bc799f345067352b"
),
},
"p2_e20":{
"display_name":"P2-960-E20",
"prediction":(
RESULTS 
/"architecture_gate/"
"borderline_adjudication_v5_1/"
"predictions/p2_epoch20/"
"predictions.json"
),
"prediction_sha256":(
"17167dad4b6158e68d78262ca43f60b37"
"71372db5ae228a156ac8022a30e2b81"
),
},
}

NEW_MODELS ={
"common40":{
"display_name":"P2-960-Common40",
"checkpoint":(
RESULTS 
/"formal_training/training_v1/"
"common_uniform_e1_40/weights/"
"common40_ema.pt"
),
"checkpoint_sha256":(
"c05ed70a608d026fef6178f40a4dee62"
"e96323731700b42e426c311dd78b27f3"
),
},
"gate_uniform":{
"display_name":"Uniform-Gate-E10",
"checkpoint":(
RESULTS 
/"method_gate/training_v1/"
"uniform/weights/last.pt"
),
"checkpoint_sha256":(
"45f397e81cf6aa3e8fbb08a8fef79b8"
"e95975f85f5b360fa9f4bf2a4f6757210"
),
},
"gate_scalar":{
"display_name":"Scalar-Risk-Gate-E10",
"checkpoint":(
RESULTS 
/"method_gate/training_v1/"
"scalar_risk/weights/last.pt"
),
"checkpoint_sha256":(
"3093b249d8052924d8324c826731a77b"
"82ab98327a41ff68ab3a2ad615f74b9d"
),
},
"gate_typed":{
"display_name":"Typed-CAFR-Gate-E10",
"checkpoint":(
RESULTS 
/"method_gate/training_v1/"
"typed_cafr/weights/last.pt"
),
"checkpoint_sha256":(
"217b6d453d399c345a7e1be559578304"
"4f2555a65ae95f047d6a7278d105c4ee"
),
},
"gate_typed_v2":{
"display_name":"Typed-CAFR-V2-Gate-E10",
"checkpoint":(
RESULTS 
/"method_gate/cafr_v2_training_v1/"
"typed_cafr/weights/last.pt"
),
"checkpoint_sha256":(
"eaa3dd7eee88f7d0c74cfbe45f6237c8"
"2ecb4eedacb9615a260ce351bc2da7dd"
),
},
"uniform_e56":{
"display_name":"Uniform-E56",
"checkpoint":(
RESULTS 
/"formal_training/training_v2/"
"uniform_e41_100/weights/epoch15.pt"
),
"checkpoint_sha256":(
"84aa86771d185c2c72fa277c609c1222"
"d7a417a8197523c4a5ced2a6a91ebb1b"
),
},
"risk_e56":{
"display_name":"Risk-E56",
"checkpoint":(
RESULTS 
/"formal_training/training_v2/"
"rebu_risk_e41_100/weights/epoch15.pt"
),
"checkpoint_sha256":(
"879b179c219914a83775cacb7074f1d7b"
"d9836971fae24592a6a16d6e87fe102"
),
},
"uniform_e100":{
"display_name":"Uniform-E100",
"checkpoint":(
RESULTS 
/"formal_training/training_v2/"
"uniform_e41_100/weights/"
"uniform100_ema.pt"
),
"checkpoint_sha256":(
"2af91824191b0d48d40e35856abafd388"
"776815aaf0a629ace4e734a9be01cf7"
),
},
"risk_e100":{
"display_name":"Risk-E100",
"checkpoint":(
RESULTS 
/"formal_training/training_v2/"
"rebu_risk_e41_100/weights/"
"rebu_yolo100_ema.pt"
),
"checkpoint_sha256":(
"d85a864b7f84d0f408a68697cf9f4cbb"
"fb73a9b821bb8ca392ba49c2956c447a"
),
},
}

METRIC_NAMES =[
"AP",
"AP50",
"AP75",
"AP_small",
"AP_medium",
"AP_large",
"AR1",
"AR10",
"AR100",
"AR_small",
"AR_medium",
"AR_large",
]


def sha256 (path :Path )->str :
    digest =hashlib .sha256 ()
    with path .open ("rb")as handle :
        while True :
            block =handle .read (8 *1024 *1024 )
            if not block :
                break 
            digest .update (block )
    return digest .hexdigest ()


def write_json_atomic (path :Path ,data )->None :
    path .parent .mkdir (parents =True ,exist_ok =True )
    temp =path .with_suffix (path .suffix +".tmp")
    temp .write_text (
    json .dumps (
    data ,
    indent =2 ,
    ensure_ascii =False ,
    sort_keys =True ,
    )+"\n",
    encoding ="utf-8",
    )
    os .replace (temp ,path )


def write_csv_atomic (
path :Path ,
rows :list [dict ],
fieldnames :list [str ],
)->None :
    path .parent .mkdir (parents =True ,exist_ok =True )
    temp =path .with_suffix (path .suffix +".tmp")

    with temp .open (
    "w",
    newline ="",
    encoding ="utf-8",
    )as handle :
        writer =csv .DictWriter (
        handle ,
        fieldnames =fieldnames ,
        )
        writer .writeheader ()
        writer .writerows (rows )

    os .replace (temp ,path )


def verify_file (
path :Path ,
expected_sha256 :str ,
)->None :
    if not path .is_file ():
        raise FileNotFoundError (path )

    actual =sha256 (path )

    if actual !=expected_sha256 :
        raise AssertionError (
        f"SHA256 mismatch\n"
        f"path: {path }\n"
        f"expected: {expected_sha256 }\n"
        f"actual:   {actual }"
        )


def preflight ()->None :
    print ("=== FINAL CANONICAL-CLEAN EXECUTOR PREFLIGHT ===")

    if FINAL_OUTPUT .exists ():
        raise FileExistsError (
        f"Refusing to overwrite: {FINAL_OUTPUT }"
        )

    for path ,expected in PROTOCOL_FILES .items ():
        verify_file (path ,expected )
        print ("PASS:",path )

    verify_file (GT ,GT_SHA256 )
    verify_file (EXPORTER ,EXPORTER_SHA256 )

    for key ,item in REUSABLE .items ():
        verify_file (
        item ["prediction"],
        item ["prediction_sha256"],
        )
        print ("PASS reusable:",key )

    for key ,item in NEW_MODELS .items ():
        verify_file (
        item ["checkpoint"],
        item ["checkpoint_sha256"],
        )
        print ("PASS checkpoint:",key )

    usage =shutil .disk_usage (
    "/root/rivermind-data"
    )
    free_gib =usage .free /1024 **3 

    print ("free_GiB:",round (free_gib ,3 ))
    print ("work_output:",WORK_OUTPUT )
    print ("work_output_exists:",WORK_OUTPUT .exists ())

    if free_gib <4.0 :
        raise RuntimeError (
        f"At least 4 GiB free required; found "
        f"{free_gib :.3f} GiB"
        )

    print ("numpy:",np .__version__ )
    print ("python:",sys .version .replace ("\n"," "))
    print ("PASS: canonical-clean execution is ready")


def validate_predictions (
path :Path ,
image_dimensions :dict [int ,tuple [float ,float ]],
valid_category_ids :set [int ],
)->dict :
    with path .open ("r",encoding ="utf-8")as handle :
        predictions =json .load (handle )

    if not isinstance (predictions ,list ):
        raise TypeError (
        f"Prediction file must contain a list: {path }"
        )

    image_ids =set ()
    category_counts =Counter ()

    negative_size =0 
    zero_area =0 
    out_of_bounds =0 
    minimum_score =math .inf 
    maximum_score =-math .inf 

    for index ,item in enumerate (predictions ):
        image_id =int (item ["image_id"])
        category_id =int (item ["category_id"])
        bbox =item ["bbox"]
        score =float (item ["score"])

        if image_id not in image_dimensions :
            raise AssertionError (
            f"Unknown image_id at row {index }: "
            f"{image_id }"
            )

        if category_id not in valid_category_ids :
            raise AssertionError (
            f"Invalid category_id at row {index }: "
            f"{category_id }"
            )

        if len (bbox )!=4 :
            raise AssertionError (
            f"Invalid bbox at row {index }: {bbox }"
            )

        if not math .isfinite (score ):
            raise AssertionError (
            f"Non-finite score at row {index }"
            )

        x ,y ,width ,height =map (float ,bbox )
        image_width ,image_height =(
        image_dimensions [image_id ]
        )

        if width <0 or height <0 :
            negative_size +=1 

        if width ==0 or height ==0 :
            zero_area +=1 

        if (
        x <0 
        or y <0 
        or x +width >image_width 
        or y +height >image_height 
        ):
            out_of_bounds +=1 

        image_ids .add (image_id )
        category_counts [category_id ]+=1 
        minimum_score =min (minimum_score ,score )
        maximum_score =max (maximum_score ,score )

    result ={
    "predictions":len (predictions ),
    "unique_prediction_image_ids":len (image_ids ),
    "minimum_score":(
    None 
    if not predictions 
    else minimum_score 
    ),
    "maximum_score":(
    None 
    if not predictions 
    else maximum_score 
    ),
    "negative_size_bbox":negative_size ,
    "zero_area_bbox":zero_area ,
    "out_of_bounds_bbox":out_of_bounds ,
    "per_category_predictions":{
    str (key ):value 
    for key ,value in sorted (
    category_counts .items ()
    )
    },
    "prediction_sha256":sha256 (path ),
    "prediction_bytes":path .stat ().st_size ,
    }

    del predictions 
    gc .collect ()

    return result 


def mean_valid (values :np .ndarray )->float :
    valid =values [values >-1 ]

    if valid .size ==0 :
        return float ("nan")

    return float (np .mean (valid ))


def evaluate_predictions (
coco_gt :COCO ,
prediction_path :Path ,
model_key :str ,
display_name :str ,
prediction_audit :dict ,
)->tuple [dict ,list [dict ]]:
    print ("\n"+"="*100 )
    print ("COCO EVALUATION:",model_key )
    print ("="*100 )

    started =time .time ()

    coco_dt =coco_gt .loadRes (
    str (prediction_path )
    )

    evaluator =COCOeval (
    coco_gt ,
    coco_dt ,
    "bbox",
    )
    evaluator .params .imgIds =sorted (
    coco_gt .getImgIds ()
    )
    evaluator .params .catIds =sorted (
    coco_gt .getCatIds ()
    )
    evaluator .params .maxDets =[1 ,10 ,100 ]

    evaluator .evaluate ()
    evaluator .accumulate ()
    evaluator .summarize ()

    stats =[
    float (value )
    for value in evaluator .stats 
    ]

    overall ={
    "model_key":model_key ,
    "display_name":display_name ,
    "AP":stats [0 ],
    "AP50":stats [1 ],
    "AP75":stats [2 ],
    "AP_small":stats [3 ],
    "AP_medium":stats [4 ],
    "AP_large":stats [5 ],
    "AR1":stats [6 ],
    "AR10":stats [7 ],
    "AR100":stats [8 ],
    "AR_small":stats [9 ],
    "AR_medium":stats [10 ],
    "AR_large":stats [11 ],
    "predictions":prediction_audit ["predictions"],
    "prediction_sha256":(
    prediction_audit ["prediction_sha256"]
    ),
    "elapsed_minutes":(
    time .time ()-started 
    )/60.0 ,
    }

    precision =evaluator .eval ["precision"]
    recall =evaluator .eval ["recall"]

    iou50_index =int (
    np .where (
    np .isclose (
    evaluator .params .iouThrs ,
    0.50 ,
    )
    )[0 ][0 ]
    )
    iou75_index =int (
    np .where (
    np .isclose (
    evaluator .params .iouThrs ,
    0.75 ,
    )
    )[0 ][0 ]
    )

    categories ={
    category ["id"]:category ["name"]
    for category in coco_gt .dataset ["categories"]
    }

    per_class =[]

    for category_index ,category_id in enumerate (
    evaluator .params .catIds 
    ):
        gt_annotation_ids =coco_gt .getAnnIds (
        catIds =[category_id ]
        )
        gt_annotations =coco_gt .loadAnns (
        gt_annotation_ids 
        )

        gt_image_ids ={
        int (item ["image_id"])
        for item in gt_annotations 
        }

        per_class .append (
        {
        "model_key":model_key ,
        "display_name":display_name ,
        "category_id":int (category_id ),
        "class_name":categories [category_id ],
        "AP":mean_valid (
        precision [
        :,
        :,
        category_index ,
        0 ,
        2 ,
        ]
        ),
        "AP50":mean_valid (
        precision [
        iou50_index ,
        :,
        category_index ,
        0 ,
        2 ,
        ]
        ),
        "AP75":mean_valid (
        precision [
        iou75_index ,
        :,
        category_index ,
        0 ,
        2 ,
        ]
        ),
        "AR100":mean_valid (
        recall [
        :,
        category_index ,
        0 ,
        2 ,
        ]
        ),
        "num_gt":len (gt_annotations ),
        "num_images_with_gt":len (gt_image_ids ),
        "num_predictions":int (
        prediction_audit [
        "per_category_predictions"
        ].get (
        str (category_id ),
        0 ,
        )
        ),
        }
        )

    del evaluator 
    del coco_dt 
    gc .collect ()

    return overall ,per_class 


def export_model (
key :str ,
item :dict ,
output_dir :Path ,
)->None :
    prediction_path =(
    output_dir /"predictions.json"
    )
    metadata_path =(
    output_dir /"metadata.json"
    )

    if prediction_path .is_file ()and metadata_path .is_file ():
        print (
        f"REUSE COMPLETED NEW EXPORT: {key }"
        )
        return 

    if output_dir .exists ():
        raise RuntimeError (
        "Incomplete export directory requires "
        f"manual audit before retry: {output_dir }"
        )

    command =[
    sys .executable ,
    "-u",
    str (EXPORTER ),
    "--model",
    str (item ["checkpoint"]),
    "--dataset-root",
    str (DATASET ),
    "--annotation",
    str (GT ),
    "--output-dir",
    str (output_dir ),
    "--device",
    "0",
    "--imgsz",
    "960",
    "--batch",
    "1",
    "--conf",
    "0.001",
    "--iou",
    "0.7",
    "--max-det",
    "300",
    "--limit",
    "0",
    "--save-vis",
    "0",
    ]

    print ("\n"+"="*100 )
    print ("EXPORTING:",key )
    print ("checkpoint:",item ["checkpoint"])
    print ("output:",output_dir )
    print ("="*100 )

    subprocess .run (
    command ,
    cwd =str (PROJECT ),
    check =True ,
    )

    if not prediction_path .is_file ():
        raise FileNotFoundError (prediction_path )

    if not metadata_path .is_file ():
        raise FileNotFoundError (metadata_path )


def execute ()->None :
    preflight ()

    WORK_OUTPUT .mkdir (
    parents =True ,
    exist_ok =True ,
    )

    progress_path =(
    WORK_OUTPUT /"progress.json"
    )

    progress ={
    "version":"final_canonical_clean_progress_v1",
    "status":"running",
    "updated_utc":datetime .now (
    timezone .utc 
    ).isoformat (),
    "completed_exports":[],
    "completed_evaluations":[],
    }

    if progress_path .is_file ():
        previous =json .loads (
        progress_path .read_text ()
        )
        if isinstance (previous ,dict ):
            progress .update (previous )
            progress ["status"]="running"

    write_json_atomic (
    progress_path ,
    progress ,
    )

    for key ,item in NEW_MODELS .items ():
        model_output =(
        WORK_OUTPUT 
        /"predictions"
        /key 
        )

        export_model (
        key ,
        item ,
        model_output ,
        )

        if key not in progress ["completed_exports"]:
            progress ["completed_exports"].append (key )

        progress ["updated_utc"]=datetime .now (
        timezone .utc 
        ).isoformat ()

        write_json_atomic (
        progress_path ,
        progress ,
        )

    print ("\n=== LOADING OFFICIAL-VAL GT ===")

    coco_gt =COCO (str (GT ))

    image_dimensions ={
    int (item ["id"]):(
    float (item ["width"]),
    float (item ["height"]),
    )
    for item in coco_gt .dataset ["images"]
    }
    category_ids ={
    int (item ["id"])
    for item in coco_gt .dataset ["categories"]
    }

    if len (image_dimensions )!=10_000 :
        raise AssertionError (
        len (image_dimensions )
        )

    registry ={}
    runtime_prediction_paths ={}

    for key ,item in REUSABLE .items ():
        path =item ["prediction"]

        audit =validate_predictions (
        path ,
        image_dimensions ,
        category_ids ,
        )

        if (
        audit ["prediction_sha256"]
        !=item ["prediction_sha256"]
        ):
            raise AssertionError (key )

        registry [key ]={
        "model_key":key ,
        "display_name":item ["display_name"],
        "prediction_location":"external_frozen",
        "prediction_path":str (path ),
        "prediction_sha256":audit [
        "prediction_sha256"
        ],
        "prediction_bytes":audit [
        "prediction_bytes"
        ],
        "audit":audit ,
        }
        runtime_prediction_paths [key ]=path 

    for key ,item in NEW_MODELS .items ():
        path =(
        WORK_OUTPUT 
        /"predictions"
        /key 
        /"predictions.json"
        )

        audit =validate_predictions (
        path ,
        image_dimensions ,
        category_ids ,
        )

        registry [key ]={
        "model_key":key ,
        "display_name":item ["display_name"],
        "checkpoint_path":str (
        item ["checkpoint"]
        ),
        "checkpoint_sha256":item [
        "checkpoint_sha256"
        ],
        "prediction_location":"internal",
        "prediction_path_relative":str (
        Path ("predictions")
        /key 
        /"predictions.json"
        ),
        "prediction_sha256":audit [
        "prediction_sha256"
        ],
        "prediction_bytes":audit [
        "prediction_bytes"
        ],
        "audit":audit ,
        }
        runtime_prediction_paths [key ]=path 

    write_json_atomic (
    WORK_OUTPUT /"prediction_registry.json",
    {
    "version":(
    "final_canonical_prediction_registry_v1"
    ),
    "status":"complete",
    "protocol_sha256":(
    "693c39375cc1eefc4f3ff9f957a8355f"
    "b947552a19dfef228cae6cfc21e3dcfe"
    ),
    "models":registry ,
    },
    )

    overall_rows =[]
    per_class_rows =[]

    ordered_keys =(
    list (REUSABLE .keys ())
    +list (NEW_MODELS .keys ())
    )

    for key in ordered_keys :
        if key in REUSABLE :
            display_name =REUSABLE [key ][
            "display_name"
            ]
        else :
            display_name =NEW_MODELS [key ][
            "display_name"
            ]

        audit =registry [key ]["audit"]

        overall ,per_class =evaluate_predictions (
        coco_gt ,
        runtime_prediction_paths [key ],
        key ,
        display_name ,
        audit ,
        )

        overall_rows .append (overall )
        per_class_rows .extend (per_class )

        evaluation_dir =(
        WORK_OUTPUT 
        /"evaluations"
        /key 
        )

        write_json_atomic (
        evaluation_dir /"metrics.json",
        overall ,
        )
        write_json_atomic (
        evaluation_dir 
        /"prediction_audit.json",
        audit ,
        )

        progress ["completed_evaluations"]=[
        row ["model_key"]
        for row in overall_rows 
        ]
        progress ["updated_utc"]=datetime .now (
        timezone .utc 
        ).isoformat ()

        write_json_atomic (
        progress_path ,
        progress ,
        )

    write_csv_atomic (
    WORK_OUTPUT /"canonical_clean_metrics.csv",
    overall_rows ,
    [
    "model_key",
    "display_name",
    *METRIC_NAMES ,
    "predictions",
    "prediction_sha256",
    "elapsed_minutes",
    ],
    )

    write_csv_atomic (
    WORK_OUTPUT 
    /"canonical_clean_per_class_metrics.csv",
    per_class_rows ,
    [
    "model_key",
    "display_name",
    "category_id",
    "class_name",
    "AP",
    "AP50",
    "AP75",
    "AR100",
    "num_gt",
    "num_images_with_gt",
    "num_predictions",
    ],
    )

    summary ={
    "version":"final_canonical_clean_summary_v1",
    "status":"complete",
    "created_utc":datetime .now (
    timezone .utc 
    ).isoformat (),
    "protocol_sha256":(
    "693c39375cc1eefc4f3ff9f957a8355f"
    "b947552a19dfef228cae6cfc21e3dcfe"
    ),
    "dataset":{
    "name":"BDD100K official validation",
    "images":10_000 ,
    "annotations":185_523 ,
    "gt_path":str (GT ),
    "gt_sha256":GT_SHA256 ,
    },
    "canonical_configuration":{
    "image_size":960 ,
    "prediction_confidence_floor":0.001 ,
    "native_nms_iou":0.7 ,
    "native_max_detections":300 ,
    "coco_max_detections":100 ,
    },
    "models":{
    row ["model_key"]:row 
    for row in overall_rows 
    },
    }

    write_json_atomic (
    WORK_OUTPUT /"canonical_clean_summary.json",
    summary ,
    )

    progress ["status"]="complete"
    progress ["updated_utc"]=datetime .now (
    timezone .utc 
    ).isoformat ()

    write_json_atomic (
    progress_path ,
    progress ,
    )

    files ={}

    for path in sorted (WORK_OUTPUT .rglob ("*")):
        if (
        not path .is_file ()
        or path .name =="artifact_manifest.json"
        ):
            continue 

        relative =str (
        path .relative_to (WORK_OUTPUT )
        )

        files [relative ]={
        "bytes":path .stat ().st_size ,
        "sha256":sha256 (path ),
        }

    manifest ={
    "version":(
    "final_canonical_clean_artifact_manifest_v1"
    ),
    "status":"complete",
    "created_utc":datetime .now (
    timezone .utc 
    ).isoformat (),
    "protocol_sha256":(
    "693c39375cc1eefc4f3ff9f957a8355f"
    "b947552a19dfef228cae6cfc21e3dcfe"
    ),
    "files":files ,
    }

    write_json_atomic (
    WORK_OUTPUT /"artifact_manifest.json",
    manifest ,
    )

    os .replace (
    WORK_OUTPUT ,
    FINAL_OUTPUT ,
    )

    final_manifest =(
    FINAL_OUTPUT /"artifact_manifest.json"
    )

    print ("\n"+"="*100 )
    print ("FINAL CANONICAL-CLEAN EVALUATION COMPLETE")
    print ("="*100 )

    for row in overall_rows :
        print (
        f"{row ['model_key']:20s} "
        f"AP={row ['AP']:.6f} "
        f"APs={row ['AP_small']:.6f} "
        f"AP75={row ['AP75']:.6f} "
        f"AR100={row ['AR100']:.6f}"
        )

    print ("output:",FINAL_OUTPUT )
    print ("manifest:",final_manifest )
    print ("manifest_sha256:",sha256 (final_manifest ))
    print ("new_predictions_retained:",len (NEW_MODELS ))
    print ("new_training_performed:",False )
    print (
    "NEXT: freeze official-val failure subsets "
    "from baseline predictions"
    )
    print ("="*100 )


def main ()->None :
    parser =argparse .ArgumentParser ()

    group =parser .add_mutually_exclusive_group (
    required =True 
    )
    group .add_argument (
    "--preflight-only",
    action ="store_true",
    )
    group .add_argument (
    "--execute",
    action ="store_true",
    )

    args =parser .parse_args ()

    if args .preflight_only :
        preflight ()
        print ("NOTHING CREATED, MODIFIED, EVALUATED OR TRAINED")
        return 

    execute ()


if __name__ =="__main__":
    main ()
