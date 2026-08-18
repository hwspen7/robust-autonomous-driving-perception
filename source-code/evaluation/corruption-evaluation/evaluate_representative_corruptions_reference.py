from __future__ import annotations 

import argparse 
import csv 
import hashlib 
import importlib .util 
import json 
import os 
import shutil 
import subprocess 
import sys 
import time 
from collections import defaultdict 
from datetime import datetime ,timezone 
from pathlib import Path 

import numpy as np 


PROJECT =Path (
"/root/rivermind-data/autodrive/code/"
"robust-autonomous-driving-perception"
)
EVALUATION =Path (
"/root/rivermind-data/autodrive/results/evaluation"
)
FINAL_ROOT =EVALUATION /"final_analysis_v1"
DATASET =Path (
"/root/rivermind-data/autodrive/datasets/datasets/"
"bdd100k_final"
)

PROTOCOL =(
FINAL_ROOT 
/"frozen_protocol/final_testing_protocol_v1.json"
)
CANONICAL_ROOT =FINAL_ROOT /"canonical_clean_v1"
CANONICAL_MANIFEST =(
CANONICAL_ROOT /"artifact_manifest.json"
)
CANONICAL_SUMMARY =(
CANONICAL_ROOT /"canonical_clean_summary.json"
)
SCALE_CONTEXT_MANIFEST =(
FINAL_ROOT 
/"crop_scale_context_audit_v1/"
"artifact_manifest.json"
)

GT =(
DATASET 
/"coco/annotations/instances_val.json"
)
EXPORTER =(
PROJECT 
/"scripts/export_yolo11m_controlled_corruptions.py"
)
CORRUPTION_SOURCE =(
PROJECT /"scripts/controlled_corruptions.py"
)
FROZEN_BENCHMARK =(
PROJECT 
/"scripts/run_controlled_corruption_benchmark.py"
)

OUTPUT =(
FINAL_ROOT /"representative_corruptions_v1"
)
WORK =OUTPUT .with_name (
OUTPUT .name +"-work-693c39375cc1"
)
TEMP_ROOT =Path (
"/dev/shm/rebu_yolo/"
"final_representative_corruptions_v1"
)

EXPECTED_SHA256 ={
PROTOCOL :
"693c39375cc1eefc4f3ff9f957a8355fb947552a19dfef228cae6cfc21e3dcfe",
CANONICAL_MANIFEST :
"88d2bace7a2f69e26c34e7eed4e2814b12e8317980602968b407d1b94282338b",
SCALE_CONTEXT_MANIFEST :
"30a8c5043acab642bd025802ffd318e31515422f9671a5a54a56cbd820de558d",
GT :
"7be8a02e147743123a4a62b2c92ed2e4a5712d261c8868223e6325bad6e346d6",
EXPORTER :
"ce8f84f3ac4e687cdf9127fa2a538e81606005b8329c763e6fe66df579a3f44e",
CORRUPTION_SOURCE :
"cb451b3ab1c5ba134bd5251298293f38f645d279d30614c7dbba7ecac550273a",
FROZEN_BENCHMARK :
"caaf850b01b785365185c847af7942144d29d49061cc457f3896ce417949ad84",
}

MODEL_SPECS ={
"standard_e20":{
"display_name":"Standard-960-E20",
"clean_key":"standard_e20",
"checkpoint":(
EVALUATION 
/"architecture_gate/formal_gate_v4/"
"standard/weights/resume_epoch20.pt"
),
"sha256":
"dc1e3ef68e233a2fad4911955bbe178f0104cd9000dcbdc772dac8d0a58438b5",
},
"p2_e20":{
"display_name":"P2-960-E20",
"clean_key":"p2_e20",
"checkpoint":(
EVALUATION 
/"architecture_gate/formal_gate_v4/"
"p2_projected/weights/resume_epoch20.pt"
),
"sha256":
"642a6b4009f085649b6da3f1647821d3e760e6724b019e08d022080c8050bc29",
},
"common40":{
"display_name":"P2-960-Common40",
"clean_key":"common40",
"checkpoint":(
EVALUATION 
/"formal_training/training_v1/"
"common_uniform_e1_40/weights/common40_ema.pt"
),
"sha256":
"c05ed70a608d026fef6178f40a4dee62e96323731700b42e426c311dd78b27f3",
},
"uniform_e56":{
"display_name":"Uniform-E56",
"clean_key":"uniform_e56",
"checkpoint":(
EVALUATION 
/"formal_training/training_v2/"
"uniform_e41_100/weights/epoch15.pt"
),
"sha256":
"84aa86771d185c2c72fa277c609c1222d7a417a8197523c4a5ced2a6a91ebb1b",
},
"rebu_risk_e56":{
"display_name":"Rebu-Risk-E56",
"clean_key":"risk_e56",
"checkpoint":(
EVALUATION 
/"formal_training/training_v2/"
"rebu_risk_e41_100/weights/epoch15.pt"
),
"sha256":
"879b179c219914a83775cacb7074f1d7bd9836971fae24592a6a16d6e87fe102",
},
}

METRICS =[
"AP",
"AP50",
"AP75",
"AP_small",
"AP_medium",
"AP_large",
"AR_100",
"AR_small",
"AR_medium",
"AR_large",
]

PAIR_SPECS ={
"p2_minus_standard":(
"p2_e20",
"standard_e20",
),
"rebu_risk_minus_uniform":(
"rebu_risk_e56",
"uniform_e56",
),
}


def parse_args ():
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

    parser .add_argument (
    "--batch",
    type =int ,
    default =8 ,
    )

    return parser .parse_args ()


def now_utc ():
    return datetime .now (timezone .utc ).isoformat ()


def sha256_file (path :Path ):
    digest =hashlib .sha256 ()

    with path .open ("rb")as handle :
        while True :
            block =handle .read (8 *1024 *1024 )
            if not block :
                break 
            digest .update (block )

    return digest .hexdigest ()


def verify (path :Path ,expected :str ):
    if not path .is_file ():
        raise FileNotFoundError (path )

    actual =sha256_file (path )

    print (path )
    print (" expected:",expected )
    print (" actual  :",actual )

    if actual !=expected :
        raise AssertionError (path )


def load_json (path :Path ):
    return json .loads (path .read_text ())


def write_json_atomic (path :Path ,payload ):
    path .parent .mkdir (
    parents =True ,
    exist_ok =True ,
    )

    temporary =path .with_suffix (
    path .suffix +".tmp"
    )

    temporary .write_text (
    json .dumps (
    payload ,
    indent =2 ,
    ensure_ascii =False ,
    sort_keys =True ,
    allow_nan =False ,
    )+"\n",
    encoding ="utf-8",
    )

    os .replace (temporary ,path )


def write_csv_atomic (
path :Path ,
rows :list [dict ],
):
    if not rows :
        raise ValueError (path )

    path .parent .mkdir (
    parents =True ,
    exist_ok =True ,
    )

    temporary =path .with_suffix (
    path .suffix +".tmp"
    )

    with temporary .open (
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

    os .replace (temporary ,path )


def verify_manifest_member (
root :Path ,
manifest_path :Path ,
relative :str ,
):
    manifest =load_json (manifest_path )

    if "files"in manifest :
        record =manifest ["files"][relative ]
    elif "artifacts"in manifest :
        matches =[
        row 
        for row in manifest ["artifacts"]
        if row ["path"]==relative 
        ]

        if len (matches )!=1 :
            raise KeyError (
            (manifest_path ,relative )
            )

        record =matches [0 ]
    else :
        raise KeyError (manifest_path )

    path =root /relative 

    if not path .is_file ():
        raise FileNotFoundError (path )

    if path .stat ().st_size !=int (record ["bytes"]):
        raise AssertionError (path )

    if sha256_file (path )!=record ["sha256"]:
        raise AssertionError (path )


def load_benchmark_module ():
    spec =importlib .util .spec_from_file_location (
    "frozen_corruption_benchmark",
    FROZEN_BENCHMARK ,
    )

    if spec is None or spec .loader is None :
        raise ImportError (FROZEN_BENCHMARK )

    module =importlib .util .module_from_spec (spec )
    spec .loader .exec_module (module )

    return module 


def load_clean_metrics ():
    summary =load_json (CANONICAL_SUMMARY )

    if summary ["status"]!="complete":
        raise AssertionError (summary ["status"])

    if (
    summary ["protocol_sha256"]
    !=EXPECTED_SHA256 [PROTOCOL ]
    ):
        raise AssertionError (
        summary ["protocol_sha256"]
        )

    result ={}

    for model_key ,spec in MODEL_SPECS .items ():
        clean_key =spec ["clean_key"]

        if clean_key not in summary ["models"]:
            raise KeyError (clean_key )

        row =summary ["models"][clean_key ]

        result [model_key ]={
        field :float (row [field ])
        for field in (
        "AP",
        "AP50",
        "AP75",
        "AP_small",
        "AP_medium",
        "AP_large",
        )
        }

    return result 


def preflight (batch :int ):
    print (
    "=== REPRESENTATIVE CORRUPTION "
    "EXECUTOR PREFLIGHT ==="
    )

    if batch <=0 :
        raise ValueError (batch )

    for path ,expected in EXPECTED_SHA256 .items ():
        verify (path ,expected )

    verify_manifest_member (
    CANONICAL_ROOT ,
    CANONICAL_MANIFEST ,
    "canonical_clean_summary.json",
    )

    for model_key ,spec in MODEL_SPECS .items ():
        verify (
        spec ["checkpoint"],
        spec ["sha256"],
        )
        print ("PASS checkpoint:",model_key )

    protocol =load_json (PROTOCOL )
    frozen =protocol ["corruption_evaluation"]

    if frozen ["models"]!=list (MODEL_SPECS ):
        raise AssertionError (
        frozen ["models"]
        )

    if frozen ["corruptions"]!=[
    "blur",
    "noise",
    "fog",
    "rain",
    "low_light",
    ]:
        raise AssertionError (
        frozen ["corruptions"]
        )

    if frozen ["severities"]!=[1 ,2 ,3 ]:
        raise AssertionError (
        frozen ["severities"]
        )

    if (
    frozen ["total_model_condition_runs"]
    !=75 
    ):
        raise AssertionError (frozen )

    clean =load_clean_metrics ()

    if set (clean )!=set (MODEL_SPECS ):
        raise AssertionError (clean )

    if OUTPUT .exists ():
        raise FileExistsError (OUTPUT )

    progress_path =WORK /"progress.json"

    completed =0 

    if progress_path .is_file ():
        progress =load_json (progress_path )

        completed =sum (
        1 
        for value in progress [
        "completed"
        ].values ()
        if value .get ("status")=="done"
        )

    if TEMP_ROOT .exists ()and not WORK .exists ():
        raise FileExistsError (
        "Temporary corruption directory exists "
        "without matching resumable work output:\n"
        f"{TEMP_ROOT }"
        )

    usage =shutil .disk_usage (
    "/root/rivermind-data"
    )
    shm_usage =shutil .disk_usage (
    "/dev/shm"
    )

    print ("batch:",batch )
    print ("formal_output_exists:",OUTPUT .exists ())
    print ("resumable_work_exists:",WORK .exists ())
    print ("completed_conditions:",completed )
    print (
    "data_disk_free_GiB:",
    round (usage .free /1024 **3 ,3 ),
    )
    print (
    "shm_free_GiB:",
    round (shm_usage .free /1024 **3 ,3 ),
    )

    print (
    "PASS: 5 checkpoints, 15 corruption "
    "conditions and canonical clean references "
    "are valid"
    )
    print (
    "NOTHING CREATED, MODIFIED, "
    "INFERRED OR TRAINED"
    )

    return protocol ,clean 


def safe_remove_temp (path :Path ):
    if not path .exists ():
        return 

    if path .parent !=TEMP_ROOT :
        raise AssertionError (path )

    shutil .rmtree (path )


def run_exporter (
model_key :str ,
checkpoint :Path ,
corruption :str ,
severity :int ,
output_dir :Path ,
batch :int ,
):
    command =[
    sys .executable ,
    "-u",
    str (EXPORTER ),
    "--model",
    str (checkpoint ),
    "--dataset-root",
    str (DATASET ),
    "--annotation",
    str (GT ),
    "--output-dir",
    str (output_dir ),
    "--corruption",
    corruption ,
    "--severity",
    str (severity ),
    "--device",
    "0",
    "--imgsz",
    "960",
    "--batch",
    str (batch ),
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
    "--overwrite",
    ]

    print ()
    print ("="*100 )
    print (
    "INFERENCE:",
    model_key ,
    corruption ,
    severity ,
    )
    print ("="*100 )

    started =time .time ()

    subprocess .run (
    command ,
    cwd =str (PROJECT ),
    check =True ,
    )

    return time .time ()-started 


def validate_export (
temp_dir :Path ,
model_key :str ,
corruption :str ,
severity :int ,
batch :int ,
benchmark ,
gt_id_set :set [int ],
):
    prediction =temp_dir /"predictions.json"
    image_summary =temp_dir /"image_summary.json"
    metadata_path =temp_dir /"metadata.json"

    for path in (
    prediction ,
    image_summary ,
    metadata_path ,
    ):
        if not path .is_file ():
            raise FileNotFoundError (path )

    benchmark .validate_summary (
    image_summary ,
    gt_id_set ,
    )

    metadata =load_json (metadata_path )
    spec =MODEL_SPECS [model_key ]

    if (
    metadata ["checkpoint_sha256"]
    !=spec ["sha256"]
    ):
        raise AssertionError (
        metadata ["checkpoint_sha256"]
        )

    controlled =metadata [
    "controlled_corruption"
    ]

    if controlled ["corruption"]!=corruption :
        raise AssertionError (controlled )

    if int (controlled ["severity"])!=severity :
        raise AssertionError (controlled )

    inference =metadata ["inference"]

    expected_inference ={
    "imgsz":960 ,
    "batch":batch ,
    "conf":0.001 ,
    "nms_iou":0.7 ,
    "max_det":300 ,
    }

    for key ,expected in expected_inference .items ():
        actual =inference [key ]

        if isinstance (expected ,float ):
            if abs (float (actual )-expected )>1e-12 :
                raise AssertionError (
                (key ,actual ,expected )
                )
        elif int (actual )!=expected :
            raise AssertionError (
            (key ,actual ,expected )
            )

    if int (metadata ["num_gt_images"])!=10_000 :
        raise AssertionError (metadata )

    if (
    int (metadata ["num_gt_annotations"])
    !=185_523 
    ):
        raise AssertionError (metadata )

    return (
    prediction ,
    image_summary ,
    metadata_path ,
    metadata ,
    )


def save_derived_outputs (
benchmark ,
metrics :list [dict ],
clean_metrics :dict ,
):
    if len (metrics )!=75 :
        raise AssertionError (len (metrics ))

    keys ={
    (
    row ["model"],
    row ["corruption"],
    int (row ["severity"]),
    )
    for row in metrics 
    }

    if len (keys )!=75 :
        raise AssertionError (
        "Duplicate model-condition rows."
        )

    family_rows =[]

    grouped =defaultdict (list )

    for row in metrics :
        grouped [
        (
        row ["model"],
        row ["corruption"],
        )
        ].append (row )

    for (
    model_key ,
    corruption ,
    ),rows in sorted (grouped .items ()):
        if len (rows )!=3 :
            raise AssertionError (
            (model_key ,corruption ,len (rows ))
            )

        family_row ={
        "model":model_key ,
        "corruption":corruption ,
        "conditions":len (rows ),
        }

        for metric in METRICS :
            family_row [
            f"mean_{metric }"
            ]=float (np .mean ([
            float (row [metric ])
            for row in rows 
            ]))

        family_row ["mean_AP_retention"]=float (
        np .mean ([
        float (row ["AP_retention"])
        for row in rows 
        ])
        )
        family_row [
        "mean_AP_small_retention"
        ]=float (
        np .mean ([
        float (
        row ["AP_small_retention"]
        )
        for row in rows 
        ])
        )

        family_rows .append (family_row )

    comparison_rows =[]

    by_condition ={
    (
    row ["model"],
    row ["corruption"],
    int (row ["severity"]),
    ):row 
    for row in metrics 
    }

    corruptions =[
    "blur",
    "noise",
    "fog",
    "rain",
    "low_light",
    ]

    for comparison ,(
    numerator ,
    denominator ,
    )in PAIR_SPECS .items ():
        for corruption in corruptions :
            for severity in (1 ,2 ,3 ):
                numerator_row =by_condition [
                (
                numerator ,
                corruption ,
                severity ,
                )
                ]
                denominator_row =by_condition [
                (
                denominator ,
                corruption ,
                severity ,
                )
                ]

                row ={
                "comparison":comparison ,
                "numerator":numerator ,
                "denominator":denominator ,
                "corruption":corruption ,
                "severity":severity ,
                }

                for metric in METRICS :
                    row [f"delta_{metric }"]=(
                    float (numerator_row [metric ])
                    -float (denominator_row [metric ])
                    )

                row ["delta_AP_retention"]=(
                float (
                numerator_row [
                "AP_retention"
                ]
                )
                -float (
                denominator_row [
                "AP_retention"
                ]
                )
                )

                row [
                "delta_AP_small_retention"
                ]=(
                float (
                numerator_row [
                "AP_small_retention"
                ]
                )
                -float (
                denominator_row [
                "AP_small_retention"
                ]
                )
                )

                comparison_rows .append (row )

    model_summary ={}

    for model_key in MODEL_SPECS :
        rows =[
        row 
        for row in metrics 
        if row ["model"]==model_key 
        ]

        if len (rows )!=15 :
            raise AssertionError (
            (model_key ,len (rows ))
            )

        model_summary [model_key ]={
        "conditions":len (rows ),
        "clean_metrics":
        clean_metrics [model_key ],
        **{
        f"mean_corruption_{metric }":
        float (np .mean ([
        float (row [metric ])
        for row in rows 
        ]))
        for metric in METRICS 
        },
        "mean_AP_retention":
        float (np .mean ([
        float (row ["AP_retention"])
        for row in rows 
        ])),
        "mean_AP_small_retention":
        float (np .mean ([
        float (
        row [
        "AP_small_retention"
        ]
        )
        for row in rows 
        ])),
        }

    comparison_summary ={}

    for comparison in PAIR_SPECS :
        rows =[
        row 
        for row in comparison_rows 
        if row ["comparison"]==comparison 
        ]

        comparison_summary [comparison ]={
        "conditions":len (rows ),
        **{
        f"mean_delta_{metric }":
        float (np .mean ([
        float (row [f"delta_{metric }"])
        for row in rows 
        ]))
        for metric in METRICS 
        },
        "positive_AP_condition_fraction":
        sum (
        float (row ["delta_AP"])>0.0 
        for row in rows 
        )/len (rows ),
        "positive_AP_small_condition_fraction":
        sum (
        float (
        row ["delta_AP_small"]
        )>0.0 
        for row in rows 
        )/len (rows ),
        }

    summary ={
    "version":(
    "final_representative_corruptions_v1"
    ),
    "status":"complete",
    "created_utc":now_utc (),
    "protocol_sha256":
    EXPECTED_SHA256 [PROTOCOL ],
    "models":model_summary ,
    "comparisons":comparison_summary ,
    "interpretation_limits":{
    "new_training":False ,
    "corruptions_are_controlled_synthetic":
    True ,
    "official_val_is_untouched_test_set":
    False ,
    "single_training_seed":
    True ,
    "causal_claim_from_corruption_results":
    False ,
    },
    }

    write_csv_atomic (
    WORK /"model_corruption_family_summary.csv",
    family_rows ,
    )
    write_csv_atomic (
    WORK /"model_condition_comparisons.csv",
    comparison_rows ,
    )
    write_json_atomic (
    WORK /"summary.json",
    summary ,
    )

    return summary 


def build_manifest (directory :Path ):
    files ={}

    for path in sorted (directory .rglob ("*")):
        if not path .is_file ():
            continue 

        if path .name =="artifact_manifest.json":
            continue 

        relative =str (
        path .relative_to (directory )
        )

        files [relative ]={
        "bytes":path .stat ().st_size ,
        "sha256":sha256_file (path ),
        }

    manifest ={
    "version":(
    "final_representative_corruptions_"
    "artifact_manifest_v1"
    ),
    "status":"complete",
    "created_utc":now_utc (),
    "protocol_sha256":
    EXPECTED_SHA256 [PROTOCOL ],
    "files":files ,
    }

    path =directory /"artifact_manifest.json"
    write_json_atomic (path ,manifest )

    return path ,sha256_file (path )


def execute (
protocol :dict ,
clean_metrics :dict ,
batch :int ,
):
    benchmark =load_benchmark_module ()

    benchmark .GT_PATH =GT 
    benchmark .CLEAN =clean_metrics 
    benchmark .VARIANTS =[
    (corruption ,severity )
    for corruption in protocol [
    "corruption_evaluation"
    ]["corruptions"]
    for severity in protocol [
    "corruption_evaluation"
    ]["severities"]
    ]

    (
    gt_ids ,
    gt_id_set ,
    gt_counts ,
    )=benchmark .validate_gt ()

    WORK .mkdir (
    parents =True ,
    exist_ok =True ,
    )
    (WORK /"variant_records").mkdir (
    parents =True ,
    exist_ok =True ,
    )
    TEMP_ROOT .mkdir (
    parents =True ,
    exist_ok =True ,
    )

    metric_json =WORK /"corruption_metrics.json"
    class_json =(
    WORK 
    /"corruption_per_class_metrics.json"
    )
    progress_path =WORK /"progress.json"

    metrics =benchmark .load_existing (metric_json )
    per_class =benchmark .load_existing (class_json )

    if progress_path .is_file ():
        progress =load_json (progress_path )
    else :
        progress ={
        "version":(
        "representative_corruption_progress_v1"
        ),
        "status":"running",
        "protocol_sha256":
        EXPECTED_SHA256 [PROTOCOL ],
        "batch":batch ,
        "completed":{},
        "created_utc":now_utc (),
        }

    if int (progress ["batch"])!=batch :
        raise AssertionError ({
        "existing_batch":progress ["batch"],
        "requested_batch":batch ,
        })

    total =75 
    completed_before =sum (
    benchmark .has_metric (
    metrics ,
    model_key ,
    benchmark .vname (
    corruption ,
    severity ,
    ),
    )
    for model_key in MODEL_SPECS 
    for corruption ,severity 
    in benchmark .VARIANTS 
    )

    print ()
    print ("="*100 )
    print (
    "REPRESENTATIVE CHECKPOINT "
    "CORRUPTION EVALUATION"
    )
    print ("="*100 )
    print ("total_conditions:",total )
    print ("completed_before:",completed_before )
    print ("batch:",batch )
    print ("temporary_root:",TEMP_ROOT )
    print ("resumable_output:",WORK )
    print ("="*100 )

    run_index =0 

    for model_key ,spec in MODEL_SPECS .items ():
        for corruption ,severity in benchmark .VARIANTS :
            run_index +=1 

            variant =benchmark .vname (
            corruption ,
            severity ,
            )

            if benchmark .has_metric (
            metrics ,
            model_key ,
            variant ,
            ):
                print (
                f"[SKIP {run_index }/{total }] "
                f"{model_key } {variant }"
                )
                continue 

            temp_dir =(
            TEMP_ROOT 
            /f"{model_key }_{variant }"
            )
            request_path =(
            temp_dir /"request.json"
            )

            expected_request ={
            "model":model_key ,
            "checkpoint_sha256":
            spec ["sha256"],
            "corruption":corruption ,
            "severity":severity ,
            "image_size":960 ,
            "batch":batch ,
            "confidence_floor":0.001 ,
            "nms_iou":0.7 ,
            "max_det":300 ,
            }

            reusable =False 

            if request_path .is_file ():
                reusable =(
                load_json (request_path )
                ==expected_request 
                and (
                temp_dir 
                /"predictions.json"
                ).is_file ()
                and (
                temp_dir 
                /"image_summary.json"
                ).is_file ()
                and (
                temp_dir 
                /"metadata.json"
                ).is_file ()
                )

            if reusable :
                print (
                f"[RESUME TEMP {run_index }/{total }] "
                f"{model_key } {variant }"
                )
                inference_seconds =float (
                load_json (
                temp_dir /"metadata.json"
                )["elapsed_seconds"]
                )
            else :
                if temp_dir .exists ():
                    safe_remove_temp (temp_dir )

                temp_dir .mkdir (parents =True )

                write_json_atomic (
                request_path ,
                expected_request ,
                )

                progress ["completed"][
                f"{model_key }::{variant }"
                ]={
                "status":"inference_started",
                "updated_utc":now_utc (),
                }
                progress ["updated_utc"]=now_utc ()

                benchmark .save_outputs (
                WORK ,
                metrics ,
                per_class ,
                progress ,
                )

                inference_seconds =run_exporter (
                model_key =model_key ,
                checkpoint =spec ["checkpoint"],
                corruption =corruption ,
                severity =severity ,
                output_dir =temp_dir ,
                batch =batch ,
                )

            (
            prediction_path ,
            _ ,
            metadata_path ,
            exporter_metadata ,
            )=validate_export (
            temp_dir =temp_dir ,
            model_key =model_key ,
            corruption =corruption ,
            severity =severity ,
            batch =batch ,
            benchmark =benchmark ,
            gt_id_set =gt_id_set ,
            )

            metric_row ,class_rows =benchmark .evaluate (
            model =model_key ,
            variant =variant ,
            corruption =corruption ,
            severity =severity ,
            prediction_path =prediction_path ,
            gt_ids =gt_ids ,
            gt_count_by_class =gt_counts ,
            inference_seconds =inference_seconds ,
            )

            metrics =benchmark .replace_rows (
            metrics ,
            model_key ,
            variant ,
            [metric_row ],
            )
            per_class =benchmark .replace_rows (
            per_class ,
            model_key ,
            variant ,
            class_rows ,
            )

            record ={
            "version":(
            "representative_corruption_"
            "condition_record_v1"
            ),
            "status":"complete",
            "model":model_key ,
            "display_name":
            spec ["display_name"],
            "variant":variant ,
            "metric":metric_row ,
            "exporter_metadata":
            exporter_metadata ,
            "temporary_prediction_deleted":
            True ,
            "saved_utc":now_utc (),
            }

            write_json_atomic (
            WORK 
            /"variant_records"
            /f"{model_key }_{variant }.json",
            record ,
            )

            progress ["completed"][
            f"{model_key }::{variant }"
            ]={
            "status":"done",
            "AP":metric_row ["AP"],
            "AP_small":
            metric_row ["AP_small"],
            "prediction_sha256":
            metric_row [
            "prediction_sha256"
            ],
            "updated_utc":now_utc (),
            }
            progress ["updated_utc"]=now_utc ()

            benchmark .save_outputs (
            WORK ,
            metrics ,
            per_class ,
            progress ,
            )

            safe_remove_temp (temp_dir )

            print (
            f"[DONE {run_index }/{total }] "
            f"{model_key } {variant } "
            f"AP={metric_row ['AP']:.6f} "
            f"APs={metric_row ['AP_small']:.6f}"
            )

    if len (metrics )!=75 :
        raise AssertionError (len (metrics ))

    if len (per_class )!=750 :
        raise AssertionError (len (per_class ))

    if list (TEMP_ROOT .rglob ("predictions.json")):
        raise AssertionError (
        "Temporary predictions remain."
        )

    summary =save_derived_outputs (
    benchmark ,
    metrics ,
    clean_metrics ,
    )

    progress ["status"]="complete"
    progress ["completed_conditions"]=75 
    progress ["updated_utc"]=now_utc ()

    benchmark .save_outputs (
    WORK ,
    metrics ,
    per_class ,
    progress ,
    )

    metadata ={
    "version":(
    "final_representative_corruptions_"
    "executor_v1"
    ),
    "status":"complete",
    "created_utc":now_utc (),
    "script":str (Path (__file__ ).resolve ()),
    "script_sha256":
    sha256_file (Path (__file__ ).resolve ()),
    "protocol_sha256":
    EXPECTED_SHA256 [PROTOCOL ],
    "exporter_sha256":
    EXPECTED_SHA256 [EXPORTER ],
    "corruption_source_sha256":
    EXPECTED_SHA256 [
    CORRUPTION_SOURCE 
    ],
    "frozen_benchmark_sha256":
    EXPECTED_SHA256 [
    FROZEN_BENCHMARK 
    ],
    "configuration":{
    "models":list (MODEL_SPECS ),
    "corruptions":[
    "blur",
    "noise",
    "fog",
    "rain",
    "low_light",
    ],
    "severities":[1 ,2 ,3 ],
    "conditions":75 ,
    "image_size":960 ,
    "batch":batch ,
    "confidence_floor":0.001 ,
    "nms_iou":0.7 ,
    "native_max_det":300 ,
    "coco_max_det":100 ,
    },
    "storage_policy":{
    "corruptions_generated_on_the_fly":
    True ,
    "full_corrupted_dataset_saved":
    False ,
    "one_temporary_prediction_at_a_time":
    True ,
    "temporary_prediction_root":
    str (TEMP_ROOT ),
    "temporary_predictions_remaining":
    0 ,
    },
    "new_training":False ,
    }

    write_json_atomic (
    WORK /"benchmark_metadata.json",
    metadata ,
    )

    manifest_path ,manifest_sha =build_manifest (
    WORK 
    )

    os .replace (WORK ,OUTPUT )

    if TEMP_ROOT .exists ():
        try :
            TEMP_ROOT .rmdir ()
        except OSError :
            pass 

    print ()
    print ("="*100 )
    print (
    "REPRESENTATIVE CORRUPTION "
    "EVALUATION COMPLETE"
    )
    print ("="*100 )

    for model_key ,row in summary ["models"].items ():
        print (
        model_key ,
        "mean_AP=",
        round (
        row ["mean_corruption_AP"],
        6 ,
        ),
        "mean_APs=",
        round (
        row [
        "mean_corruption_AP_small"
        ],
        6 ,
        ),
        "AP_retention=",
        round (
        row ["mean_AP_retention"],
        6 ,
        ),
        )

    print ("output:",OUTPUT )
    print (
    "manifest:",
    OUTPUT /manifest_path .name ,
    )
    print ("manifest_sha256:",manifest_sha )
    print ("temporary_predictions_remaining: 0")
    print ("NOTHING TRAINED")
    print ("NEXT: frozen efficiency evaluation")
    print ("="*100 )


def main ():
    args =parse_args ()

    protocol ,clean =preflight (args .batch )

    if args .preflight_only :
        return 

    execute (
    protocol =protocol ,
    clean_metrics =clean ,
    batch =args .batch ,
    )


if __name__ =="__main__":
    main ()
