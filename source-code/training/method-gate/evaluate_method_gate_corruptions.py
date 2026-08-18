from __future__ import annotations 

from pathlib import Path 
import argparse 
import gc 
import hashlib 
import json 
import os 
import shutil 
import sys 
import time 
import numpy as np 

import torch 
from pycocotools .coco import COCO 
from pycocotools .cocoeval import COCOeval 

import export_yolo11m_controlled_corruptions as exporter 


ROOT =Path (
"/root/rivermind-data/autodrive/results/evaluation"
)
REPO =Path (
"/root/rivermind-data/autodrive/code/"
"robust-autonomous-driving-perception"
)
METHOD_ROOT =ROOT /"method_gate"

PROTOCOL =(
METHOD_ROOT /"frozen_manifests/method_gate_protocol_v1.json"
)
GT =(
ROOT /"architecture_gate/prepared_inputs_v3/"
"instances_train_dev_effective.json"
)
CONTROLLED =REPO /"scripts/controlled_corruptions.py"
EXPORTER =REPO /"scripts/export_yolo11m_controlled_corruptions.py"

CLEAN_MANIFEST =(
METHOD_ROOT /"evaluation_v2_clean/clean_artifact_manifest.json"
)
FAILURE_MANIFEST =(
METHOD_ROOT /"evaluation_v3_failures/artifact_manifest.json"
)
FAILURE_RESULT =(
METHOD_ROOT /"evaluation_v3_failures/failure_evaluation.json"
)

OUTPUT =METHOD_ROOT /"evaluation_v4_corruptions"

DATASET_ROOT =Path (
"/root/rivermind-data/autodrive/datasets/datasets/bdd100k_final"
)

MODELS ={
"uniform":(
METHOD_ROOT /"training_v1/uniform/weights/last.pt",
"45f397e81cf6aa3e8fbb08a8fef79b8e95975f85f5b360fa9f4bf2a4f6757210",
),
"scalar_risk":(
METHOD_ROOT /"training_v1/scalar_risk/weights/last.pt",
"3093b249d8052924d8324c826731a77b82ab98327a41ff68ab3a2ad615f74b9d",
),
"typed_cafr":(
METHOD_ROOT /"training_v1/typed_cafr/weights/last.pt",
"217b6d453d399c345a7e1be5595783044f2555a65ae95f047d6a7278d105c4ee",
),
}

EXPECTED ={
PROTOCOL :
"6be7ae25550360e38195bb785a064bd19cb9c4a65b4625b30b5698861cc1b00a",
GT :
"5e13ec651d0582032e5eacc1a46d885279fabe278463bd7c2be58b2429796faf",
CONTROLLED :
"cb451b3ab1c5ba134bd5251298293f38f645d279d30614c7dbba7ecac550273a",
EXPORTER :
"ce8f84f3ac4e687cdf9127fa2a538e81606005b8329c763e6fe66df579a3f44e",
CLEAN_MANIFEST :
"29dc8882d383c37c6e9eb5a8cf826b9d562a3c48e9d0ef8d5a23be5b78350f97",
FAILURE_MANIFEST :
"2e57f0e7cb31731362511ca8d29fa9fa52328336241aea696a322eee9f5c2457",
}

FAMILIES =("blur","noise","low_light")
SEVERITY =2 


def sha256 (path :Path )->str :
    digest =hashlib .sha256 ()
    with path .open ("rb")as handle :
        while block :=handle .read (8 *1024 *1024 ):
            digest .update (block )
    return digest .hexdigest ()


def verify (path :Path ,expected :str )->None :
    if not path .is_file ():
        raise FileNotFoundError (path )

    actual =sha256 (path )
    print (path )
    print (" expected:",expected )
    print (" actual  :",actual )

    if actual !=expected :
        raise AssertionError (path )


def write_json (path :Path ,payload )->None :
    temporary =path .with_name (
    f"{path .name }.incomplete-{os .getpid ()}"
    )
    temporary .write_text (
    json .dumps (
    payload ,
    ensure_ascii =False ,
    indent =2 ,
    sort_keys =True ,
    )+"\n"
    )
    os .replace (temporary ,path )


def verify_manifest (root :Path ,manifest_path :Path )->None :
    manifest =json .loads (manifest_path .read_text ())
    for relative ,record in manifest ["files"].items ():
        path =root /relative 
        verify (path ,record ["sha256"])
        assert path .stat ().st_size ==int (record ["bytes"])


def preflight ()->None :
    print ("=== METHOD-GATE CORRUPTION PREFLIGHT ===")

    for path ,expected in EXPECTED .items ():
        verify (path ,expected )

    for _ ,(path ,expected )in MODELS .items ():
        verify (path ,expected )

    verify_manifest (
    METHOD_ROOT /"evaluation_v2_clean",
    CLEAN_MANIFEST ,
    )
    verify_manifest (
    METHOD_ROOT /"evaluation_v3_failures",
    FAILURE_MANIFEST ,
    )

    protocol =json .loads (PROTOCOL .read_text ())
    gate =protocol ["evaluation"]["corruption_gate"]

    assert tuple (gate ["families"])==FAMILIES 
    assert int (gate ["severity"])==SEVERITY 
    assert gate ["split"]=="frozen_train_dev_5000"
    assert gate ["metrics"]==["AP","AP_small"]

    gt =json .loads (GT .read_text ())
    assert len (gt ["images"])==5000 
    assert len (gt ["annotations"])==92392 
    assert {
    int (row ["id"])
    for row in gt ["categories"]
    }==set (range (1 ,11 ))

    failure =json .loads (FAILURE_RESULT .read_text ())
    assert failure ["status"]=="failure_evaluation_complete"

    if OUTPUT .exists ():
        raise FileExistsError (
        f"Refusing to overwrite: {OUTPUT }"
        )

    incomplete =list (
    OUTPUT .parent .glob (f"{OUTPUT .name }.incomplete-*")
    )
    if incomplete :
        raise FileExistsError (incomplete )

    free_gib =(
    shutil .disk_usage ("/root/rivermind-data").free 
    /1024 **3 
    )
    print ("free_GiB:",round (free_gib ,3 ))
    assert free_gib >=8.0 

    assert torch .cuda .is_available ()
    print ("GPU:",torch .cuda .get_device_name (0 ))
    print ("PASS: frozen corruption inputs and rules are valid")
    print ("NOTHING INFERRED OR CREATED")


def run_export (
model_name :str ,
checkpoint :Path ,
corruption :str ,
output_dir :Path ,
)->None :
    exporter .EXPECTED_VAL_IMAGES =5000 
    exporter .EXPECTED_VAL_ANNOTATIONS =92392 

    previous_argv =sys .argv [:]
    try :
        sys .argv =[
        str (EXPORTER ),
        "--model",str (checkpoint ),
        "--dataset-root",str (DATASET_ROOT ),
        "--annotation",str (GT ),
        "--output-dir",str (output_dir ),
        "--corruption",corruption ,
        "--severity",str (SEVERITY ),
        "--device","0",
        "--imgsz","960",
        "--batch","4",
        "--conf","0.001",
        "--iou","0.7",
        "--max-det","300",
        "--limit","0",
        "--save-vis","0",
        ]
        exporter .main ()
    finally :
        sys .argv =previous_argv 

    metadata_path =output_dir /"metadata.json"
    metadata =json .loads (metadata_path .read_text ())

    assert metadata ["num_gt_images"]==5000 
    assert metadata ["num_gt_annotations"]==92392 
    assert metadata ["num_images_exported"]==5000 

    metadata ["split"]="frozen_train_dev_5000"
    metadata ["method_gate_arm"]=model_name 
    metadata ["method_gate_protocol"]=str (PROTOCOL )
    metadata ["method_gate_protocol_sha256"]=EXPECTED [PROTOCOL ]
    metadata ["metadata_split_corrected_by_controller"]=True 

    write_json (metadata_path ,metadata )


def evaluate_run (
coco_gt :COCO ,
model_name :str ,
corruption :str ,
run_dir :Path ,
)->dict :
    prediction_path =run_dir /"predictions.json"
    metadata_path =run_dir /"metadata.json"

    metadata =json .loads (metadata_path .read_text ())

    started =time .time ()
    coco_dt =coco_gt .loadRes (str (prediction_path ))

    evaluator =COCOeval (coco_gt ,coco_dt ,"bbox")
    evaluator .params .imgIds =sorted (coco_gt .getImgIds ())
    evaluator .evaluate ()
    evaluator .accumulate ()
    evaluator .summarize ()

    stats =[float (value )for value in evaluator .stats .tolist ()]

    result ={
    "model":model_name ,
    "corruption":corruption ,
    "severity":SEVERITY ,
    "images":5000 ,
    "gt_objects":92392 ,
    "predictions":int (metadata ["num_predictions"]),
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
    "prediction_sha256":sha256 (prediction_path ),
    "metadata_sha256":sha256 (metadata_path ),
    "evaluation_minutes":(time .time ()-started )/60.0 ,
    }

    write_json (run_dir /"metrics.json",result )

    del evaluator 
    del coco_dt 
    gc .collect ()

    if torch .cuda .is_available ():
        torch .cuda .empty_cache ()

    return result 


def execute ()->None :
    preflight ()

    staging =OUTPUT .with_name (
    f"{OUTPUT .name }.incomplete-{os .getpid ()}"
    )
    staging .mkdir (parents =False ,exist_ok =False )

    coco_gt =COCO (str (GT ))
    records =[]

    for model_name in ("uniform","scalar_risk","typed_cafr"):
        checkpoint ,_ =MODELS [model_name ]

        for corruption in FAMILIES :
            print ("\n"+"="*100 )
            print (
            f"CORRUPTION RUN: {model_name } | "
            f"{corruption }_s{SEVERITY }"
            )
            print ("="*100 )

            run_dir =(
            staging /"runs"
            /model_name 
            /f"{corruption }_s{SEVERITY }"
            )

            run_export (
            model_name ,
            checkpoint ,
            corruption ,
            run_dir ,
            )
            records .append (
            evaluate_run (
            coco_gt ,
            model_name ,
            corruption ,
            run_dir ,
            )
            )

    summary ={}
    for model_name in MODELS :
        rows =[
        row 
        for row in records 
        if row ["model"]==model_name 
        ]
        assert len (rows )==3 

        summary [model_name ]={
        "mean_corruption_AP":float (np .mean ([
        row ["AP"]for row in rows 
        ])),
        "mean_corruption_AP_small":float (np .mean ([
        row ["AP_small"]for row in rows 
        ])),
        "families":{
        row ["corruption"]:{
        "AP":row ["AP"],
        "AP_small":row ["AP_small"],
        }
        for row in rows 
        },
        }

    comparisons ={}
    for left ,right in (
    ("scalar_risk","uniform"),
    ("typed_cafr","uniform"),
    ("typed_cafr","scalar_risk"),
    ):
        comparisons [f"{left }_minus_{right }"]={
        "delta_mean_corruption_AP":(
        summary [left ]["mean_corruption_AP"]
        -summary [right ]["mean_corruption_AP"]
        ),
        "delta_mean_corruption_AP_small":(
        summary [left ]["mean_corruption_AP_small"]
        -summary [right ]["mean_corruption_AP_small"]
        ),
        }

    failure =json .loads (FAILURE_RESULT .read_text ())
    previous_checks =failure ["rule_checks"]

    corruption_delta =comparisons [
    "typed_cafr_minus_uniform"
    ]["delta_mean_corruption_AP_small"]

    corruption_check =corruption_delta >=-0.002 

    all_previous =all (
    bool (value )
    for family in previous_checks .values ()
    for value in family .values ()
    )
    advance =all_previous and corruption_check 

    decision ={
    "version":"method_gate_final_decision_v1",
    "status":"complete",
    "method":"typed_cafr_v1",
    "previous_rule_checks":previous_checks ,
    "corruption_guardrail":{
    "delta_mean_corruption_AP_small":corruption_delta ,
    "minimum":-0.002 ,
    "passed":corruption_check ,
    },
    "all_rule_families_required":True ,
    "advance_typed_cafr_v1":advance ,
    "gate_result":(
    "PASS"if advance else "FAIL"
    ),
    "if_failed":(
    "Stop before formal training; diagnose and design CAFR V2 "
    "without changing frozen V1 thresholds."
    ),
    }

    write_json (staging /"corruption_metrics.json",{
    "version":"method_gate_corruption_metrics_v1",
    "records":records ,
    "summary":summary ,
    "comparisons":comparisons ,
    })
    write_json (staging /"final_gate_decision.json",decision )

    files ={}
    for path in sorted (staging .rglob ("*")):
        if not path .is_file ():
            continue 

        relative =str (path .relative_to (staging ))
        files [relative ]={
        "bytes":path .stat ().st_size ,
        "sha256":sha256 (path ),
        }

    write_json (staging /"artifact_manifest.json",{
    "version":"method_gate_corruption_artifacts_v1",
    "status":"complete",
    "files":files ,
    })

    os .replace (staging ,OUTPUT )

    print ("\n"+"="*100 )
    print ("METHOD-GATE CORRUPTION EVALUATION COMPLETE")
    print ("="*100 )

    for model_name in MODELS :
        print (
        model_name ,
        "mean_AP=",
        round (summary [model_name ]["mean_corruption_AP"],6 ),
        "mean_AP_small=",
        round (summary [model_name ]["mean_corruption_AP_small"],6 ),
        )

    print (json .dumps (
    comparisons ,
    ensure_ascii =False ,
    indent =2 ,
    sort_keys =True ,
    ))
    print (json .dumps (
    decision ,
    ensure_ascii =False ,
    indent =2 ,
    sort_keys =True ,
    ))
    print ("output:",OUTPUT )
    print (
    "manifest_sha256:",
    sha256 (OUTPUT /"artifact_manifest.json"),
    )


def main ()->None :
    parser =argparse .ArgumentParser ()
    parser .add_argument ("--execute",action ="store_true")
    args =parser .parse_args ()

    if args .execute :
        execute ()
    else :
        preflight ()


if __name__ =="__main__":
    main ()
