from __future__ import annotations 

from pathlib import Path 
import argparse 
import hashlib 
import importlib .util 
import json 
import os 

REPO =Path (
"/root/rivermind-data/autodrive/code/"
"robust-autonomous-driving-perception"
)
RESULTS =Path (
"/root/rivermind-data/autodrive/results/evaluation/method_gate"
)

ENGINE =REPO /"scripts/run_yolo_architecture_gate_v4_fixed.py"
OUTPUT =RESULTS /"evaluation_v2_clean"

PROTOCOL =(
RESULTS /"frozen_manifests/method_gate_protocol_v1.json"
)
TRAINING_MANIFEST =(
RESULTS /"training_v1/training_artifact_manifest.json"
)
GT =Path (
"/root/rivermind-data/autodrive/results/evaluation/"
"architecture_gate/prepared_inputs_v3/"
"instances_train_dev_effective.json"
)
DEV_LIST =Path (
"/root/rivermind-data/autodrive/results/evaluation/"
"architecture_gate/prepared_inputs_v1/train_dev.txt"
)

MODELS ={
"uniform":(
RESULTS /"training_v1/uniform/weights/last.pt",
"45f397e81cf6aa3e8fbb08a8fef79b8e95975f85f5b360fa9f4bf2a4f6757210",
),
"scalar_risk":(
RESULTS /"training_v1/scalar_risk/weights/last.pt",
"3093b249d8052924d8324c826731a77b82ab98327a41ff68ab3a2ad615f74b9d",
),
"typed_cafr":(
RESULTS /"training_v1/typed_cafr/weights/last.pt",
"217b6d453d399c345a7e1be5595783044f2555a65ae95f047d6a7278d105c4ee",
),
}

EXPECTED ={
ENGINE :
"fd48e9575a9ef8606e2c20c882ec2dff6552e10d19016f43da8dbfd78547345b",
PROTOCOL :
"6be7ae25550360e38195bb785a064bd19cb9c4a65b4625b30b5698861cc1b00a",
TRAINING_MANIFEST :
"c40007992f0aa49e8062734201fcd2933d5ffb7c4ca6dca3e8866757c8e74e57",
GT :
"5e13ec651d0582032e5eacc1a46d885279fabe278463bd7c2be58b2429796faf",
DEV_LIST :
"d4e1c82556eb6adb30fa2d29f6d86e838922d70ad701f7403058b49184c0fcf2",
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


def verify (path :Path ,expected :str )->None :
    if not path .is_file ():
        raise FileNotFoundError (path )

    actual =sha256 (path )
    print (path )
    print (" expected:",expected )
    print (" actual  :",actual )

    if actual !=expected :
        raise AssertionError (f"SHA256 mismatch: {path }")


def load_engine ():
    spec =importlib .util .spec_from_file_location (
    "architecture_gate_v4_fixed",
    ENGINE ,
    )
    if spec is None or spec .loader is None :
        raise ImportError (ENGINE )

    module =importlib .util .module_from_spec (spec )
    spec .loader .exec_module (module )
    return module 


def preflight ()->None :
    print ("=== METHOD-GATE CLEAN EVALUATION PREFLIGHT ===")

    for path ,expected in EXPECTED .items ():
        verify (path ,expected )

    for _ ,(path ,expected )in MODELS .items ():
        verify (path ,expected )

    protocol =json .loads (PROTOCOL .read_text ())
    assert protocol ["version"]=="method_gate_protocol_v1"
    assert protocol ["evaluation"]["split"]=="frozen_train_dev_5000"
    assert protocol ["evaluation"]["images"]==5000 
    assert protocol ["evaluation"]["effective_gt_objects"]==92392 
    assert protocol ["evaluation"]["confidence_floor"]==0.001 
    assert protocol ["evaluation"]["nms_iou"]==0.7 
    assert protocol ["evaluation"]["max_detections"]==300 

    gt =json .loads (GT .read_text ())
    assert len (gt ["images"])==5000 
    assert len (gt ["annotations"])==92392 
    assert {int (row ["id"])for row in gt ["categories"]}==set (range (1 ,11 ))

    dev_paths =[
    row .strip ()
    for row in DEV_LIST .read_text ().splitlines ()
    if row .strip ()
    ]
    assert len (dev_paths )==5000 

    gt_names ={
    Path (row ["file_name"]).name 
    for row in gt ["images"]
    }
    dev_names ={
    Path (row ).name 
    for row in dev_paths 
    }
    assert gt_names ==dev_names 

    if OUTPUT .exists ():
        raise FileExistsError (
        f"Refusing to overwrite evaluation output: {OUTPUT }"
        )

    print ("PASS: frozen 5k GT, model hashes and evaluator are valid")
    print ("PASS: output directory does not exist")
    print ("NOTHING EVALUATED OR CREATED")


def execute ()->None :
    preflight ()

    engine =load_engine ()


    engine .FORMAL_ROOT =OUTPUT 
    engine .EFFECTIVE_DEV_COCO =GT 
    engine .PROTOCOL_PATH =PROTOCOL 
    engine .EXPECTED_FROZEN_HASHES ={
    PROTOCOL :EXPECTED [PROTOCOL ],
    GT :EXPECTED [GT ],
    }

    expected_by_path ={
    path .resolve ():expected 
    for path ,expected in MODELS .values ()
    }

    def validate_gate_checkpoint (
    path :Path ,
    expected_human_epoch :int ,
    expected_data_yaml :Path ,
    )->dict :
        del expected_human_epoch ,expected_data_yaml 

        resolved =Path (path ).resolve ()
        if resolved not in expected_by_path :
            raise AssertionError (f"Unexpected checkpoint: {resolved }")

        verify (resolved ,expected_by_path [resolved ])
        return {
        "path":str (resolved ),
        "sha256":expected_by_path [resolved ],
        }

    engine .validate_resume_checkpoint =validate_gate_checkpoint 

    engine_protocol ={
    "data":{
    "train_dev_list":str (DEV_LIST ),
    },
    "evaluation":{
    "image_size":960 ,
    "confidence_threshold":0.001 ,
    "nms_iou_threshold":0.7 ,
    "max_detections_per_image":300 ,
    },
    "training":{
    "device":0 ,
    },
    }

    OUTPUT .mkdir (parents =True ,exist_ok =False )

    metrics ={}
    for model_key in ("uniform","scalar_risk","typed_cafr"):
        checkpoint ,_ =MODELS [model_key ]
        metrics [model_key ]=engine .evaluate_checkpoint (
        engine_protocol ,
        model_key ,
        10 ,
        checkpoint ,
        )

    metric_keys =(
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR1",
    "AR10",
    "AR100",
    )

    comparisons ={}
    for left ,right in (
    ("scalar_risk","uniform"),
    ("typed_cafr","uniform"),
    ("typed_cafr","scalar_risk"),
    ):
        comparisons [f"{left }_minus_{right }"]={
        f"delta_{key }":
        float (metrics [left ][key ])
        -float (metrics [right ][key ])
        for key in metric_keys 
        }

    summary ={
    "version":"method_gate_clean_evaluation_v1",
    "status":"clean_evaluation_complete",
    "split":"frozen_train_dev_5000",
    "images":5000 ,
    "effective_gt_objects":92392 ,
    "protocol":str (PROTOCOL ),
    "protocol_sha256":EXPECTED [PROTOCOL ],
    "metrics":metrics ,
    "comparisons":comparisons ,
    "decision_allowed_from_clean_only":False ,
    }
    engine .atomic_write_json (
    OUTPUT /"clean_comparison.json",
    summary ,
    )

    files ={}
    for path in sorted (OUTPUT .rglob ("*")):
        if not path .is_file ():
            continue 
        relative =str (path .relative_to (OUTPUT ))
        files [relative ]={
        "bytes":path .stat ().st_size ,
        "sha256":sha256 (path ),
        }

    manifest ={
    "version":"method_gate_clean_artifacts_v1",
    "status":"clean_evaluation_complete",
    "files":files ,
    }
    engine .atomic_write_json (
    OUTPUT /"clean_artifact_manifest.json",
    manifest ,
    )

    print ("\n"+"="*100 )
    print ("METHOD-GATE CLEAN EVALUATION COMPLETE")
    print ("="*100 )

    for model_key in ("uniform","scalar_risk","typed_cafr"):
        row =metrics [model_key ]
        print (
        model_key ,
        "AP=",round (row ["AP"],6 ),
        "APs=",round (row ["AP_small"],6 ),
        "AP75=",round (row ["AP75"],6 ),
        "AR100=",round (row ["AR100"],6 ),
        )

    print (json .dumps (
    comparisons ,
    ensure_ascii =False ,
    indent =2 ,
    sort_keys =True ,
    ))
    print ("output:",OUTPUT )
    print (
    "manifest_sha256:",
    sha256 (OUTPUT /"clean_artifact_manifest.json"),
    )
    print ("NEXT: frozen failure-subset and corruption evaluation")


def main ()->None :
    parser =argparse .ArgumentParser ()
    parser .add_argument (
    "--execute",
    action ="store_true",
    )
    args =parser .parse_args ()

    if not args .execute :
        preflight ()
        return 

    execute ()


if __name__ =="__main__":
    main ()
