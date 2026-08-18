from __future__ import annotations 

import argparse 
import csv 
import gc 
import hashlib 
import json 
import math 
import os 
from datetime import datetime ,timezone 
from pathlib import Path 

import numpy as np 
import torch 
import ultralytics 
from ultralytics import YOLO 
from ultralytics .data .utils import img2label_paths 

ROOT =Path (
"/root/rivermind-data/autodrive/results/evaluation"
)
V2 =(
ROOT 
/"method_gate/materialization_v2_cache_isolated"
)
V2_MANIFEST =V2 /"artifact_manifest.json"
PROTOCOL =(
ROOT 
/"method_gate/frozen_manifests/"
"method_gate_protocol_v1.json"
)
CHECKPOINT =(
ROOT 
/"architecture_gate/formal_gate_v4/"
"p2_projected/weights/resume_epoch20.pt"
)

SHM_ROOT =Path (
"/dev/shm/rebu_yolo/method_gate_v1"
)
SMOKE_FINAL =Path (
"/dev/shm/rebu_yolo/method_gate_loader_smoke_v1"
)
PERSISTENT_FINAL =(
ROOT /"method_gate/loader_smoke_v1"
)

ORIGINAL_CACHE =Path (
"/root/rivermind-data/autodrive/datasets/"
"datasets/bdd100k_final/labels/train.cache"
)

EXPECTED ={
V2_MANIFEST :(
"434df46b40c443505d68d4f3b55929ffc488afa4c8ab76ac3d8618d7bf2eb21d"
),
PROTOCOL :(
"6be7ae25550360e38195bb785a064bd19cb9c4a65b4625b30b5698861cc1b00a"
),
CHECKPOINT :(
"642a6b4009f085649b6da3f1647821d3e760e6724b019e08d022080c8050bc29"
),
}

ARMS =[
"uniform",
"scalar_risk",
"typed_cafr",
]
SMOKE_IMAGES =32 
BATCH =8 

NAMES ={
0 :"pedestrian",
1 :"rider",
2 :"car",
3 :"truck",
4 :"bus",
5 :"train",
6 :"motorcycle",
7 :"bicycle",
8 :"traffic light",
9 :"traffic sign",
}


def sha256_file (path :Path )->str :
    digest =hashlib .sha256 ()
    with path .open ("rb")as handle :
        while True :
            block =handle .read (8 *1024 *1024 )
            if not block :
                break 
            digest .update (block )
    return digest .hexdigest ()


def read_paths (path :Path )->list [str ]:
    values =[
    line .strip ()
    for line in path .read_text ().splitlines ()
    if line .strip ()
    ]
    if len (values )!=len (set (values )):
        raise AssertionError (path )
    return values 


def predicted_cache (image_path :str )->Path :
    label_path =img2label_paths (
    [image_path ]
    )[0 ]
    return (
    Path (label_path )
    .parent 
    .with_suffix (".cache")
    )


def yaml_text (
train_list :Path ,
val_list :Path ,
)->str :
    lines =[
    "path: /root/rivermind-data/autodrive/"
    "datasets/datasets/bdd100k_final",
    f"train: {train_list }",
    f"val: {val_list }",
    "",
    "names:",
    ]

    for class_id ,name in NAMES .items ():
        lines .append (f"  {class_id }: {name }")

    return "\n".join (lines )+"\n"


def read_last_results_row (path :Path )->dict :
    if not path .is_file ():
        raise FileNotFoundError (path )

    with path .open (
    "r",
    encoding ="utf-8",
    newline ="",
    )as handle :
        rows =list (csv .DictReader (handle ))

    if not rows :
        raise AssertionError (path )

    result =rows [-1 ]

    for key ,value in result .items ():
        if value is None or value =="":
            continue 
        try :
            number =float (value )
        except ValueError :
            continue 
        if not math .isfinite (number ):
            raise AssertionError (
            (path ,key ,value )
            )

    return result 


def inspect_cache (path :Path )->dict :
    if not path .is_file ():
        raise FileNotFoundError (path )

    data =np .load (
    path ,
    allow_pickle =True ,
    ).item ()

    labels =data .get ("labels",[])
    return {
    "path":str (path ),
    "bytes":path .stat ().st_size ,
    "sha256":sha256_file (path ),
    "keys":sorted (data .keys ()),
    "label_records":len (labels ),
    "results":data .get ("results"),
    "version":data .get ("version"),
    }


def main ()->None :
    parser =argparse .ArgumentParser ()
    parser .add_argument (
    "--execute",
    action ="store_true",
    )
    args =parser .parse_args ()

    print ("=== METHOD-GATE LOADER-SMOKE PREFLIGHT ===")

    for path ,expected in EXPECTED .items ():
        if not path .is_file ():
            raise FileNotFoundError (path )

        actual =sha256_file (path )
        print (path )
        print (" expected:",expected )
        print (" actual  :",actual )

        if actual !=expected :
            raise AssertionError (path )

    v2_manifest =json .loads (
    V2_MANIFEST .read_text ()
    )
    for name ,record in (
    v2_manifest ["files"].items ()
    ):
        path =V2 /name 
        if not path .is_file ():
            raise FileNotFoundError (path )
        if sha256_file (path )!=record ["sha256"]:
            raise AssertionError (path )

    protocol =json .loads (
    PROTOCOL .read_text ()
    )
    training =protocol ["training"]

    train_lists ={
    arm :read_paths (
    V2 /f"{arm }_train.txt"
    )
    for arm in ARMS 
    }
    dev_paths =read_paths (
    V2 /"dev_cache_safe.txt"
    )

    for arm ,paths in train_lists .items ():
        if len (paths )!=65_000 :
            raise AssertionError (
            (arm ,len (paths ))
            )

    if len (dev_paths )!=5_000 :
        raise AssertionError (len (dev_paths ))

    cache_routes ={
    arm :predicted_cache (
    train_lists [arm ][0 ]
    )
    for arm in ARMS 
    }
    cache_routes ["dev"]=predicted_cache (
    dev_paths [0 ]
    )

    if len (set (cache_routes .values ()))!=4 :
        raise AssertionError (cache_routes )

    print ()
    print ("cache_routes:")
    for name ,path in cache_routes .items ():
        print (f"{name }: {path }")
        if not str (path ).startswith (
        "/dev/shm/"
        ):
            raise AssertionError (path )
        if path .exists ():
            raise FileExistsError (
            f"Smoke cache already exists: {path }"
            )

    if not torch .cuda .is_available ():
        raise RuntimeError ("CUDA unavailable")

    print ()
    print ("GPU:",torch .cuda .get_device_name (0 ))
    print (
    "GPU_memory_GiB:",
    torch .cuda .get_device_properties (
    0 
    ).total_memory /(1024 **3 ),
    )
    print ("torch:",torch .__version__ )
    print ("ultralytics:",ultralytics .__version__ )

    if SMOKE_FINAL .exists ():
        raise FileExistsError (SMOKE_FINAL )
    if PERSISTENT_FINAL .exists ():
        raise FileExistsError (
        PERSISTENT_FINAL 
        )

    if not args .execute :
        print ()
        print ("PASS: loader-smoke preflight only")
        print ("NOTHING TRAINED OR CREATED")
        return 

    original_cache_before =(
    sha256_file (ORIGINAL_CACHE )
    if ORIGINAL_CACHE .is_file ()
    else None 
    )
    checkpoint_before =sha256_file (
    CHECKPOINT 
    )

    token =str (os .getpid ())
    smoke_incomplete =Path (
    str (SMOKE_FINAL )
    +f".incomplete-{token }"
    )
    persistent_incomplete =Path (
    str (PERSISTENT_FINAL )
    +f".incomplete-{token }"
    )

    smoke_incomplete .mkdir (parents =True )
    persistent_incomplete .mkdir (parents =True )

    inputs_dir =smoke_incomplete /"inputs"
    runs_dir =smoke_incomplete /"runs"
    inputs_dir .mkdir ()
    runs_dir .mkdir ()

    smoke_dev =dev_paths [:SMOKE_IMAGES ]
    smoke_dev_path =(
    inputs_dir /"dev_smoke.txt"
    )
    smoke_dev_path .write_text (
    "\n".join (smoke_dev )+"\n",
    encoding ="utf-8",
    )

    smoke_yamls ={}

    for arm in ARMS :
        smoke_train =train_lists [arm ][
        :SMOKE_IMAGES 
        ]

        if len (smoke_train )!=SMOKE_IMAGES :
            raise AssertionError (arm )

        train_path =(
        inputs_dir 
        /f"{arm }_train_smoke.txt"
        )
        train_path .write_text (
        "\n".join (smoke_train )+"\n",
        encoding ="utf-8",
        )

        yaml_path =(
        inputs_dir /f"{arm }_smoke.yaml"
        )
        yaml_path .write_text (
        yaml_text (
        train_path ,
        smoke_dev_path ,
        ),
        encoding ="utf-8",
        )
        smoke_yamls [arm ]=yaml_path 

    results ={}

    for arm in ARMS :
        print ()
        print ("="*88 )
        print (f"LOADER SMOKE: {arm }")
        print ("="*88 )

        gc .collect ()
        torch .cuda .empty_cache ()
        torch .cuda .reset_peak_memory_stats (0 )

        model =YOLO (
        str (CHECKPOINT ),
        task ="detect",
        )

        model .train (
        data =str (smoke_yamls [arm ]),
        epochs =1 ,
        imgsz =int (
        training ["image_size"]
        ),
        batch =int (
        training ["physical_batch"]
        ),
        nbs =int (
        training ["nominal_batch"]
        ),
        device =0 ,
        workers =int (
        training ["workers"]
        ),
        cache =False ,
        optimizer ="SGD",
        lr0 =float (training ["lr0"]),
        lrf =float (training ["lrf"]),
        cos_lr =True ,
        momentum =float (
        training ["momentum"]
        ),
        weight_decay =float (
        training ["weight_decay"]
        ),
        warmup_epochs =0.0 ,
        warmup_momentum =float (
        training ["warmup_momentum"]
        ),
        warmup_bias_lr =0.0 ,
        box =float (training ["box"]),
        cls =float (training ["cls"]),
        dfl =float (training ["dfl"]),
        mosaic =0.0 ,
        close_mosaic =0 ,
        mixup =0.0 ,
        cutmix =0.0 ,
        copy_paste =0.0 ,
        hsv_h =float (training ["hsv_h"]),
        hsv_s =float (training ["hsv_s"]),
        hsv_v =float (training ["hsv_v"]),
        degrees =0.0 ,
        translate =float (
        training ["translate"]
        ),
        scale =float (training ["scale"]),
        shear =0.0 ,
        perspective =0.0 ,
        flipud =0.0 ,
        fliplr =float (
        training ["fliplr"]
        ),
        amp =True ,
        deterministic =True ,
        seed =int (training ["seed"]),
        rect =False ,
        multi_scale =False ,
        fraction =1.0 ,
        pretrained =True ,
        resume =False ,
        val =True ,
        save =False ,
        save_period =-1 ,
        plots =False ,
        profile =False ,
        patience =0 ,
        project =str (runs_dir ),
        name =arm ,
        exist_ok =False ,
        verbose =True ,
        )

        trainer =model .trainer 
        save_dir =Path (trainer .save_dir )
        results_csv =(
        save_dir /"results.csv"
        )
        last_row =read_last_results_row (
        results_csv 
        )

        peak_memory =(
        torch .cuda .max_memory_allocated (0 )
        /(1024 **3 )
        )

        optimizer_name =type (
        trainer .optimizer 
        ).__name__ 

        if optimizer_name !="SGD":
            raise AssertionError (
            optimizer_name 
            )
        if bool (trainer .args .resume ):
            raise AssertionError (
            "Old optimizer was resumed"
            )

        results [arm ]={
        "smoke_train_images":SMOKE_IMAGES ,
        "smoke_val_images":SMOKE_IMAGES ,
        "save_dir":str (save_dir ).replace (
        str (smoke_incomplete ),
        str (SMOKE_FINAL ),
        ),
        "peak_allocated_GiB":peak_memory ,
        "optimizer":optimizer_name ,
        "resume":bool (
        trainer .args .resume 
        ),
        "last_results_row":last_row ,
        }

        print ("optimizer:",optimizer_name )
        print ("resume:",trainer .args .resume )
        print (
        "peak_allocated_GiB:",
        peak_memory ,
        )
        print ("last_results_row:",last_row )

        del trainer 
        del model 
        gc .collect ()
        torch .cuda .empty_cache ()

    print ()
    print ("=== CACHE AUDIT AFTER SMOKE ===")

    cache_records ={}
    for name ,path in cache_routes .items ():
        record =inspect_cache (path )
        cache_records [name ]=record 
        print (
        name ,
        "records=",record ["label_records"],
        "sha256=",record ["sha256"],
        )

        if record ["label_records"]!=(
        SMOKE_IMAGES 
        ):
            raise AssertionError (
            (name ,record ["label_records"])
            )

    checkpoint_after =sha256_file (
    CHECKPOINT 
    )
    if checkpoint_after !=checkpoint_before :
        raise AssertionError (
        "Start checkpoint changed"
        )

    original_cache_after =(
    sha256_file (ORIGINAL_CACHE )
    if ORIGINAL_CACHE .is_file ()
    else None 
    )
    if original_cache_after !=(
    original_cache_before 
    ):
        raise AssertionError (
        "Original dataset cache changed"
        )

    summary ={
    "version":"method_gate_loader_smoke_v1",
    "status":"complete",
    "created_utc":datetime .now (
    timezone .utc 
    ).isoformat (),
    "protocol":str (PROTOCOL ),
    "protocol_sha256":EXPECTED [
    PROTOCOL 
    ],
    "cache_isolation_manifest":str (
    V2_MANIFEST 
    ),
    "cache_isolation_manifest_sha256":(
    EXPECTED [V2_MANIFEST ]
    ),
    "checkpoint":str (CHECKPOINT ),
    "checkpoint_sha256_before":(
    checkpoint_before 
    ),
    "checkpoint_sha256_after":(
    checkpoint_after 
    ),
    "original_cache_sha256_before":(
    original_cache_before 
    ),
    "original_cache_sha256_after":(
    original_cache_after 
    ),
    "cache_records":cache_records ,
    "arms":results ,
    "formal_gate_training_started":False ,
    "technical_smoke_only":True ,
    }

    summary_path =(
    persistent_incomplete 
    /"summary.json"
    )
    summary_path .write_text (
    json .dumps (
    summary ,
    indent =2 ,
    ensure_ascii =False ,
    )+"\n",
    encoding ="utf-8",
    )

    artifact_manifest ={
    "version":(
    "method_gate_loader_smoke_v1"
    ),
    "status":"complete",
    "files":{
    "summary.json":{
    "bytes":(
    summary_path .stat ().st_size 
    ),
    "sha256":sha256_file (
    summary_path 
    ),
    },
    },
    }

    artifact_path =(
    persistent_incomplete 
    /"artifact_manifest.json"
    )
    artifact_path .write_text (
    json .dumps (
    artifact_manifest ,
    indent =2 ,
    ensure_ascii =False ,
    )+"\n",
    encoding ="utf-8",
    )

    smoke_incomplete .rename (SMOKE_FINAL )
    persistent_incomplete .rename (
    PERSISTENT_FINAL 
    )

    print ()
    print ("="*92 )
    print ("METHOD-GATE LOADER SMOKE COMPLETE")
    print ("="*92 )

    for arm in ARMS :
        print (
        arm ,
        "peak_GiB=",
        round (
        results [arm ][
        "peak_allocated_GiB"
        ],
        3 ,
        ),
        "optimizer=",
        results [arm ]["optimizer"],
        "resume=",
        results [arm ]["resume"],
        )

    print (
    "checkpoint_unchanged:",
    checkpoint_before 
    ==checkpoint_after ,
    )
    print (
    "original_cache_unchanged:",
    original_cache_before 
    ==original_cache_after ,
    )
    print (
    "artifact_manifest_sha256:",
    sha256_file (
    PERSISTENT_FINAL 
    /"artifact_manifest.json"
    ),
    )
    print (
    "PASS: all three arms load, train and "
    "validate with isolated caches"
    )
    print ("FORMAL GATE TRAINING HAS NOT STARTED")


if __name__ =="__main__":
    main ()
