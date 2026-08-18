from __future__ import annotations 

import argparse 
import csv 
import gc 
import hashlib 
import json 
import math 
import os 
import shutil 
import time 
from datetime import datetime ,timezone 
from pathlib import Path 

import torch 
import ultralytics 
from ultralytics import YOLO 
from ultralytics .data .utils import img2label_paths 

ROOT =Path (
"/root/rivermind-data/autodrive/results/evaluation"
)
PROTOCOL =(
ROOT 
/"method_gate/frozen_manifests/"
"method_gate_protocol_v1.json"
)
MATERIALIZATION_V1_MANIFEST =(
ROOT 
/"method_gate/materialization_v1/"
"artifact_manifest.json"
)
CACHE_V2 =(
ROOT 
/"method_gate/materialization_v2_cache_isolated"
)
CACHE_V2_MANIFEST =(
CACHE_V2 /"artifact_manifest.json"
)
LOADER_SMOKE_MANIFEST =(
ROOT 
/"method_gate/loader_smoke_v1/"
"artifact_manifest.json"
)
CHECKPOINT =(
ROOT 
/"architecture_gate/formal_gate_v4/"
"p2_projected/weights/resume_epoch20.pt"
)
SHM_ROOT =Path (
"/dev/shm/rebu_yolo/method_gate_v1"
)

OUTPUT_ROOT =(
ROOT /"method_gate/training_v1"
)
STATE_PATH =(
OUTPUT_ROOT /"controller_state.json"
)

EXPECTED ={
PROTOCOL :(
"6be7ae25550360e38195bb785a064bd19cb9c4a65b4625b30b5698861cc1b00a"
),
MATERIALIZATION_V1_MANIFEST :(
"74f26c5cc14a2f5bea497bea921966d5c12cbf81a43ac19c39fe71d226e18c06"
),
CACHE_V2_MANIFEST :(
"434df46b40c443505d68d4f3b55929ffc488afa4c8ab76ac3d8618d7bf2eb21d"
),
LOADER_SMOKE_MANIFEST :(
"9ccb56e2a04182349e3ebffbe356549a3cbf81201d767a000ca7826707418410"
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


def sha256_file (path :Path )->str :
    digest =hashlib .sha256 ()
    with path .open ("rb")as handle :
        while True :
            block =handle .read (8 *1024 *1024 )
            if not block :
                break 
            digest .update (block )
    return digest .hexdigest ()


def atomic_json (path :Path ,value :dict )->None :
    temporary =Path (
    str (path )+f".incomplete-{os .getpid ()}"
    )
    temporary .write_text (
    json .dumps (
    value ,
    indent =2 ,
    ensure_ascii =False ,
    )+"\n",
    encoding ="utf-8",
    )
    os .replace (temporary ,path )


def read_paths (path :Path )->list [str ]:
    values =[
    line .strip ()
    for line in path .read_text ().splitlines ()
    if line .strip ()
    ]
    if len (values )!=len (set (values )):
        raise AssertionError (path )
    return values 


def predicted_cache (first_image :str )->Path :
    label =img2label_paths (
    [first_image ]
    )[0 ]
    return (
    Path (label )
    .parent 
    .with_suffix (".cache")
    )


def verify_manifest (
directory :Path ,
manifest_path :Path ,
)->None :
    manifest =json .loads (
    manifest_path .read_text ()
    )

    for name ,record in (
    manifest ["files"].items ()
    ):
        path =directory /name 
        if not path .is_file ():
            raise FileNotFoundError (path )

        actual =sha256_file (path )
        if actual !=record ["sha256"]:
            raise AssertionError (path )


def read_results (path :Path )->list [dict ]:
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

    for row in rows :
        for key ,value in row .items ():
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

    return rows 


def exact_ema_loaded (
checkpoint :dict ,
loaded_model ,
)->bool :
    ema =checkpoint .get ("ema")
    if ema is None :
        return False 

    first_state =ema .state_dict ()
    second_state =loaded_model .state_dict ()

    if set (first_state )!=set (second_state ):
        return False 

    for key in first_state :
        first =(
        first_state [key ]
        .detach ()
        .cpu ()
        .float ()
        )
        second =(
        second_state [key ]
        .detach ()
        .cpu ()
        .float ()
        )
        if not torch .equal (first ,second ):
            return False 

    return True 


def main ()->None :
    parser =argparse .ArgumentParser ()
    parser .add_argument (
    "--execute",
    action ="store_true",
    )
    args =parser .parse_args ()

    print ("=== METHOD-GATE TRAINING PREFLIGHT ===")

    for path ,expected in EXPECTED .items ():
        if not path .is_file ():
            raise FileNotFoundError (path )

        actual =sha256_file (path )
        print (path )
        print (" expected:",expected )
        print (" actual  :",actual )

        if actual !=expected :
            raise AssertionError (path )

    verify_manifest (
    CACHE_V2 ,
    CACHE_V2_MANIFEST ,
    )

    protocol =json .loads (
    PROTOCOL .read_text ()
    )
    training =protocol ["training"]

    assert protocol ["status"]==(
    "frozen_before_materialization_and_training"
    )
    assert protocol [
    "fixed_execution_order"
    ]==ARMS 
    assert training [
    "gate_finetune_epochs"
    ]==10 
    assert training ["optimizer"]=="SGD"
    assert training [
    "resume_old_optimizer"
    ]is False 
    assert training ["mosaic"]==0.0 

    train_lists ={}
    cache_routes ={}

    for arm in ARMS :
        yaml_path =(
        CACHE_V2 /f"{arm }_data.yaml"
        )
        train_list =(
        CACHE_V2 /f"{arm }_train.txt"
        )

        if not yaml_path .is_file ():
            raise FileNotFoundError (yaml_path )

        paths =read_paths (train_list )
        if len (paths )!=65_000 :
            raise AssertionError (
            (arm ,len (paths ))
            )

        train_lists [arm ]=train_list 
        cache_routes [arm ]=predicted_cache (
        paths [0 ]
        )

    dev_paths =read_paths (
    CACHE_V2 /"dev_cache_safe.txt"
    )
    if len (dev_paths )!=5_000 :
        raise AssertionError (len (dev_paths ))

    cache_routes ["dev"]=predicted_cache (
    dev_paths [0 ]
    )

    if len (set (cache_routes .values ()))!=4 :
        raise AssertionError (cache_routes )

    for name ,route in cache_routes .items ():
        print (
        f"cache_route_{name }:",
        route ,
        )
        if not str (route ).startswith (
        "/dev/shm/"
        ):
            raise AssertionError (route )

    for arm in (
    "scalar_risk",
    "typed_cafr",
    ):
        images =list (
        (SHM_ROOT /arm /"images")
        .glob ("*.jpg")
        )
        labels =list (
        (SHM_ROOT /arm /"labels")
        .glob ("*.txt")
        )

        if len (images )!=19_500 :
            raise AssertionError (
            (arm ,len (images ))
            )
        if len (labels )!=19_500 :
            raise AssertionError (
            (arm ,len (labels ))
            )

    checkpoint_data =torch .load (
    CHECKPOINT ,
    map_location ="cpu",
    )
    assert checkpoint_data .get ("ema")is not None 
    assert checkpoint_data .get ("model")is None 
    assert int (checkpoint_data ["epoch"])==19 

    loaded =YOLO (
    str (CHECKPOINT ),
    task ="detect",
    )
    assert exact_ema_loaded (
    checkpoint_data ,
    loaded .model ,
    )
    assert loaded .model .stride .tolist ()==[
    4.0 ,8.0 ,16.0 ,32.0 
    ]

    del loaded 
    del checkpoint_data 
    gc .collect ()

    if not torch .cuda .is_available ():
        raise RuntimeError ("CUDA unavailable")

    free_gpu ,total_gpu =(
    torch .cuda .mem_get_info (0 )
    )
    disk =shutil .disk_usage (
    "/root/rivermind-data"
    )
    shm =shutil .disk_usage ("/dev/shm")

    print ()
    print ("GPU:",torch .cuda .get_device_name (0 ))
    print (
    "GPU_free_GiB:",
    free_gpu /(1024 **3 ),
    )
    print (
    "GPU_total_GiB:",
    total_gpu /(1024 **3 ),
    )
    print (
    "data_disk_free_GiB:",
    disk .free /(1024 **3 ),
    )
    print (
    "shm_free_GiB:",
    shm .free /(1024 **3 ),
    )
    print ("torch:",torch .__version__ )
    print ("ultralytics:",ultralytics .__version__ )

    if free_gpu <19 *1024 **3 :
        raise RuntimeError (
        "Insufficient free GPU memory"
        )
    if disk .free <8 *1024 **3 :
        raise RuntimeError (
        "Insufficient persistent storage"
        )
    if shm .free <16 *1024 **3 :
        raise RuntimeError (
        "Insufficient shared-memory reserve"
        )

    if OUTPUT_ROOT .exists ():
        raise FileExistsError (
        f"Refusing to overwrite: {OUTPUT_ROOT }"
        )

    if not args .execute :
        print ()
        print ("PASS: formal method-gate training preflight")
        print ("NOTHING TRAINED OR CREATED")
        return 

    OUTPUT_ROOT .mkdir (parents =True )

    state ={
    "version":"method_gate_training_v1",
    "status":"running",
    "created_utc":datetime .now (
    timezone .utc 
    ).isoformat (),
    "protocol_sha256":EXPECTED [
    PROTOCOL 
    ],
    "start_checkpoint_sha256":(
    EXPECTED [CHECKPOINT ]
    ),
    "execution_order":ARMS ,
    "current_arm":None ,
    "completed_arms":[],
    "arm_states":{
    arm :{
    "status":"pending",
    "completed_epochs":0 ,
    }
    for arm in ARMS 
    },
    }
    atomic_json (STATE_PATH ,state )

    checkpoint_before =sha256_file (
    CHECKPOINT 
    )
    arm_summaries ={}

    for arm in ARMS :
        print ()
        print ("="*100 )
        print (
        f"TRAINING METHOD-GATE ARM: {arm }"
        )
        print ("="*100 )

        state ["current_arm"]=arm 
        state ["arm_states"][arm ]={
        "status":"running",
        "completed_epochs":0 ,
        "started_utc":datetime .now (
        timezone .utc 
        ).isoformat (),
        }
        atomic_json (STATE_PATH ,state )

        gc .collect ()
        torch .cuda .empty_cache ()
        torch .cuda .reset_peak_memory_stats (0 )

        model =YOLO (
        str (CHECKPOINT ),
        task ="detect",
        )


        checkpoint_data =torch .load (
        CHECKPOINT ,
        map_location ="cpu",
        )
        if not exact_ema_loaded (
        checkpoint_data ,
        model .model ,
        ):
            raise AssertionError (
            f"EMA initialization failed: {arm }"
            )
        del checkpoint_data 
        gc .collect ()

        def on_train_epoch_end (trainer ):
            completed =int (
            trainer .epoch 
            )+1 
            state ["arm_states"][arm ][
            "completed_epochs"
            ]=completed 
            state ["arm_states"][arm ][
            "last_update_utc"
            ]=datetime .now (
            timezone .utc 
            ).isoformat ()
            atomic_json (STATE_PATH ,state )

        model .add_callback (
        "on_train_epoch_end",
        on_train_epoch_end ,
        )

        started =time .time ()

        model .train (
        data =str (
        CACHE_V2 /f"{arm }_data.yaml"
        ),
        epochs =10 ,
        imgsz =int (training ["image_size"]),
        batch =int (
        training ["physical_batch"]
        ),
        nbs =int (
        training ["nominal_batch"]
        ),
        device =0 ,
        workers =int (training ["workers"]),
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
        val =False ,
        save =True ,
        save_period =5 ,
        plots =False ,
        profile =False ,
        patience =0 ,
        project =str (OUTPUT_ROOT ),
        name =arm ,
        exist_ok =False ,
        verbose =True ,
        )

        elapsed_minutes =(
        time .time ()-started 
        )/60.0 

        trainer =model .trainer 
        save_dir =Path (trainer .save_dir )
        results_path =(
        save_dir /"results.csv"
        )
        rows =read_results (results_path )

        if len (rows )!=10 :
            raise AssertionError (
            (arm ,len (rows ))
            )

        last_checkpoint =(
        save_dir /"weights/last.pt"
        )
        if not last_checkpoint .is_file ():
            raise FileNotFoundError (
            last_checkpoint 
            )

        peak_memory =(
        torch .cuda .max_memory_allocated (0 )
        /(1024 **3 )
        )

        summary ={
        "arm":arm ,
        "status":"complete",
        "epochs":len (rows ),
        "elapsed_minutes":elapsed_minutes ,
        "save_dir":str (save_dir ),
        "last_checkpoint":str (
        last_checkpoint 
        ),
        "last_checkpoint_sha256":(
        sha256_file (last_checkpoint )
        ),
        "results_csv":str (
        results_path 
        ),
        "results_csv_sha256":(
        sha256_file (results_path )
        ),
        "last_results_row":rows [-1 ],
        "peak_allocated_GiB":(
        peak_memory 
        ),
        "optimizer":type (
        trainer .optimizer 
        ).__name__ ,
        "resume":bool (
        trainer .args .resume 
        ),
        }

        if summary ["optimizer"]!="SGD":
            raise AssertionError (summary )
        if summary ["resume"]:
            raise AssertionError (summary )

        arm_summaries [arm ]=summary 

        state ["arm_states"][arm ]={
        **summary ,
        "completed_utc":datetime .now (
        timezone .utc 
        ).isoformat (),
        }
        state ["completed_arms"].append (arm )
        state ["current_arm"]=None 
        atomic_json (STATE_PATH ,state )

        print ()
        print ("ARM COMPLETE:",arm )
        print (
        "elapsed_minutes:",
        elapsed_minutes ,
        )
        print (
        "last_checkpoint:",
        last_checkpoint ,
        )
        print (
        "last_checkpoint_sha256:",
        summary [
        "last_checkpoint_sha256"
        ],
        )
        print (
        "peak_allocated_GiB:",
        peak_memory ,
        )

        del trainer 
        del model 
        gc .collect ()
        torch .cuda .empty_cache ()

        if sha256_file (CHECKPOINT )!=(
        checkpoint_before 
        ):
            raise AssertionError (
            "Shared initialization changed"
            )

    training_summary ={
    "version":"method_gate_training_v1",
    "status":"training_complete_pending_evaluation",
    "created_utc":state ["created_utc"],
    "completed_utc":datetime .now (
    timezone .utc 
    ).isoformat (),
    "protocol":str (PROTOCOL ),
    "protocol_sha256":EXPECTED [
    PROTOCOL 
    ],
    "initialization_checkpoint":str (
    CHECKPOINT 
    ),
    "initialization_sha256":(
    checkpoint_before 
    ),
    "arm_order":ARMS ,
    "arms":arm_summaries ,
    "formal_final_models":False ,
    "used_for_formal_initialization":False ,
    "evaluation_complete":False ,
    }

    summary_path =(
    OUTPUT_ROOT /"training_summary.json"
    )
    atomic_json (
    summary_path ,
    training_summary ,
    )

    artifact_files ={
    "controller_state.json":{
    "bytes":STATE_PATH .stat ().st_size ,
    "sha256":sha256_file (
    STATE_PATH 
    ),
    },
    "training_summary.json":{
    "bytes":summary_path .stat ().st_size ,
    "sha256":sha256_file (
    summary_path 
    ),
    },
    }

    for arm ,summary in arm_summaries .items ():
        for key in (
        "last_checkpoint",
        "results_csv",
        ):
            path =Path (summary [key ])
            relative =str (
            path .relative_to (OUTPUT_ROOT )
            )
            artifact_files [relative ]={
            "bytes":path .stat ().st_size ,
            "sha256":sha256_file (path ),
            }

    artifact_manifest ={
    "version":"method_gate_training_v1",
    "status":(
    "training_complete_pending_evaluation"
    ),
    "files":artifact_files ,
    }

    artifact_path =(
    OUTPUT_ROOT 
    /"training_artifact_manifest.json"
    )
    atomic_json (
    artifact_path ,
    artifact_manifest ,
    )

    state ["status"]=(
    "training_complete_pending_evaluation"
    )
    state ["completed_utc"]=datetime .now (
    timezone .utc 
    ).isoformat ()
    atomic_json (STATE_PATH ,state )

    print ()
    print ("="*100 )
    print ("METHOD-GATE TRAINING COMPLETE")
    print ("="*100 )

    for arm in ARMS :
        summary =arm_summaries [arm ]
        print (
        arm ,
        "minutes=",
        round (
        summary ["elapsed_minutes"],
        2 ,
        ),
        "peak_GiB=",
        round (
        summary [
        "peak_allocated_GiB"
        ],
        3 ,
        ),
        "checkpoint_sha256=",
        summary [
        "last_checkpoint_sha256"
        ],
        )

    print (
    "training_manifest:",
    artifact_path ,
    )
    print (
    "training_manifest_sha256:",
    sha256_file (artifact_path ),
    )
    print (
    "PASS: all three 10-epoch arms completed "
    "from the same EMA initialization"
    )
    print (
    "NEXT: frozen clean/failure/corruption evaluation"
    )


if __name__ =="__main__":
    main ()
