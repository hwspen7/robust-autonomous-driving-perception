from __future__ import annotations 

from copy import deepcopy 
from datetime import datetime 
from pathlib import Path 
import hashlib 
import json 
import os 
import subprocess 


ARCH_ROOT =Path (
"/root/rivermind-data/autodrive/results/evaluation/"
"architecture_gate"
)
V3 =ARCH_ROOT /"prepared_inputs_v3"
OUTPUT_DIR =ARCH_ROOT /"prepared_inputs_v4"
STAGING_DIR =ARCH_ROOT /(
f"prepared_inputs_v4.incomplete-{os .getpid ()}"
)

V3_MANIFEST =V3 /"artifact_manifest.json"
V3_PROTOCOL =V3 /"formal_gate_protocol_v3.json"
V3_DATA_YAML =V3 /"bdd100k_architecture_gate_v3.yaml"
V3_EFFECTIVE_COCO =(
V3 /"instances_train_dev_effective.json"
)

EXPECTED_V3_HASHES ={
V3_MANIFEST :
"ba151a0e87d211a55ce50a990b2d7f88085f9032f2b7632cf7206b4d0da2a071",
V3_PROTOCOL :
"1f19201fdd1d456b32a64135461fe61d6a7ce945ff5adbcbb4fa879bd7841b8f",
V3_DATA_YAML :
"208a57dacf467c4efab4d6d7d687fce63938d24bc4fc2012d551bb49100e808c",
V3_EFFECTIVE_COCO :
"5e13ec651d0582032e5eacc1a46d885279fabe278463bd7c2be58b2429796faf",
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


def verify_hash (path :Path ,expected :str )->None :
    if not path .is_file ():
        raise FileNotFoundError (path )

    actual =sha256 (path )
    print (path )
    print (" expected:",expected )
    print (" actual  :",actual )

    if actual !=expected :
        raise AssertionError (
        f"SHA256 mismatch:\n{path }\n"
        f"expected={expected }\nactual={actual }"
        )


def write_json (path :Path ,payload :object )->None :
    text =json .dumps (
    payload ,
    ensure_ascii =False ,
    indent =2 ,
    sort_keys =True ,
    )+"\n"

    with path .open ("x",encoding ="utf-8")as handle :
        handle .write (text )
        handle .flush ()
        os .fsync (handle .fileno ())


def active_controller_lines ()->list [str ]:
    result =subprocess .run (
    [
    "pgrep",
    "-af",
    (
    "run_yolo_architecture_gate_"
    "v[34].py.*--execute"
    ),
    ],
    text =True ,
    stdout =subprocess .PIPE ,
    stderr =subprocess .DEVNULL ,
    check =False ,
    )
    return [
    row .strip ()
    for row in result .stdout .splitlines ()
    if row .strip ()
    ]


def inspect_aborted_run ()->dict :
    if (ARCH_ROOT /"formal_gate_v3").exists ():
        raise FileExistsError (
        "formal_gate_v3 still exists and has not "
        "been archived."
        )

    candidates =sorted (
    ARCH_ROOT .glob (
    "formal_gate_v3.aborted-ram-cache-*"
    ),
    key =lambda path :path .stat ().st_mtime ,
    )
    if not candidates :
        raise FileNotFoundError (
        "No archived RAM-cache V3 run was found."
        )

    aborted_dir =candidates [-1 ]
    suffix =aborted_dir .name .removeprefix (
    "formal_gate_v3."
    )
    aborted_log =ARCH_ROOT /(
    f"formal_gate_v3_controller.{suffix }.log"
    )

    if not aborted_log .is_file ():
        raise FileNotFoundError (aborted_log )

    log_text =aborted_log .read_text (
    encoding ="utf-8",
    errors ="replace",
    )
    warning_fragment =(
    "cache='ram' may produce non-deterministic "
    "training results"
    )
    if warning_fragment not in log_text :
        raise AssertionError (
        "The archived log does not contain the "
        "RAM-cache determinism warning."
        )

    resume_checkpoints =list (
    aborted_dir .rglob ("resume_epoch10.pt")
    )
    if resume_checkpoints :
        raise AssertionError (
        "The aborted run already produced an "
        "epoch-10 checkpoint."
        )

    epoch_checkpoints =list (
    aborted_dir .rglob ("epoch*.pt")
    )
    if epoch_checkpoints :
        raise AssertionError (
        "The aborted run already produced epoch "
        f"checkpoints: {epoch_checkpoints [:5 ]}"
        )

    result_rows =0 
    result_files =sorted (
    aborted_dir .rglob ("results.csv")
    )
    for path in result_files :
        lines =[
        row 
        for row in path .read_text (
        encoding ="utf-8",
        errors ="replace",
        ).splitlines ()
        if row .strip ()
        ]
        result_rows +=max (0 ,len (lines )-1 )

    if result_rows !=0 :
        raise AssertionError (
        f"Completed result rows found: {result_rows }"
        )

    state_path =(
    aborted_dir /"controller_state.json"
    )
    metadata_path =(
    aborted_dir /"run_metadata.json"
    )

    state =(
    json .loads (state_path .read_text ())
    if state_path .is_file ()
    else None 
    )

    return {
    "aborted_output":str (aborted_dir ),
    "aborted_log":str (aborted_log ),
    "aborted_log_sha256":
    sha256 (aborted_log ),
    "controller_state":state ,
    "controller_state_sha256":(
    sha256 (state_path )
    if state_path .is_file ()
    else None 
    ),
    "run_metadata_sha256":(
    sha256 (metadata_path )
    if metadata_path .is_file ()
    else None 
    ),
    "completed_training_epochs":0 ,
    "epoch10_checkpoint_created":False ,
    "reason":(
    "Ultralytics emitted a warning that "
    "cache='ram' may produce non-deterministic "
    "training results."
    ),
    }


def main ()->None :
    print ("=== V4 DETERMINISTIC-AMENDMENT PREFLIGHT ===")

    controllers =active_controller_lines ()
    if controllers :
        raise RuntimeError (
        "An architecture-gate controller is still "
        f"running:\n{controllers }"
        )

    if OUTPUT_DIR .exists ():
        raise FileExistsError (
        f"Refusing to overwrite: {OUTPUT_DIR }"
        )

    incomplete =sorted (
    ARCH_ROOT .glob (
    "prepared_inputs_v4.incomplete-*"
    )
    )
    if incomplete :
        raise FileExistsError (
        f"Incomplete V4 directories exist: "
        f"{incomplete }"
        )

    if (ARCH_ROOT /"formal_gate_v4").exists ():
        raise FileExistsError (
        "formal_gate_v4 already exists."
        )

    for path ,expected in EXPECTED_V3_HASHES .items ():
        verify_hash (path ,expected )

    v3_manifest =json .loads (
    V3_MANIFEST .read_text ()
    )
    v3_protocol =json .loads (
    V3_PROTOCOL .read_text ()
    )

    assert v3_manifest ["status"]=="frozen"
    assert v3_protocol ["version"]==(
    "architecture_gate_protocol_v3"
    )
    assert v3_protocol ["status"]==(
    "frozen_before_training"
    )
    assert v3_protocol ["training"]["cache"]=="ram"
    assert (
    v3_protocol ["training"]["deterministic"]
    is True 
    )
    assert (
    v3_protocol ["training"]
    ["planned_schedule_epochs"]
    ==20 
    )
    assert (
    v3_protocol ["training"]
    ["checkpoint_epochs"]
    ==[10 ,20 ]
    )

    aborted =inspect_aborted_run ()

    STAGING_DIR .mkdir (
    parents =False ,
    exist_ok =False ,
    )

    amendment_path =(
    STAGING_DIR /"determinism_amendment.json"
    )
    protocol_path =(
    STAGING_DIR /"formal_gate_protocol_v4.json"
    )
    manifest_path =(
    STAGING_DIR /"artifact_manifest.json"
    )

    amendment ={
    "version":"architecture_gate_amendment_v4",
    "status":"frozen_before_training",
    "created_at":
    datetime .now ().astimezone ().isoformat (),
    "parent_protocol":str (V3_PROTOCOL ),
    "parent_protocol_sha256":
    EXPECTED_V3_HASHES [V3_PROTOCOL ],
    "changed_fields":{
    "training.cache":{
    "from":"ram",
    "to":False ,
    }
    },
    "unchanged":[
    "dataset",
    "65k_train_core",
    "5k_train_dev",
    "effective_dev_coco",
    "architectures",
    "initialization",
    "image_size",
    "physical_batch",
    "nominal_batch",
    "optimizer",
    "learning_rate_schedule",
    "augmentations",
    "seed",
    "10_to_20_state_machine",
    "decision_thresholds",
    "latency_evidence",
    ],
    "rationale":(
    "Disable Ultralytics image RAM caching "
    "so deterministic=True is not contradicted "
    "by the runtime cache warning. Linux page "
    "cache remains available without altering "
    "decoded training samples."
    ),
    "disk_cache_rejected":(
    "Available data-disk capacity is insufficient "
    "for a complete decoded 960px image cache."
    ),
    "aborted_v3_run":aborted ,
    }
    write_json (amendment_path ,amendment )

    v4_protocol =deepcopy (v3_protocol )
    v4_protocol ["version"]=(
    "architecture_gate_protocol_v4"
    )
    v4_protocol ["status"]=(
    "frozen_before_training"
    )
    v4_protocol ["ancestry"][
    "prepared_inputs_v3_manifest_sha256"
    ]=EXPECTED_V3_HASHES [V3_MANIFEST ]
    v4_protocol ["ancestry"][
    "formal_gate_protocol_v3_sha256"
    ]=EXPECTED_V3_HASHES [V3_PROTOCOL ]

    v4_protocol ["amendment"]={
    "record":str (
    OUTPUT_DIR /amendment_path .name 
    ),
    "record_sha256":sha256 (amendment_path ),
    "single_changed_field":
    "training.cache",
    "reason":
    "deterministic_image_loading",
    }

    v4_protocol ["training"]["cache"]=False 
    v4_protocol ["training"][
    "cache_policy"
    ]=(
    "Ultralytics decoded-image caching disabled; "
    "operating-system file page cache permitted."
    )
    v4_protocol ["training"][
    "deterministic"
    ]=True 

    write_json (protocol_path ,v4_protocol )

    manifest ={
    "version":"prepared_inputs_v4",
    "status":"frozen",
    "parent_v3_manifest_sha256":
    EXPECTED_V3_HASHES [V3_MANIFEST ],
    "files":{
    amendment_path .name :{
    "bytes":
    amendment_path .stat ().st_size ,
    "sha256":
    sha256 (amendment_path ),
    },
    protocol_path .name :{
    "bytes":
    protocol_path .stat ().st_size ,
    "sha256":
    sha256 (protocol_path ),
    },
    },
    "referenced_v3_artifacts":{
    str (V3_DATA_YAML ):
    EXPECTED_V3_HASHES [V3_DATA_YAML ],
    str (V3_EFFECTIVE_COCO ):
    EXPECTED_V3_HASHES [
    V3_EFFECTIVE_COCO 
    ],
    },
    }
    write_json (manifest_path ,manifest )

    for path in (
    amendment_path ,
    protocol_path ,
    manifest_path ,
    ):
        path .chmod (0o444 )

    os .replace (STAGING_DIR ,OUTPUT_DIR )

    final_manifest =(
    OUTPUT_DIR /manifest_path .name 
    )
    final_protocol =(
    OUTPUT_DIR /protocol_path .name 
    )
    final_amendment =(
    OUTPUT_DIR /amendment_path .name 
    )

    print ("\n=== V4 PREPARATION COMPLETE ===")
    print ("output:",OUTPUT_DIR )
    print ("manifest:",final_manifest )
    print (
    "manifest_sha256:",
    sha256 (final_manifest ),
    )
    print ("protocol:",final_protocol )
    print (
    "protocol_sha256:",
    sha256 (final_protocol ),
    )
    print ("amendment:",final_amendment )
    print (
    "amendment_sha256:",
    sha256 (final_amendment ),
    )
    print ("training.cache: False")
    print ("training.deterministic: True")
    print ("completed_v3_epochs: 0")
    print (
    "PASS: deterministic V4 protocol frozen "
    "before formal training"
    )


if __name__ =="__main__":
    main ()
