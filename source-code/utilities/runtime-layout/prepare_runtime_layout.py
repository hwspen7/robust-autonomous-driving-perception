from pathlib import Path 
import argparse 
import csv 
import hashlib 
import json 
import os 
import platform 


EXACT_COMPATIBILITY_ROOT =Path ("/root/rivermind-data")


def sha256 (path :Path )->str :
    digest =hashlib .sha256 ()
    with path .open ("rb")as handle :
        for block in iter (lambda :handle .read (1024 *1024 ),b""):
            digest .update (block )
    return digest .hexdigest ()


def project_root_from_script ()->Path :
    return Path (__file__ ).resolve ().parents [3 ]


def load_script_mapping (project_root :Path ):
    inventory =(
    project_root 
    /"project-management"
    /"inventories"
    /"source-code-classification.tsv"
    )

    if not inventory .is_file ():
        raise FileNotFoundError (inventory )

    mappings =[]

    with inventory .open (
    "r",
    encoding ="utf-8",
    newline ="",
    )as handle :
        reader =csv .DictReader (handle ,delimiter ="\t")

        for row in reader :
            original =row ["source_relative_path"]

            if original .startswith ("._"):
                continue 

            if Path (original ).suffix not in {".py",".sh"}:
                continue 

            active =(
            project_root 
            /row ["destination_relative_path"]
            ).resolve ()

            if not active .is_file ():
                raise FileNotFoundError (active )

            actual =sha256 (active )
            expected =row ["sha256"]

            if actual !=expected :
                raise RuntimeError ({
                "file":str (active ),
                "expected_sha256":expected ,
                "actual_sha256":actual ,
                })

            mappings .append ({
            "original":original ,
            "active":active ,
            "sha256":actual ,
            })

    if len (mappings )!=68 :
        raise RuntimeError (
        f"Expected 68 frozen project scripts, "
        f"found {len (mappings )}"
        )

    return mappings 


def ensure_link (
target :Path ,
link :Path ,
):
    target =target .resolve ()

    if os .path .lexists (link ):
        if link .is_symlink ()and link .resolve ()==target :
            return "existing"

        raise FileExistsError (
        f"Refusing to replace existing runtime path: {link }"
        )

    link .parent .mkdir (
    parents =True ,
    exist_ok =True ,
    )
    link .symlink_to (
    target ,
    target_is_directory =target .is_dir (),
    )
    return "created"


def preflight (project_root :Path ):
    mappings =load_script_mapping (project_root )

    required =[
    project_root /"source-code",
    project_root /"configs",
    (
    project_root 
    /"source-code"
    /"third-party"
    /"D-FINE"
    ),
    ]

    for path in required :
        if not path .exists ():
            raise FileNotFoundError (path )

    print ("project_root:",project_root )
    print ("verified_frozen_scripts:",len (mappings ))
    print (
    "compatibility_root:",
    EXACT_COMPATIBILITY_ROOT ,
    )
    print (
    "runtime_code_root:",
    EXACT_COMPATIBILITY_ROOT 
    /"autodrive/code/"
    /"robust-autonomous-driving-perception",
    )
    print (
    "runtime_dataset_root:",
    EXACT_COMPATIBILITY_ROOT 
    /"autodrive/datasets/datasets/bdd100k_final",
    )
    print (
    "runtime_results_root:",
    EXACT_COMPATIBILITY_ROOT 
    /"autodrive/results",
    )
    print ("PASS: frozen source hashes are valid")

    return mappings 


def execute (
project_root :Path ,
dataset_root :Path ,
results_root :Path ,
):
    if platform .system ()!="Linux":
        raise RuntimeError (
        "--execute must run inside a compatible Linux "
        "environment"
        )

    if not dataset_root .is_dir ():
        raise FileNotFoundError (dataset_root )

    if not results_root .is_dir ():
        raise FileNotFoundError (results_root )

    mappings =preflight (project_root )

    compatibility_root =EXACT_COMPATIBILITY_ROOT 
    code_root =(
    compatibility_root 
    /"autodrive"
    /"code"
    /"robust-autonomous-driving-perception"
    )
    scripts_root =code_root /"scripts"
    scratch_root =Path ("/dev/shm/rebu_yolo")

    scripts_root .mkdir (
    parents =True ,
    exist_ok =True ,
    )
    scratch_root .mkdir (
    parents =True ,
    exist_ok =True ,
    )

    links =[]

    fixed_links =[
    (
    dataset_root ,
    compatibility_root 
    /"autodrive/datasets/datasets/bdd100k_final",
    ),
    (
    results_root ,
    compatibility_root /"autodrive/results",
    ),
    (
    project_root /"configs",
    code_root /"configs",
    ),
    (
    project_root /"source-code",
    code_root /"source-code",
    ),
    (
    project_root /"source-code/third-party",
    code_root /"third_party",
    ),
    ]

    for target ,link in fixed_links :
        state =ensure_link (target ,link )
        links .append ({
        "link":str (link ),
        "target":str (target .resolve ()),
        "state":state ,
        })

    for record in mappings :
        link =scripts_root /record ["original"]
        state =ensure_link (record ["active"],link )

        links .append ({
        "link":str (link ),
        "target":str (record ["active"]),
        "sha256":record ["sha256"],
        "state":state ,
        })

    initialization =(
    project_root 
    /"data-and-baselines"
    /"yolo11m"
    /"pretrained"
    /"yolo11m-coco.pt"
    )

    if initialization .is_file ():
        link =code_root /"yolo11m.pt"
        state =ensure_link (initialization ,link )

        links .append ({
        "link":str (link ),
        "target":str (initialization .resolve ()),
        "state":state ,
        })

    manifest ={
    "status":"complete",
    "version":"portable-runtime-layout",
    "project_root":str (project_root ),
    "dataset_root":str (dataset_root .resolve ()),
    "results_root":str (results_root .resolve ()),
    "compatibility_root":str (
    compatibility_root 
    ),
    "scratch_root":str (scratch_root ),
    "frozen_scripts":len (mappings ),
    "links":links ,
    }

    manifest_path =(
    compatibility_root 
    /"runtime-layout-manifest.json"
    )

    manifest_path .write_text (
    json .dumps (
    manifest ,
    indent =2 ,
    ensure_ascii =False ,
    )+"\n",
    encoding ="utf-8",
    )

    print ("created_or_verified_links:",len (links ))
    print ("manifest:",manifest_path )
    print ("PASS: exact frozen runtime layout is ready")


def main ():
    parser =argparse .ArgumentParser (
    description =(
    "Prepare the exact runtime paths used by the "
    "frozen autonomous-driving experiments."
    )
    )

    mode =parser .add_mutually_exclusive_group (
    required =True 
    )
    mode .add_argument (
    "--preflight-only",
    action ="store_true",
    )
    mode .add_argument (
    "--execute",
    action ="store_true",
    )

    parser .add_argument (
    "--project-root",
    type =Path ,
    default =project_root_from_script (),
    )
    parser .add_argument (
    "--dataset-root",
    type =Path ,
    )
    parser .add_argument (
    "--results-root",
    type =Path ,
    )

    args =parser .parse_args ()
    project_root =args .project_root .expanduser ().resolve ()

    if args .preflight_only :
        preflight (project_root )
        print ("NOTHING LINKED OR MODIFIED")
        return 

    if args .dataset_root is None :
        parser .error ("--dataset-root is required with --execute")

    if args .results_root is None :
        parser .error ("--results-root is required with --execute")

    execute (
    project_root =project_root ,
    dataset_root =args .dataset_root .expanduser ().resolve (),
    results_root =args .results_root .expanduser ().resolve (),
    )


if __name__ =="__main__":
    main ()
