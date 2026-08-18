from __future__ import annotations 

import argparse 
import csv 
import hashlib 
import json 
import math 
import os 
import time 
from collections import Counter ,defaultdict 
from functools import lru_cache 
from pathlib import Path 

import numpy as np 


EVAL_ROOT =Path (
"/root/rivermind-data/autodrive/results/evaluation"
)
FINAL_ROOT =EVAL_ROOT /"final_analysis_v1"

PROTOCOL =(
FINAL_ROOT 
/"frozen_protocol/final_testing_protocol_v1.json"
)
BOOTSTRAP_MANIFEST =(
FINAL_ROOT 
/"paired_bootstrap_v1/artifact_manifest.json"
)
FORMAL_ROOT =(
EVAL_ROOT 
/"formal_training/prepared_inputs_v1"
)
MATERIALIZATION_ROOT =(
EVAL_ROOT 
/"method_gate/materialization_v1"
)
SAMPLING_ROOT =(
EVAL_ROOT 
/"method_gate/sampling_audit_v3"
)

DATASET_ROOT =Path (
"/root/rivermind-data/autodrive/datasets/datasets/"
"bdd100k_final"
)
TRAIN_COCO =(
DATASET_ROOT 
/"coco/annotations/instances_train.json"
)
ORIGINAL_LABEL_ROOT =(
DATASET_ROOT /"labels/train"
)

UNIFORM_LIST =(
FORMAL_ROOT /"uniform_full70k_train.txt"
)
REBU_LIST =(
FORMAL_ROOT /"rebu_risk_full70k_train.txt"
)
CROP_MANIFEST =(
MATERIALIZATION_ROOT 
/"crop_materialization_manifest.csv"
)
RAM_ROOT =Path (
"/dev/shm/rebu_yolo/method_gate_v1"
)

OUTPUT =(
FINAL_ROOT /"exposure_audit_v1"
)

EXPECTED_SHA256 ={
PROTOCOL :
"693c39375cc1eefc4f3ff9f957a8355fb947552a19dfef228cae6cfc21e3dcfe",
BOOTSTRAP_MANIFEST :
"35b07a65bffb2e1e72a31dcee9cc568c1dea36d181f7bdf68e51401c594c0115",
FORMAL_ROOT /"artifact_manifest.json":
"ccf01cc4092237665dc4d73a587bd2a4f9a454cf7fceb0b4f17712ab8c6854aa",
FORMAL_ROOT /"formal_training_protocol.json":
"c40ad7567ebddd44d96390aa5e3c0340a19cbe6f84f83c944d0442ae69205a14",
UNIFORM_LIST :
"d86707c944b4802105c11438621847e50b0af8512ee8ff9dce19a290adc4542c",
REBU_LIST :
"7a4c8f01a1ecc1ca2c4d6b4c6cd5a0afd341c2a091e8102e08995bd0351dfb92",
MATERIALIZATION_ROOT /"artifact_manifest.json":
"74f26c5cc14a2f5bea497bea921966d5c12cbf81a43ac19c39fe71d226e18c06",
SAMPLING_ROOT /"artifact_manifest.json":
"7c96b15970798025e2482613c6fb0304d972e480a7166c5f2e8817b1e7ec93a9",
TRAIN_COCO :
"a8e9c188e58a328901af6e3a93b747d517da354e2dab3292ff26d7c0fbd32cd2",
}

EXPECTED_VIEWS =70_000 
EXPECTED_CROPS =19_500 
NETWORK_SIZE =960 


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

    return parser .parse_args ()


def sha256_file (path :Path )->str :
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

    actual =sha256_file (path )

    print (path )
    print (" expected:",expected )
    print (" actual  :",actual )

    if actual !=expected :
        raise AssertionError (path )


def write_json (path :Path ,payload )->None :
    path .write_text (
    json .dumps (
    payload ,
    indent =2 ,
    ensure_ascii =False ,
    sort_keys =True ,
    )+"\n"
    )


def write_csv (path :Path ,rows :list [dict ])->None :
    if not rows :
        raise ValueError (path )

    with path .open (
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


def scale_name (area :float )->str :
    if area <32 **2 :
        return "small"
    if area <96 **2 :
        return "medium"
    return "large"


def distribution (values :list [float ])->dict :
    array =np .asarray (values ,dtype =np .float64 )

    if len (array )==0 :
        return {
        "count":0 ,
        "mean":None ,
        "median":None ,
        "p10":None ,
        "p90":None ,
        "minimum":None ,
        "maximum":None ,
        }

    return {
    "count":int (len (array )),
    "mean":float (np .mean (array )),
    "median":float (np .median (array )),
    "p10":float (np .quantile (array ,0.10 )),
    "p90":float (np .quantile (array ,0.90 )),
    "minimum":float (np .min (array )),
    "maximum":float (np .max (array )),
    }


@lru_cache (maxsize =None )
def read_labels (path_string :str ):
    path =Path (path_string )

    if not path .is_file ():
        raise FileNotFoundError (path )

    rows =[]

    for line_number ,raw in enumerate (
    path .read_text ().splitlines (),
    start =1 ,
    ):
        stripped =raw .strip ()

        if not stripped :
            continue 

        values =stripped .split ()

        if len (values )<5 :
            raise ValueError (
            (path ,line_number ,raw )
            )

        class_id =int (float (values [0 ]))
        x_center =float (values [1 ])
        y_center =float (values [2 ])
        width =float (values [3 ])
        height =float (values [4 ])

        numeric =(
        x_center ,
        y_center ,
        width ,
        height ,
        )

        if not all (
        math .isfinite (value )
        for value in numeric 
        ):
            raise ValueError (
            (path ,line_number ,raw )
            )

        if width <=0 or height <=0 :
            raise ValueError (
            (path ,line_number ,raw )
            )

        rows .append ((
        class_id ,
        x_center ,
        y_center ,
        width ,
        height ,
        ))

    return tuple (rows )


def load_list (path :Path )->list [Path ]:
    rows =[
    Path (line .strip ())
    for line in path .read_text ().splitlines ()
    if line .strip ()
    ]

    if len (rows )!=EXPECTED_VIEWS :
        raise AssertionError (
        (path ,len (rows ))
        )

    if len (set (map (str ,rows )))!=EXPECTED_VIEWS :
        raise AssertionError (
        f"Duplicate listed paths: {path }"
        )

    return rows 


def load_crop_manifest ():
    with CROP_MANIFEST .open (
    newline ="",
    encoding ="utf-8",
    )as handle :
        rows =list (csv .DictReader (handle ))

    scalar_rows =[
    row 
    for row in rows 
    if row ["arm"]=="scalar_risk"
    ]

    if len (scalar_rows )!=EXPECTED_CROPS :
        raise AssertionError (
        len (scalar_rows )
        )

    relative_images =[
    row ["relative_image"]
    for row in scalar_rows 
    ]
    annotation_ids =[
    int (row ["annotation_id"])
    for row in scalar_rows 
    ]
    image_ids =[
    int (row ["image_id"])
    for row in scalar_rows 
    ]

    if len (set (relative_images ))!=EXPECTED_CROPS :
        raise AssertionError (
        "Duplicate scalar crop paths."
        )

    if len (set (annotation_ids ))!=EXPECTED_CROPS :
        raise AssertionError (
        "Duplicate target annotations."
        )

    if len (set (image_ids ))!=EXPECTED_CROPS :
        raise AssertionError (
        "Scalar selections must use unique source images."
        )

    return scalar_rows 


def preflight ():
    print ("=== FINAL EXPOSURE AUDIT PREFLIGHT ===")

    for path ,expected in EXPECTED_SHA256 .items ():
        verify (path ,expected )

    protocol =json .loads (PROTOCOL .read_text ())

    exposure =protocol ["exposure_audit"]

    assert exposure ["required"]is True 
    assert (
    exposure ["equal_budget_wording"]
    ==
    "Equal in training views and optimization steps; "
    "not necessarily equal in objects, pixels, "
    "context, or supervision content."
    )

    required_metrics ={
    "training_views",
    "original_views",
    "object_centric_views",
    "unique_source_images",
    "source_image_exposure_counts",
    "total_supervised_objects",
    "target_objects",
    "non_target_objects",
    "objects_per_view",
    "class_exposure",
    "scale_exposure",
    "input_pixel_budget",
    "supervision_density",
    }

    if set (exposure ["metrics"])!=required_metrics :
        raise AssertionError (
        exposure ["metrics"]
        )

    if OUTPUT .exists ():
        raise FileExistsError (OUTPUT )

    incomplete =sorted (
    OUTPUT .parent .glob (
    OUTPUT .name +".incomplete-*"
    )
    )

    if incomplete :
        raise AssertionError (incomplete )

    if not RAM_ROOT .is_dir ():
        raise FileNotFoundError (RAM_ROOT )

    uniform_paths =load_list (UNIFORM_LIST )
    rebu_paths =load_list (REBU_LIST )
    crop_rows =load_crop_manifest ()

    for path in (
    uniform_paths [0 ],
    uniform_paths [-1 ],
    rebu_paths [0 ],
    rebu_paths [-1 ],
    ):
        if not path .resolve ().is_file ():
            raise FileNotFoundError (path )

    free_gib =(
    os .statvfs (OUTPUT .parent ).f_bavail 
    *os .statvfs (OUTPUT .parent ).f_frsize 
    /1024 **3 
    )

    if free_gib <0.25 :
        raise RuntimeError (
        "Insufficient output space."
        )

    print ("uniform_views:",len (uniform_paths ))
    print ("rebu_views:",len (rebu_paths ))
    print ("scalar_crop_manifest:",len (crop_rows ))
    print ("free_GiB:",round (free_gib ,3 ))
    print ("PASS: frozen exposure inputs are valid")

    return (
    protocol ,
    uniform_paths ,
    rebu_paths ,
    crop_rows ,
    )


def verify_crop_payload (
crop_rows :list [dict ],
)->None :
    print ("\n=== VERIFYING 19.5K CROP PAYLOAD ===")

    for index ,row in enumerate (
    crop_rows ,
    start =1 ,
    ):
        image_path =(
        RAM_ROOT /row ["relative_image"]
        )
        label_path =(
        RAM_ROOT /row ["relative_label"]
        )

        if not image_path .is_file ():
            raise FileNotFoundError (image_path )

        if not label_path .is_file ():
            raise FileNotFoundError (label_path )

        if image_path .stat ().st_size !=int (
        row ["image_bytes"]
        ):
            raise AssertionError (image_path )

        if label_path .stat ().st_size !=int (
        row ["label_bytes"]
        ):
            raise AssertionError (label_path )

        if sha256_file (image_path )!=row [
        "image_sha256"
        ]:
            raise AssertionError (image_path )

        if sha256_file (label_path )!=row [
        "label_sha256"
        ]:
            raise AssertionError (label_path )

        labels =read_labels (str (label_path ))

        if len (labels )!=int (
        row ["objects_in_view"]
        ):
            raise AssertionError (
            (
            label_path ,
            len (labels ),
            row ["objects_in_view"],
            )
            )

        target_class =(
        int (row ["target_category_id"])-1 
        )

        if not any (
        label [0 ]==target_class 
        for label in labels 
        ):
            raise AssertionError (
            (
            "Target class absent",
            label_path ,
            target_class ,
            )
            )

        if index %2000 ==0 :
            print (
            f"crop_payload={index }/{len (crop_rows )}",
            flush =True ,
            )

    print ("PASS: crop images and labels match frozen hashes")


def audit_arm (
arm :str ,
listed_paths :list [Path ],
coco_images_by_name :dict [str ,dict ],
crop_by_relative_image :dict [str ,dict ],
):
    print ("\n"+"="*100 )
    print ("AUDITING ARM:",arm )
    print ("="*100 )

    class_counts =Counter ()
    native_scale_counts =Counter ()
    network_scale_counts =Counter ()
    source_counts =Counter ()
    original_source_counts =Counter ()

    object_counts =[]
    raw_pixels =0 
    original_views =0 
    crop_views =0 

    resolved_paths =set ()
    crop_relatives_seen =set ()

    for index ,listed in enumerate (
    listed_paths ,
    start =1 ,
    ):
        resolved =listed .resolve ()

        if not resolved .is_file ():
            raise FileNotFoundError (resolved )

        resolved_string =str (resolved )
        resolved_paths .add (resolved_string )

        if (
        "/bdd100k_final/images/train/"
        in resolved_string 
        ):
            kind ="original"
            original_views +=1 

            image =coco_images_by_name .get (
            resolved .name 
            )

            if image is None :
                raise KeyError (resolved .name )

            source_image_id =int (image ["id"])
            width =int (image ["width"])
            height =int (image ["height"])

            label_path =(
            ORIGINAL_LABEL_ROOT 
            /f"{resolved .stem }.txt"
            )

            original_source_counts [
            source_image_id 
            ]+=1 

        elif (
        "/dev/shm/rebu_yolo/method_gate_v1/"
        in resolved_string 
        ):
            kind ="crop"
            crop_views +=1 

            relative_image =(
            resolved .relative_to (
            RAM_ROOT 
            ).as_posix ()
            )

            crop_row =crop_by_relative_image .get (
            relative_image 
            )

            if crop_row is None :
                raise KeyError (relative_image )

            crop_relatives_seen .add (
            relative_image 
            )

            source_image_id =int (
            crop_row ["image_id"]
            )
            width =int (
            crop_row ["crop_width"]
            )
            height =int (
            crop_row ["crop_height"]
            )

            label_path =(
            RAM_ROOT 
            /crop_row ["relative_label"]
            )

        else :
            raise AssertionError (
            (
            "Unknown training-view path",
            listed ,
            resolved ,
            )
            )

        labels =read_labels (str (label_path ))

        if kind =="crop":
            expected_objects =int (
            crop_row ["objects_in_view"]
            )

            if len (labels )!=expected_objects :
                raise AssertionError (
                (
                label_path ,
                len (labels ),
                expected_objects ,
                )
                )

        source_counts [source_image_id ]+=1 
        object_counts .append (len (labels ))
        raw_pixels +=width *height 

        letterbox_scale =min (
        NETWORK_SIZE /width ,
        NETWORK_SIZE /height ,
        )

        for (
        class_id ,
        _ ,
        _ ,
        normalized_width ,
        normalized_height ,
        )in labels :
            if class_id <0 or class_id >9 :
                raise AssertionError (
                (label_path ,class_id )
                )

            box_width =(
            normalized_width *width 
            )
            box_height =(
            normalized_height *height 
            )
            native_area =(
            box_width *box_height 
            )

            network_area =(
            box_width 
            *letterbox_scale 
            *box_height 
            *letterbox_scale 
            )

            class_counts [class_id ]+=1 
            native_scale_counts [
            scale_name (native_area )
            ]+=1 
            network_scale_counts [
            scale_name (network_area )
            ]+=1 

        if index %5000 ==0 :
            print (
            f"{arm }_views={index }/{len (listed_paths )}",
            flush =True ,
            )

    if len (resolved_paths )!=EXPECTED_VIEWS :
        raise AssertionError (
        (
        arm ,
        "resolved path duplication",
        len (resolved_paths ),
        )
        )

    return {
    "arm":arm ,
    "training_views":len (listed_paths ),
    "original_views":original_views ,
    "object_centric_views":crop_views ,
    "unique_source_images":
    len (source_counts ),
    "source_counts":source_counts ,
    "original_source_counts":
    original_source_counts ,
    "class_counts":class_counts ,
    "native_scale_counts":
    native_scale_counts ,
    "network_scale_counts":
    network_scale_counts ,
    "total_supervised_objects":
    int (sum (object_counts )),
    "object_counts":object_counts ,
    "raw_input_pixels":int (raw_pixels ),
    "network_input_pixels":int (
    len (listed_paths )
    *NETWORK_SIZE 
    *NETWORK_SIZE 
    ),
    "crop_relatives_seen":
    crop_relatives_seen ,
    }


def enrich_target_exposure (
audits :dict [str ,dict ],
crop_rows :list [dict ],
):
    target_exposure_by_arm ={
    arm :Counter ()
    for arm in audits 
    }

    for row in crop_rows :
        source_image_id =int (row ["image_id"])
        class_id =(
        int (row ["target_category_id"])-1 
        )

        uniform_original_occurrences =(
        audits ["uniform"][
        "original_source_counts"
        ][source_image_id ]
        )

        rebu_original_occurrences =(
        audits ["rebu_risk"][
        "original_source_counts"
        ][source_image_id ]
        )

        target_exposure_by_arm [
        "uniform"
        ][class_id ]+=(
        uniform_original_occurrences 
        )

        target_exposure_by_arm [
        "rebu_risk"
        ][class_id ]+=(
        rebu_original_occurrences +1 
        )

    for arm ,audit in audits .items ():
        by_class =target_exposure_by_arm [arm ]
        target_objects =int (sum (
        by_class .values ()
        ))

        audit ["target_exposure_by_class"]=(
        by_class 
        )
        audit ["target_objects"]=target_objects 
        audit ["non_target_objects"]=(
        audit ["total_supervised_objects"]
        -target_objects 
        )

        if audit ["non_target_objects"]<0 :
            raise AssertionError (arm )


def build_outputs (
staging :Path ,
audits :dict [str ,dict ],
category_names :dict [int ,str ],
crop_rows :list [dict ],
started :float ,
):
    arm_rows =[]
    class_rows =[]
    scale_rows =[]
    source_rows =[]
    object_histogram_rows =[]
    target_rows =[]

    summary_arms ={}

    for arm ,audit in audits .items ():
        object_distribution =distribution (
        audit ["object_counts"]
        )

        source_exposure_values =list (
        audit ["source_counts"].values ()
        )
        source_distribution =distribution (
        source_exposure_values 
        )

        objects =audit [
        "total_supervised_objects"
        ]
        views =audit ["training_views"]

        raw_pixels =audit [
        "raw_input_pixels"
        ]
        network_pixels =audit [
        "network_input_pixels"
        ]

        arm_row ={
        "arm":arm ,
        "training_views":views ,
        "original_views":
        audit ["original_views"],
        "object_centric_views":
        audit ["object_centric_views"],
        "unique_source_images":
        audit ["unique_source_images"],
        "total_supervised_objects":
        objects ,
        "target_objects":
        audit ["target_objects"],
        "non_target_objects":
        audit ["non_target_objects"],
        "objects_per_view":
        objects /views ,
        "raw_input_pixels":
        raw_pixels ,
        "network_input_pixels":
        network_pixels ,
        "objects_per_million_raw_pixels":
        objects /raw_pixels *1_000_000 ,
        "objects_per_million_network_pixels":
        objects /network_pixels *1_000_000 ,
        "mean_source_exposures":
        float (np .mean (
        source_exposure_values 
        )),
        "maximum_source_exposures":
        int (max (
        source_exposure_values 
        )),
        }
        arm_rows .append (arm_row )

        summary_arms [arm ]={
        **arm_row ,
        "objects_per_view_distribution":
        object_distribution ,
        "source_image_exposure_distribution":
        source_distribution ,
        "class_exposure":{
        str (class_id ):
        int (audit ["class_counts"][
        class_id 
        ])
        for class_id in range (10 )
        },
        "native_scale_exposure":{
        scale :int (
        audit ["native_scale_counts"][
        scale 
        ]
        )
        for scale in (
        "small",
        "medium",
        "large",
        )
        },
        "network_960_scale_exposure":{
        scale :int (
        audit [
        "network_scale_counts"
        ][scale ]
        )
        for scale in (
        "small",
        "medium",
        "large",
        )
        },
        }

        for class_id in range (10 ):
            count =int (
            audit ["class_counts"][class_id ]
            )

            class_rows .append ({
            "arm":arm ,
            "class_id":class_id ,
            "class_name":
            category_names [class_id ],
            "supervised_objects":count ,
            "fraction_of_arm_objects":
            count /objects ,
            })

            target_rows .append ({
            "arm":arm ,
            "class_id":class_id ,
            "class_name":
            category_names [class_id ],
            "scheduled_target_exposures":
            int (
            audit [
            "target_exposure_by_class"
            ][class_id ]
            ),
            })

        for coordinate_space ,counts in (
        (
        "view_native",
        audit ["native_scale_counts"],
        ),
        (
        "network_960_letterbox",
        audit ["network_scale_counts"],
        ),
        ):
            for scale in (
            "small",
            "medium",
            "large",
            ):
                count =int (counts [scale ])

                scale_rows .append ({
                "arm":arm ,
                "coordinate_space":
                coordinate_space ,
                "scale":scale ,
                "supervised_objects":count ,
                "fraction_of_arm_objects":
                count /objects ,
                })

        for exposure_count ,source_count in sorted (
        Counter (
        audit ["source_counts"].values ()
        ).items ()
        ):
            source_rows .append ({
            "arm":arm ,
            "exposures_per_source_image":
            int (exposure_count ),
            "source_images":
            int (source_count ),
            })

        for object_count ,view_count in sorted (
        Counter (
        audit ["object_counts"]
        ).items ()
        ):
            object_histogram_rows .append ({
            "arm":arm ,
            "objects_in_view":
            int (object_count ),
            "views":int (view_count ),
            })

    uniform =audits ["uniform"]
    rebu =audits ["rebu_risk"]

    equal_views =(
    uniform ["training_views"]
    ==rebu ["training_views"]
    ==EXPECTED_VIEWS 
    )

    equal_network_pixels =(
    uniform ["network_input_pixels"]
    ==rebu ["network_input_pixels"]
    )

    identical_supervision_content =(
    uniform ["total_supervised_objects"]
    ==rebu ["total_supervised_objects"]
    and uniform ["class_counts"]
    ==rebu ["class_counts"]
    and uniform ["network_scale_counts"]
    ==rebu ["network_scale_counts"]
    )

    summary ={
    "version":"final_exposure_audit_v1",
    "status":"complete",
    "created_utc":time .strftime (
    "%Y-%m-%dT%H:%M:%SZ",
    time .gmtime (),
    ),
    "protocol":str (PROTOCOL ),
    "protocol_sha256":
    sha256_file (PROTOCOL ),
    "bootstrap_manifest_sha256":
    sha256_file (
    BOOTSTRAP_MANIFEST 
    ),
    "training_budget":{
    "equal_training_views":
    equal_views ,
    "equal_optimization_steps":
    True ,
    "equal_network_input_pixels":
    equal_network_pixels ,
    "identical_supervision_content":
    identical_supervision_content ,
    "required_wording":(
    "Equal in training views and "
    "optimization steps; not necessarily "
    "equal in objects, pixels, context, "
    "or supervision content."
    ),
    },
    "selected_object_centric_views":
    len (crop_rows ),
    "arms":summary_arms ,
    "rebu_minus_uniform":{
    "training_views":
    rebu ["training_views"]
    -uniform ["training_views"],
    "original_views":
    rebu ["original_views"]
    -uniform ["original_views"],
    "object_centric_views":
    rebu ["object_centric_views"]
    -uniform ["object_centric_views"],
    "unique_source_images":
    rebu ["unique_source_images"]
    -uniform ["unique_source_images"],
    "total_supervised_objects":
    rebu ["total_supervised_objects"]
    -uniform ["total_supervised_objects"],
    "target_objects":
    rebu ["target_objects"]
    -uniform ["target_objects"],
    "non_target_objects":
    rebu ["non_target_objects"]
    -uniform ["non_target_objects"],
    "raw_input_pixels":
    rebu ["raw_input_pixels"]
    -uniform ["raw_input_pixels"],
    "network_input_pixels":
    rebu ["network_input_pixels"]
    -uniform ["network_input_pixels"],
    },
    "interpretation":(
    "This audit measures realized exposure "
    "and supervision composition. It does not "
    "by itself establish a causal explanation "
    "for downstream model differences."
    ),
    "new_training":False ,
    "new_inference":False ,
    "elapsed_minutes":
    (time .time ()-started )/60.0 ,
    }

    write_csv (
    staging /"arm_exposure_summary.csv",
    arm_rows ,
    )
    write_csv (
    staging /"class_exposure.csv",
    class_rows ,
    )
    write_csv (
    staging /"scale_exposure.csv",
    scale_rows ,
    )
    write_csv (
    staging 
    /"source_image_exposure_histogram.csv",
    source_rows ,
    )
    write_csv (
    staging 
    /"objects_per_view_histogram.csv",
    object_histogram_rows ,
    )
    write_csv (
    staging 
    /"target_exposure_by_class.csv",
    target_rows ,
    )
    write_json (
    staging /"summary.json",
    summary ,
    )

    metadata ={
    "version":"final_exposure_audit_v1",
    "status":"complete",
    "script":
    str (Path (__file__ ).resolve ()),
    "script_sha256":
    sha256_file (
    Path (__file__ ).resolve ()
    ),
    "coordinate_definition":{
    "view_native":(
    "BBox area in the decoded original "
    "image or crop coordinate system."
    ),
    "network_960_letterbox":(
    "BBox area after aspect-ratio-preserving "
    "resize to fit inside 960x960; padding "
    "does not change bbox dimensions."
    ),
    },
    "scale_thresholds":{
    "small":"[0, 1024)",
    "medium":"[1024, 9216)",
    "large":"[9216, infinity)",
    },
    "target_object_definition":(
    "Exposure count of the 19,500 scheduled "
    "target annotations: once for each target "
    "crop plus any retained original source-view "
    "exposure in the corresponding arm."
    ),
    "ephemeral_crop_payload_fully_rehashed":
    True ,
    }

    write_json (
    staging /"metadata.json",
    metadata ,
    )


def build_manifest (directory :Path ):
    artifacts =[]

    for path in sorted (directory .rglob ("*")):
        if not path .is_file ():
            continue 
        if path .name =="artifact_manifest.json":
            continue 

        artifacts .append ({
        "path":
        str (path .relative_to (directory )),
        "bytes":path .stat ().st_size ,
        "sha256":sha256_file (path ),
        })

    manifest ={
    "version":"final_exposure_audit_v1",
    "status":"complete",
    "created_utc":time .strftime (
    "%Y-%m-%dT%H:%M:%SZ",
    time .gmtime (),
    ),
    "artifacts":artifacts ,
    }

    path =(
    directory /"artifact_manifest.json"
    )
    write_json (path ,manifest )

    return path ,sha256_file (path )


def execute (
uniform_paths :list [Path ],
rebu_paths :list [Path ],
crop_rows :list [dict ],
):
    started =time .time ()

    staging =OUTPUT .with_name (
    OUTPUT .name 
    +f".incomplete-{int (time .time ())}-{os .getpid ()}"
    )
    staging .mkdir (parents =True )

    verify_crop_payload (crop_rows )

    print ("\n=== LOADING TRAIN COCO ===")

    coco =json .loads (TRAIN_COCO .read_text ())

    images_by_name ={}
    category_names ={}

    for image in coco ["images"]:
        basename =Path (
        image ["file_name"]
        ).name 

        if basename in images_by_name :
            raise AssertionError (basename )

        images_by_name [basename ]=image 

    for category in coco ["categories"]:
        category_id =int (category ["id"])
        class_id =category_id -1 

        category_names [class_id ]=(
        category ["name"]
        )

    if set (category_names )!=set (range (10 )):
        raise AssertionError (category_names )

    crop_by_relative_image ={
    row ["relative_image"]:row 
    for row in crop_rows 
    }

    audits ={
    "uniform":audit_arm (
    "uniform",
    uniform_paths ,
    images_by_name ,
    crop_by_relative_image ,
    ),
    "rebu_risk":audit_arm (
    "rebu_risk",
    rebu_paths ,
    images_by_name ,
    crop_by_relative_image ,
    ),
    }

    if audits ["uniform"][
    "object_centric_views"
    ]!=0 :
        raise AssertionError (
        audits ["uniform"][
        "object_centric_views"
        ]
        )

    if audits ["rebu_risk"][
    "object_centric_views"
    ]!=EXPECTED_CROPS :
        raise AssertionError (
        audits ["rebu_risk"][
        "object_centric_views"
        ]
        )

    expected_crop_paths =set (
    crop_by_relative_image 
    )

    if audits ["rebu_risk"][
    "crop_relatives_seen"
    ]!=expected_crop_paths :
        missing =(
        expected_crop_paths 
        -audits ["rebu_risk"][
        "crop_relatives_seen"
        ]
        )
        extra =(
        audits ["rebu_risk"][
        "crop_relatives_seen"
        ]
        -expected_crop_paths 
        )
        raise AssertionError (
        (
        "crop membership mismatch",
        len (missing ),
        len (extra ),
        )
        )

    enrich_target_exposure (
    audits ,
    crop_rows ,
    )

    build_outputs (
    staging ,
    audits ,
    category_names ,
    crop_rows ,
    started ,
    )

    manifest_path ,manifest_sha256 =(
    build_manifest (staging )
    )

    os .replace (staging ,OUTPUT )

    print ("\n"+"="*100 )
    print ("FINAL EXPOSURE AUDIT COMPLETE")
    print ("="*100 )

    for arm in ("uniform","rebu_risk"):
        audit =audits [arm ]

        print (
        arm ,
        "views=",
        audit ["training_views"],
        "original=",
        audit ["original_views"],
        "crops=",
        audit ["object_centric_views"],
        "sources=",
        audit ["unique_source_images"],
        "objects=",
        audit ["total_supervised_objects"],
        "target_exposures=",
        audit ["target_objects"],
        "objects_per_view=",
        round (
        audit ["total_supervised_objects"]
        /audit ["training_views"],
        6 ,
        ),
        )

    print ("output:",OUTPUT )
    print (
    "manifest:",
    OUTPUT /manifest_path .name ,
    )
    print (
    "manifest_sha256:",
    manifest_sha256 ,
    )
    print ("NOTHING TRAINED OR INFERRED")
    print (
    "NEXT: crop-induced scale and context-shift audit"
    )


def main ():
    args =parse_args ()

    (
    protocol ,
    uniform_paths ,
    rebu_paths ,
    crop_rows ,
    )=preflight ()

    if args .preflight_only :
        print (
        "\nPASS: exposure preflight only\n"
        "NOTHING CREATED, MODIFIED, "
        "TRAINED OR INFERRED"
        )
        return 

    execute (
    uniform_paths ,
    rebu_paths ,
    crop_rows ,
    )


if __name__ =="__main__":
    main ()
