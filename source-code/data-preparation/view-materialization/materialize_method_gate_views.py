from __future__ import annotations 

import argparse 
import csv 
import hashlib 
import json 
import math 
import os 
import random 
import shutil 
import statistics 
from collections import Counter ,defaultdict 
from concurrent .futures import ThreadPoolExecutor 
from datetime import datetime ,timezone 
from pathlib import Path 

from PIL import Image ,ImageFile 

ImageFile .LOAD_TRUNCATED_IMAGES =True 

SEED =20260809 
EXPECTED_VIEWS_PER_CROP_ARM =19_500 
EXPECTED_COMMON_BASE =45_500 
EXPECTED_UNIFORM_VIEWS =65_000 
EXCLUDED_ANNOTATION_ID =1005741 

ROOT =Path (
"/root/rivermind-data/autodrive/results/evaluation"
)
BDD_ROOT =Path (
"/root/rivermind-data/autodrive/datasets/datasets/"
"bdd100k_final"
)
TRAIN_COCO =(
BDD_ROOT 
/"coco/annotations/instances_train.json"
)

PROTOCOL =(
ROOT 
/"method_gate/frozen_manifests/"
"method_gate_protocol_v1.json"
)
SAMPLING_FREEZE =(
ROOT 
/"method_gate/frozen_manifests/"
"cafr_sampling_protocol_v1.json"
)
SAMPLING_ROOT =(
ROOT /"method_gate/sampling_audit_v3"
)
SAMPLING_MANIFEST =(
SAMPLING_ROOT /"artifact_manifest.json"
)
CROP_SMOKE_SUMMARY =(
ROOT 
/"method_gate/object_view_smoke_v1/"
"summary.json"
)
CROP_SMOKE_MANIFEST =(
ROOT 
/"method_gate/object_view_smoke_v1/"
"artifact_manifest.json"
)
TRAIN_DEV_LIST =(
ROOT 
/"architecture_gate/prepared_inputs_v1/"
"train_dev.txt"
)

SHM_FINAL =Path (
"/dev/shm/rebu_yolo/method_gate_v1"
)
PERSISTENT_FINAL =(
ROOT /"method_gate/materialization_v1"
)

EXPECTED_HASHES ={
PROTOCOL :(
"6be7ae25550360e38195bb785a064bd19cb9c4a65b4625b30b5698861cc1b00a"
),
SAMPLING_FREEZE :(
"3812c5692d63629602ebb0393d6369c49f5d84cb5b467bde47a457741a65967d"
),
SAMPLING_MANIFEST :(
"7c96b15970798025e2482613c6fb0304d972e480a7166c5f2e8817b1e7ec93a9"
),
CROP_SMOKE_MANIFEST :(
"937835f1665575fde45da0ad3f7f85ca1eb5b7b00f8451f4256262f89d82434a"
),
TRAIN_DEV_LIST :(
"d4e1c82556eb6adb30fa2d29f6d86e838922d70ad701f7403058b49184c0fcf2"
),
}

CATEGORY_NAMES ={
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


IMAGE_BY_ID ={}
ANNOTATION_BY_ID ={}
ANNOTATIONS_BY_IMAGE ={}
SHM_INCOMPLETE =None 
PROTOCOL_DATA =None 


def sha256_file (path :Path )->str :
    digest =hashlib .sha256 ()
    with path .open ("rb")as handle :
        while True :
            block =handle .read (8 *1024 *1024 )
            if not block :
                break 
            digest .update (block )
    return digest .hexdigest ()


def as_bool (value :str )->bool :
    return str (value ).strip ().lower ()in {
    "1","true","yes","y"
    }


def load_ids (path :Path )->list [int ]:
    data =json .loads (path .read_text ())

    if isinstance (data ,list ):
        values =data 
    elif isinstance (data ,dict ):
        values =None 
        for key in (
        "image_ids",
        "train_core_image_ids",
        "ids",
        ):
            if key in data and isinstance (data [key ],list ):
                values =data [key ]
                break 
        if values is None :
            raise KeyError (path )
    else :
        raise TypeError (type (data ).__name__ )

    return [int (value )for value in values ]


def read_selection (
path :Path ,
expected_arm :str ,
)->list [dict ]:
    rows =[]

    with path .open (
    "r",
    encoding ="utf-8",
    newline ="",
    )as handle :
        reader =csv .DictReader (handle )

        for row in reader :
            if row ["selection_arm"]!=expected_arm :
                raise AssertionError (
                (
                row ["selection_arm"],
                expected_arm ,
                )
                )

            rows .append ({
            "arm":expected_arm ,
            "slot_index":int (row ["slot_index"]),
            "annotation_id":int (row ["annotation_id"]),
            "image_id":int (row ["image_id"]),
            "target_category_id":int (
            row ["target_category_id"]
            ),
            "risk_score":float (row ["risk_score"]),
            "diagnostic_group":(
            row ["diagnostic_group"]
            ),
            })

    rows .sort (
    key =lambda row :row ["slot_index"]
    )

    if len (rows )!=EXPECTED_VIEWS_PER_CROP_ARM :
        raise AssertionError (
        (expected_arm ,len (rows ))
        )

    if len ({
    row ["annotation_id"]for row in rows 
    })!=len (rows ):
        raise AssertionError (
        f"Duplicate annotations: {expected_arm }"
        )

    if len ({
    row ["image_id"]for row in rows 
    })!=len (rows ):
        raise AssertionError (
        f"Duplicate source images: {expected_arm }"
        )

    return rows 


def crop_policy (task :dict )->tuple [int ,int ,float ,float ]:
    materialization =PROTOCOL_DATA [
    "materialization"
    ]

    if task ["arm"]=="scalar_risk":
        policy =materialization [
        "scalar_crop_policy"
        ]
        return (
        int (policy ["minimum_crop_height"]),
        int (policy ["maximum_crop_height"]),
        float (policy ["context_factor"][0 ]),
        float (policy ["context_factor"][1 ]),
        )

    policies =materialization [
    "typed_crop_policies"
    ]
    values =policies [
    task ["diagnostic_group"]
    ]
    return (
    int (values [0 ]),
    int (values [1 ]),
    float (values [2 ]),
    float (values [3 ]),
    )


def choose_crop_window (
image_width :int ,
image_height :int ,
bbox :list [float ],
task :dict ,
)->tuple [int ,int ,int ,int ]:
    x ,y ,width ,height =map (float ,bbox )
    min_height ,max_height ,factor_low ,factor_high =(
    crop_policy (task )
    )



    rng =random .Random (
    SEED *1_000_003 
    +int (task ["annotation_id"])
    )
    factor =rng .uniform (
    factor_low ,
    factor_high ,
    )

    crop_height =max (
    float (min_height ),
    height *factor ,
    width *factor *9.0 /16.0 ,
    )
    crop_height =min (
    crop_height ,
    float (max_height ),
    float (image_height ),
    )
    crop_width =crop_height *16.0 /9.0 

    if crop_width >image_width :
        crop_width =float (image_width )
        crop_height =min (
        float (image_height ),
        crop_width *9.0 /16.0 ,
        )

    crop_width =max (
    min (float (image_width ),crop_width ),
    min (float (image_width ),width ),
    )
    crop_height =max (
    min (float (image_height ),crop_height ),
    min (float (image_height ),height ),
    )

    crop_width =min (
    image_width ,
    max (1 ,int (round (crop_width ))),
    )
    crop_height =min (
    image_height ,
    max (1 ,int (round (crop_height ))),
    )

    target_cx =x +width /2.0 
    target_cy =y +height /2.0 

    jitter_fraction =float (
    PROTOCOL_DATA ["materialization"][
    "center_jitter_fraction"
    ]
    )
    desired_x0 =(
    target_cx 
    -crop_width /2.0 
    +rng .uniform (
    -jitter_fraction ,
    jitter_fraction ,
    )*crop_width 
    )
    desired_y0 =(
    target_cy 
    -crop_height /2.0 
    +rng .uniform (
    -jitter_fraction ,
    jitter_fraction ,
    )*crop_height 
    )

    feasible_x_low =max (
    0.0 ,
    x +width -crop_width ,
    )
    feasible_x_high =min (
    x ,
    float (image_width -crop_width ),
    )
    feasible_y_low =max (
    0.0 ,
    y +height -crop_height ,
    )
    feasible_y_high =min (
    y ,
    float (image_height -crop_height ),
    )

    if feasible_x_low <=feasible_x_high :
        x0 =min (
        max (desired_x0 ,feasible_x_low ),
        feasible_x_high ,
        )
    else :
        x0 =min (
        max (desired_x0 ,0.0 ),
        float (image_width -crop_width ),
        )

    if feasible_y_low <=feasible_y_high :
        y0 =min (
        max (desired_y0 ,feasible_y_low ),
        feasible_y_high ,
        )
    else :
        y0 =min (
        max (desired_y0 ,0.0 ),
        float (image_height -crop_height ),
        )

    x0 =min (
    max (0 ,int (round (x0 ))),
    image_width -crop_width ,
    )
    y0 =min (
    max (0 ,int (round (y0 ))),
    image_height -crop_height ,
    )

    return (
    x0 ,
    y0 ,
    x0 +crop_width ,
    y0 +crop_height ,
    )


def transform_labels (
annotations :list [dict ],
crop_box :tuple [int ,int ,int ,int ],
target_annotation_id :int ,
)->tuple [list [str ],float ]:
    x0 ,y0 ,x1 ,y1 =crop_box 
    crop_width =x1 -x0 
    crop_height =y1 -y0 

    materialization =PROTOCOL_DATA [
    "materialization"
    ]
    minimum_visible =float (
    materialization [
    "non_target_box_minimum_visible_fraction"
    ]
    )
    minimum_pixels =float (
    materialization [
    "minimum_retained_box_pixels"
    ]
    )

    rows =[]
    target_rows =0 
    target_projected_short_side =0.0 

    for annotation in annotations :
        annotation_id =int (annotation ["id"])

        if annotation_id ==EXCLUDED_ANNOTATION_ID :
            continue 
        if int (annotation .get ("iscrowd",0 ))!=0 :
            continue 

        category_id =int (annotation ["category_id"])
        if not 1 <=category_id <=10 :
            continue 

        bx ,by ,bw ,bh =map (
        float ,
        annotation ["bbox"],
        )
        if bw <=0.0 or bh <=0.0 :
            continue 

        ix0 =max (bx ,float (x0 ))
        iy0 =max (by ,float (y0 ))
        ix1 =min (bx +bw ,float (x1 ))
        iy1 =min (by +bh ,float (y1 ))

        iw =max (0.0 ,ix1 -ix0 )
        ih =max (0.0 ,iy1 -iy0 )

        original_area =bw *bh 
        visible_fraction =(
        iw *ih /original_area 
        if original_area >0.0 
        else 0.0 
        )
        center_inside =(
        x0 <=bx +bw /2.0 <=x1 
        and y0 <=by +bh /2.0 <=y1 
        )
        is_target =(
        annotation_id 
        ==target_annotation_id 
        )

        if not is_target :
            if (
            not center_inside 
            or visible_fraction <minimum_visible 
            or iw <minimum_pixels 
            or ih <minimum_pixels 
            ):
                continue 

        if iw <=0.0 or ih <=0.0 :
            continue 

        center_x =(
        ix0 -x0 +iw /2.0 
        )/crop_width 
        center_y =(
        iy0 -y0 +ih /2.0 
        )/crop_height 
        normalized_width =iw /crop_width 
        normalized_height =ih /crop_height 

        values =(
        center_x ,
        center_y ,
        normalized_width ,
        normalized_height ,
        )

        if not all (
        math .isfinite (value )
        for value in values 
        ):
            raise AssertionError (
            (annotation_id ,values )
            )

        if not all (
        0.0 <=value <=1.0 
        for value in values 
        ):
            raise AssertionError (
            (annotation_id ,values ,crop_box )
            )

        rows .append (
        f"{category_id -1 } "
        f"{center_x :.8f} "
        f"{center_y :.8f} "
        f"{normalized_width :.8f} "
        f"{normalized_height :.8f}"
        )

        if is_target :
            target_rows +=1 
            letterbox_scale =min (
            960.0 /crop_width ,
            960.0 /crop_height ,
            )
            target_projected_short_side =(
            min (iw ,ih )
            *letterbox_scale 
            )

    if target_rows !=1 :
        raise AssertionError ({
        "target_annotation_id":(
        target_annotation_id 
        ),
        "target_rows":target_rows ,
        "crop_box":crop_box ,
        })

    return rows ,target_projected_short_side 


def materialize_one (task :dict )->dict :
    annotation_id =int (task ["annotation_id"])
    image_id =int (task ["image_id"])

    target =ANNOTATION_BY_ID [
    annotation_id 
    ]
    if int (target ["image_id"])!=image_id :
        raise AssertionError (
        (annotation_id ,image_id )
        )

    image_record =IMAGE_BY_ID [image_id ]
    image_width =int (image_record ["width"])
    image_height =int (image_record ["height"])

    source_path =(
    BDD_ROOT /image_record ["file_name"]
    )
    if not source_path .is_file ():
        raise FileNotFoundError (source_path )

    crop_box =choose_crop_window (
    image_width ,
    image_height ,
    target ["bbox"],
    task ,
    )

    arm =task ["arm"]
    output_stem =(
    f"{task ['slot_index']:05d}_"
    f"{annotation_id }"
    )
    image_output =(
    SHM_INCOMPLETE 
    /arm 
    /"images"
    /f"{output_stem }.jpg"
    )
    label_output =(
    SHM_INCOMPLETE 
    /arm 
    /"labels"
    /f"{output_stem }.txt"
    )

    with Image .open (source_path )as image :
        image =image .convert ("RGB")
        if image .size !=(
        image_width ,
        image_height ,
        ):
            raise AssertionError (
            (
            source_path ,
            image .size ,
            (
            image_width ,
            image_height ,
            ),
            )
            )

        cropped =image .crop (crop_box )
        cropped .save (
        image_output ,
        format ="JPEG",
        quality =int (
        PROTOCOL_DATA [
        "materialization"
        ]["jpeg_quality"]
        ),
        subsampling =2 ,
        optimize =False ,
        )

    label_rows ,projected_short_side =(
    transform_labels (
    ANNOTATIONS_BY_IMAGE [image_id ],
    crop_box ,
    annotation_id ,
    )
    )

    if not label_rows :
        raise AssertionError (
        f"No labels: {annotation_id }"
        )

    label_output .write_text (
    "\n".join (label_rows )+"\n",
    encoding ="utf-8",
    )

    return {
    **task ,
    "source_file":image_record ["file_name"],
    "crop_box_xyxy":list (crop_box ),
    "crop_width":crop_box [2 ]-crop_box [0 ],
    "crop_height":crop_box [3 ]-crop_box [1 ],
    "objects_in_view":len (label_rows ),
    "target_projected_short_side_at_960":(
    projected_short_side 
    ),
    "relative_image":(
    f"{arm }/images/{image_output .name }"
    ),
    "relative_label":(
    f"{arm }/labels/{label_output .name }"
    ),
    "image_bytes":image_output .stat ().st_size ,
    "label_bytes":label_output .stat ().st_size ,
    "image_sha256":sha256_file (
    image_output 
    ),
    "label_sha256":sha256_file (
    label_output 
    ),
    }


def yaml_text (train_list :Path )->str :
    lines =[
    f"path: {BDD_ROOT }",
    f"train: {train_list }",
    f"val: {TRAIN_DEV_LIST }",
    "",
    "names:",
    ]
    for class_id ,name in CATEGORY_NAMES .items ():
        lines .append (f"  {class_id }: {name }")
    return "\n".join (lines )+"\n"


def main ()->None :
    parser =argparse .ArgumentParser ()
    parser .add_argument (
    "--execute",
    action ="store_true",
    )
    parser .add_argument (
    "--workers",
    type =int ,
    default =16 ,
    )
    args =parser .parse_args ()

    print ("=== MATERIALIZATION INPUT PREFLIGHT ===")

    for path ,expected in EXPECTED_HASHES .items ():
        if not path .is_file ():
            raise FileNotFoundError (path )

        actual =sha256_file (path )
        print (path )
        print (" expected:",expected )
        print (" actual  :",actual )

        if actual !=expected :
            raise AssertionError (path )

    global PROTOCOL_DATA 
    PROTOCOL_DATA =json .loads (
    PROTOCOL .read_text ()
    )

    assert PROTOCOL_DATA ["status"]==(
    "frozen_before_materialization_and_training"
    )
    assert PROTOCOL_DATA [
    "images_materialized"
    ]is False 
    assert PROTOCOL_DATA [
    "training_started"
    ]is False 

    sampling_manifest =json .loads (
    SAMPLING_MANIFEST .read_text ()
    )

    for name ,record in (
    sampling_manifest ["files"].items ()
    ):
        path =SAMPLING_ROOT /name 
        actual =sha256_file (path )
        if actual !=record ["sha256"]:
            raise AssertionError (path )

    smoke_summary =json .loads (
    CROP_SMOKE_SUMMARY .read_text ()
    )
    expected_train_coco_sha256 =(
    smoke_summary ["inputs"][
    "train_coco_sha256"
    ]
    )
    actual_train_coco_sha256 =(
    sha256_file (TRAIN_COCO )
    )

    print ("train_coco_expected:",expected_train_coco_sha256 )
    print ("train_coco_actual  :",actual_train_coco_sha256 )
    assert (
    actual_train_coco_sha256 
    ==expected_train_coco_sha256 
    )

    scalar_rows =read_selection (
    SAMPLING_ROOT 
    /"scalar_risk_selection.csv",
    "scalar_risk",
    )
    typed_rows =read_selection (
    SAMPLING_ROOT 
    /"typed_cafr_selection.csv",
    "typed_cafr",
    )

    common_ids =load_ids (
    SAMPLING_ROOT 
    /"common_base_image_ids.json"
    )
    supplement_ids =load_ids (
    SAMPLING_ROOT 
    /"uniform_supplement_image_ids.json"
    )

    assert len (common_ids )==EXPECTED_COMMON_BASE 
    assert len (supplement_ids )==(
    EXPECTED_VIEWS_PER_CROP_ARM 
    )
    assert not (
    set (common_ids )
    &set (supplement_ids )
    )

    shm_usage =shutil .disk_usage ("/dev/shm")
    print (
    "shm_free_GiB:",
    shm_usage .free /(1024 **3 ),
    )
    assert shm_usage .free >=4 *1024 **3 

    print ("scalar_rows:",len (scalar_rows ))
    print ("typed_rows:",len (typed_rows ))
    print ("common_base:",len (common_ids ))
    print ("uniform_supplement:",len (supplement_ids ))
    print ("workers:",args .workers )

    if SHM_FINAL .exists ():
        raise FileExistsError (
        f"Refusing to overwrite: {SHM_FINAL }"
        )
    if PERSISTENT_FINAL .exists ():
        raise FileExistsError (
        f"Refusing to overwrite: {PERSISTENT_FINAL }"
        )

    if not args .execute :
        print ()
        print ("PASS: materialization preflight only")
        print ("NOTHING CREATED")
        return 

    token =str (os .getpid ())
    global SHM_INCOMPLETE 
    SHM_INCOMPLETE =Path (
    str (SHM_FINAL )
    +f".incomplete-{token }"
    )
    persistent_incomplete =Path (
    str (PERSISTENT_FINAL )
    +f".incomplete-{token }"
    )

    if SHM_INCOMPLETE .exists ():
        raise FileExistsError (SHM_INCOMPLETE )
    if persistent_incomplete .exists ():
        raise FileExistsError (
        persistent_incomplete 
        )

    for arm in (
    "scalar_risk",
    "typed_cafr",
    ):
        (
        SHM_INCOMPLETE /arm /"images"
        ).mkdir (parents =True )
        (
        SHM_INCOMPLETE /arm /"labels"
        ).mkdir (parents =True )

    persistent_incomplete .mkdir (parents =True )

    print ()
    print ("=== LOADING TRAIN COCO ===")
    coco =json .loads (
    TRAIN_COCO .read_text ()
    )

    if len (coco ["images"])!=70_000 :
        raise AssertionError (
        len (coco ["images"])
        )
    if len (coco ["annotations"])!=1_286_852 :
        raise AssertionError (
        len (coco ["annotations"])
        )

    global IMAGE_BY_ID 
    global ANNOTATION_BY_ID 
    global ANNOTATIONS_BY_IMAGE 

    IMAGE_BY_ID ={
    int (image ["id"]):image 
    for image in coco ["images"]
    }
    ANNOTATION_BY_ID ={
    int (annotation ["id"]):annotation 
    for annotation in coco ["annotations"]
    }

    grouped =defaultdict (list )
    for annotation in coco ["annotations"]:
        grouped [
        int (annotation ["image_id"])
        ].append (annotation )
    ANNOTATIONS_BY_IMAGE =dict (grouped )

    for task in scalar_rows +typed_rows :
        target =ANNOTATION_BY_ID .get (
        task ["annotation_id"]
        )
        if target is None :
            raise KeyError (
            task ["annotation_id"]
            )
        if int (target ["image_id"])!=(
        task ["image_id"]
        ):
            raise AssertionError (task )

    print ("images:",len (IMAGE_BY_ID ))
    print (
    "annotations:",
    len (ANNOTATION_BY_ID ),
    )

    tasks =scalar_rows +typed_rows 

    print ()
    print ("=== MATERIALIZING RAM VIEWS ===")

    results =[]
    with ThreadPoolExecutor (
    max_workers =args .workers 
    )as executor :
        for index ,result in enumerate (
        executor .map (
        materialize_one ,
        tasks ,
        ),
        start =1 ,
        ):
            results .append (result )

            if index %1000 ==0 :
                print (
                f"materialized={index }/"
                f"{len (tasks )}"
                )

    if len (results )!=39_000 :
        raise AssertionError (len (results ))

    results .sort (
    key =lambda row :(
    row ["arm"],
    row ["slot_index"],
    )
    )

    by_arm =defaultdict (list )
    for result in results :
        by_arm [result ["arm"]].append (result )

    for arm in (
    "scalar_risk",
    "typed_cafr",
    ):
        if len (by_arm [arm ])!=(
        EXPECTED_VIEWS_PER_CROP_ARM 
        ):
            raise AssertionError (
            (arm ,len (by_arm [arm ]))
            )

    print ()
    print ("=== WRITING TRAIN LISTS ===")

    def original_path (image_id :int )->str :
        record =IMAGE_BY_ID [image_id ]
        path =BDD_ROOT /record ["file_name"]
        if not path .is_file ():
            raise FileNotFoundError (path )
        return str (path )

    common_paths =[
    original_path (image_id )
    for image_id in common_ids 
    ]
    supplement_paths =[
    original_path (image_id )
    for image_id in supplement_ids 
    ]

    uniform_paths =(
    common_paths +supplement_paths 
    )

    scalar_crop_paths =[
    str (
    SHM_FINAL 
    /result ["relative_image"]
    )
    for result in by_arm ["scalar_risk"]
    ]
    typed_crop_paths =[
    str (
    SHM_FINAL 
    /result ["relative_image"]
    )
    for result in by_arm ["typed_cafr"]
    ]

    scalar_paths =(
    common_paths +scalar_crop_paths 
    )
    typed_paths =(
    common_paths +typed_crop_paths 
    )

    for name ,paths in (
    ("uniform",uniform_paths ),
    ("scalar_risk",scalar_paths ),
    ("typed_cafr",typed_paths ),
    ):
        if len (paths )!=EXPECTED_UNIFORM_VIEWS :
            raise AssertionError (
            (name ,len (paths ))
            )
        if len (set (paths ))!=len (paths ):
            raise AssertionError (
            f"Duplicate paths: {name }"
            )

        output =(
        persistent_incomplete 
        /f"{name }_train.txt"
        )
        output .write_text (
        "\n".join (paths )+"\n",
        encoding ="utf-8",
        )

    for name in (
    "uniform",
    "scalar_risk",
    "typed_cafr",
    ):
        final_train_list =(
        PERSISTENT_FINAL 
        /f"{name }_train.txt"
        )
        yaml_path =(
        persistent_incomplete 
        /f"{name }_data.yaml"
        )
        yaml_path .write_text (
        yaml_text (final_train_list ),
        encoding ="utf-8",
        )

    print ()
    print ("=== WRITING MATERIALIZATION MANIFEST ===")

    manifest_csv =(
    persistent_incomplete 
    /"crop_materialization_manifest.csv"
    )

    fields =[
    "arm",
    "slot_index",
    "annotation_id",
    "image_id",
    "target_category_id",
    "risk_score",
    "diagnostic_group",
    "source_file",
    "crop_box_xyxy",
    "crop_width",
    "crop_height",
    "objects_in_view",
    "target_projected_short_side_at_960",
    "relative_image",
    "relative_label",
    "image_bytes",
    "label_bytes",
    "image_sha256",
    "label_sha256",
    ]

    rolling =hashlib .sha256 ()
    payload_bytes =0 
    projected_sizes =[]
    object_counts =[]

    with manifest_csv .open (
    "w",
    encoding ="utf-8",
    newline ="",
    )as handle :
        writer =csv .DictWriter (
        handle ,
        fieldnames =fields ,
        )
        writer .writeheader ()

        for result in results :
            row =dict (result )
            row ["crop_box_xyxy"]=json .dumps (
            row ["crop_box_xyxy"],
            separators =(",",":"),
            )
            writer .writerow (row )

            rolling .update (
            result ["relative_image"].encode ()
            )
            rolling .update (
            result ["image_sha256"].encode ()
            )
            rolling .update (
            result ["label_sha256"].encode ()
            )

            payload_bytes +=(
            result ["image_bytes"]
            +result ["label_bytes"]
            )
            projected_sizes .append (
            result [
            "target_projected_short_side_at_960"
            ]
            )
            object_counts .append (
            result ["objects_in_view"]
            )

    summary ={
    "version":"method_gate_materialization_v1",
    "status":"complete",
    "created_utc":datetime .now (
    timezone .utc 
    ).isoformat (),
    "protocol":str (PROTOCOL ),
    "protocol_sha256":EXPECTED_HASHES [
    PROTOCOL 
    ],
    "train_coco":str (TRAIN_COCO ),
    "train_coco_sha256":(
    actual_train_coco_sha256 
    ),
    "views":{
    "uniform":65000 ,
    "scalar_risk":65000 ,
    "typed_cafr":65000 ,
    "common_original":45500 ,
    "scalar_crops":19500 ,
    "typed_crops":19500 ,
    },
    "crop_payload":{
    "files":78000 ,
    "bytes":payload_bytes ,
    "GiB":(
    payload_bytes /(1024 **3 )
    ),
    "rolling_sha256":(
    rolling .hexdigest ()
    ),
    },
    "geometry":{
    "target_short_side_min":min (
    projected_sizes 
    ),
    "target_short_side_median":(
    statistics .median (
    projected_sizes 
    )
    ),
    "target_short_side_max":max (
    projected_sizes 
    ),
    "objects_per_view_mean":(
    statistics .mean (object_counts )
    ),
    },
    "shm_output":str (SHM_FINAL ),
    "persistent_output":str (
    PERSISTENT_FINAL 
    ),
    "training_started":False ,
    }

    summary_path =(
    persistent_incomplete /"summary.json"
    )
    summary_path .write_text (
    json .dumps (
    summary ,
    indent =2 ,
    ensure_ascii =False ,
    )+"\n",
    encoding ="utf-8",
    )

    artifact_files ={}
    for path in sorted (
    persistent_incomplete .iterdir ()
    ):
        if not path .is_file ():
            continue 
        artifact_files [path .name ]={
        "bytes":path .stat ().st_size ,
        "sha256":sha256_file (path ),
        }

    artifact_manifest ={
    "version":"method_gate_materialization_v1",
    "status":"complete",
    "files":artifact_files ,
    "ephemeral_crop_set_sha256":(
    rolling .hexdigest ()
    ),
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

    SHM_INCOMPLETE .rename (SHM_FINAL )
    persistent_incomplete .rename (
    PERSISTENT_FINAL 
    )

    final_shm_usage =shutil .disk_usage (
    "/dev/shm"
    )

    print ()
    print ("="*92 )
    print ("METHOD-GATE RAM MATERIALIZATION COMPLETE")
    print ("="*92 )
    print ("scalar_crops:",len (
    by_arm ["scalar_risk"]
    ))
    print ("typed_crops:",len (
    by_arm ["typed_cafr"]
    ))
    print (
    "payload_GiB:",
    round (
    payload_bytes /(1024 **3 ),
    4 ,
    ),
    )
    print (
    "target_short_side_median:",
    round (
    statistics .median (
    projected_sizes 
    ),
    4 ,
    ),
    )
    print (
    "objects_per_view_mean:",
    round (
    statistics .mean (object_counts ),
    4 ,
    ),
    )
    print (
    "shm_free_after_GiB:",
    round (
    final_shm_usage .free 
    /(1024 **3 ),
    4 ,
    ),
    )
    print ("shm_output:",SHM_FINAL )
    print (
    "persistent_output:",
    PERSISTENT_FINAL ,
    )
    print (
    "artifact_manifest_sha256:",
    sha256_file (
    PERSISTENT_FINAL 
    /"artifact_manifest.json"
    ),
    )
    print (
    "PASS: both equal-budget crop arms "
    "are fully materialized"
    )
    print ("NOTHING TRAINED")


if __name__ =="__main__":
    main ()
