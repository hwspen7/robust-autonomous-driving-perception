from __future__ import annotations 

import argparse 
import csv 
import hashlib 
import json 
import math 
import os 
import time 
from collections import Counter ,defaultdict 
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
EXPOSURE_ROOT =FINAL_ROOT /"exposure_audit_v1"
EXPOSURE_MANIFEST =EXPOSURE_ROOT /"artifact_manifest.json"

METHOD_ROOT =EVAL_ROOT /"method_gate"
METHOD_PROTOCOL =(
METHOD_ROOT 
/"frozen_manifests/method_gate_protocol_v1.json"
)
MATERIAL_ROOT =METHOD_ROOT /"materialization_v1"
MATERIAL_MANIFEST =MATERIAL_ROOT /"artifact_manifest.json"
CROP_MANIFEST =(
MATERIAL_ROOT /"crop_materialization_manifest.csv"
)

DATASET_ROOT =Path (
"/root/rivermind-data/autodrive/datasets/datasets/"
"bdd100k_final"
)
TRAIN_COCO =(
DATASET_ROOT 
/"coco/annotations/instances_train.json"
)

RAM_ROOT =Path ("/dev/shm/rebu_yolo/method_gate_v1")
OUTPUT =FINAL_ROOT /"crop_scale_context_audit_v1"

EXPECTED_SHA256 ={
PROTOCOL :
"693c39375cc1eefc4f3ff9f957a8355fb947552a19dfef228cae6cfc21e3dcfe",
EXPOSURE_MANIFEST :
"94b87f98f4c320da3bc0bf1d87b9c5820682d9dbd1a3443eb4f2f42cdb78d33f",
METHOD_PROTOCOL :
"6be7ae25550360e38195bb785a064bd19cb9c4a65b4625b30b5698861cc1b00a",
MATERIAL_MANIFEST :
"74f26c5cc14a2f5bea497bea921966d5c12cbf81a43ac19c39fe71d226e18c06",
TRAIN_COCO :
"a8e9c188e58a328901af6e3a93b747d517da354e2dab3292ff26d7c0fbd32cd2",
}

EXPECTED_IMAGES =70_000 
EXPECTED_ANNOTATIONS =1_286_852 
EXPECTED_CROPS =19_500 
NETWORK_SIZE =960.0 


def parse_args ():
    parser =argparse .ArgumentParser ()

    group =parser .add_mutually_exclusive_group (required =True )
    group .add_argument ("--preflight-only",action ="store_true")
    group .add_argument ("--execute",action ="store_true")

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


def verify (path :Path ,expected :str ):
    if not path .is_file ():
        raise FileNotFoundError (path )

    actual =sha256_file (path )

    print (path )
    print (" expected:",expected )
    print (" actual  :",actual )

    if actual !=expected :
        raise AssertionError (path )


def write_json (path :Path ,payload ):
    path .write_text (
    json .dumps (
    payload ,
    indent =2 ,
    ensure_ascii =False ,
    sort_keys =True ,
    )+"\n",
    encoding ="utf-8",
    )


def write_csv (path :Path ,rows :list [dict ]):
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


def distribution (values :list [float ])->dict :
    if not values :
        return {
        "count":0 ,
        "mean":None ,
        "median":None ,
        "p10":None ,
        "p90":None ,
        "minimum":None ,
        "maximum":None ,
        }

    array =np .asarray (values ,dtype =np .float64 )

    return {
    "count":int (array .size ),
    "mean":float (array .mean ()),
    "median":float (np .median (array )),
    "p10":float (np .quantile (array ,0.10 )),
    "p90":float (np .quantile (array ,0.90 )),
    "minimum":float (array .min ()),
    "maximum":float (array .max ()),
    }


def scale_name (area :float )->str :
    if area <32 **2 :
        return "small"
    if area <96 **2 :
        return "medium"
    return "large"


def verify_artifact_manifest (
root :Path ,
manifest_path :Path ,
):
    data =json .loads (manifest_path .read_text ())

    if "artifacts"in data :
        records =data ["artifacts"]

        for record in records :
            path =root /record ["path"]

            if not path .is_file ():
                raise FileNotFoundError (path )

            if path .stat ().st_size !=int (record ["bytes"]):
                raise AssertionError (path )

            if sha256_file (path )!=record ["sha256"]:
                raise AssertionError (path )

    elif "files"in data :
        for relative ,record in data ["files"].items ():
            path =root /relative 

            if not path .is_file ():
                raise FileNotFoundError (path )

            if path .stat ().st_size !=int (record ["bytes"]):
                raise AssertionError (path )

            if sha256_file (path )!=record ["sha256"]:
                raise AssertionError (path )

    else :
        raise KeyError (
        f"Unsupported manifest schema: {manifest_path }"
        )


def parse_crop_box (raw :str ):
    values =json .loads (raw )

    if len (values )!=4 :
        raise ValueError (raw )

    x0 ,y0 ,x1 ,y1 =map (float ,values )

    if not (x1 >x0 and y1 >y0 ):
        raise ValueError (raw )

    return x0 ,y0 ,x1 ,y1 


def intersection (
bbox :list [float ],
crop_box :tuple [float ,float ,float ,float ],
):
    bx ,by ,bw ,bh =map (float ,bbox )
    x0 ,y0 ,x1 ,y1 =crop_box 

    ix0 =max (bx ,x0 )
    iy0 =max (by ,y0 )
    ix1 =min (bx +bw ,x1 )
    iy1 =min (by +bh ,y1 )

    width =max (0.0 ,ix1 -ix0 )
    height =max (0.0 ,iy1 -iy0 )
    area =width *height 

    return ix0 ,iy0 ,ix1 ,iy1 ,width ,height ,area 


def load_crop_rows ():
    with CROP_MANIFEST .open (
    newline ="",
    encoding ="utf-8",
    )as handle :
        rows =[
        row 
        for row in csv .DictReader (handle )
        if row ["arm"]=="scalar_risk"
        ]

    if len (rows )!=EXPECTED_CROPS :
        raise AssertionError (len (rows ))

    annotation_ids =[
    int (row ["annotation_id"])
    for row in rows 
    ]
    image_ids =[
    int (row ["image_id"])
    for row in rows 
    ]
    relative_labels =[
    row ["relative_label"]
    for row in rows 
    ]

    if len (set (annotation_ids ))!=EXPECTED_CROPS :
        raise AssertionError (
        "Duplicate target annotation IDs."
        )

    if len (set (image_ids ))!=EXPECTED_CROPS :
        raise AssertionError (
        "Scalar crop source images are not unique."
        )

    if len (set (relative_labels ))!=EXPECTED_CROPS :
        raise AssertionError (
        "Duplicate crop label paths."
        )

    return rows 


def load_coco ():
    data =json .loads (TRAIN_COCO .read_text ())

    if len (data ["images"])!=EXPECTED_IMAGES :
        raise AssertionError (len (data ["images"]))

    if len (data ["annotations"])!=EXPECTED_ANNOTATIONS :
        raise AssertionError (len (data ["annotations"]))

    images ={
    int (row ["id"]):row 
    for row in data ["images"]
    }
    annotations ={
    int (row ["id"]):row 
    for row in data ["annotations"]
    }

    annotations_by_image =defaultdict (list )

    for row in data ["annotations"]:
        annotations_by_image [
        int (row ["image_id"])
        ].append (row )

    category_names ={
    int (row ["id"]):row ["name"]
    for row in data ["categories"]
    }

    if set (category_names )!=set (range (1 ,11 )):
        raise AssertionError (category_names )

    return (
    images ,
    annotations ,
    annotations_by_image ,
    category_names ,
    )


def valid_annotation (
annotation :dict ,
excluded_annotation_id :int ,
):
    if int (annotation ["id"])==excluded_annotation_id :
        return False 

    if int (annotation .get ("iscrowd",0 ))!=0 :
        return False 

    category_id =int (annotation ["category_id"])

    if not 1 <=category_id <=10 :
        return False 

    _ ,_ ,width ,height =map (
    float ,
    annotation ["bbox"],
    )

    return width >0.0 and height >0.0 


def read_label_count (path :Path )->int :
    if not path .is_file ():
        raise FileNotFoundError (path )

    count =0 

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
        numeric =list (map (float ,values [1 :5 ]))

        if not 0 <=class_id <=9 :
            raise ValueError (
            (path ,line_number ,class_id )
            )

        if not all (math .isfinite (v )for v in numeric ):
            raise ValueError (
            (path ,line_number ,raw )
            )

        if numeric [2 ]<=0.0 or numeric [3 ]<=0.0 :
            raise ValueError (
            (path ,line_number ,raw )
            )

        count +=1 

    return count 


def build_manifest (directory :Path ):
    artifacts =[]

    for path in sorted (directory .rglob ("*")):
        if not path .is_file ():
            continue 

        if path .name =="artifact_manifest.json":
            continue 

        artifacts .append ({
        "path":str (path .relative_to (directory )),
        "bytes":path .stat ().st_size ,
        "sha256":sha256_file (path ),
        })

    manifest ={
    "version":"final_crop_scale_context_audit_v1",
    "status":"complete",
    "created_utc":time .strftime (
    "%Y-%m-%dT%H:%M:%SZ",
    time .gmtime (),
    ),
    "artifacts":artifacts ,
    }

    path =directory /"artifact_manifest.json"
    write_json (path ,manifest )

    return path ,sha256_file (path )


def preflight ():
    print ("=== CROP SCALE / CONTEXT AUDIT PREFLIGHT ===")

    for path ,expected in EXPECTED_SHA256 .items ():
        verify (path ,expected )

    verify_artifact_manifest (
    EXPOSURE_ROOT ,
    EXPOSURE_MANIFEST ,
    )
    verify_artifact_manifest (
    MATERIAL_ROOT ,
    MATERIAL_MANIFEST ,
    )

    protocol =json .loads (PROTOCOL .read_text ())
    method_protocol =json .loads (
    METHOD_PROTOCOL .read_text ()
    )

    scale_protocol =protocol ["scale_shift_audit"]
    context_protocol =protocol ["context_shift_audit"]

    assert scale_protocol ["required"]is True 
    assert context_protocol ["required"]is True 
    assert context_protocol ["causal_language_allowed"]is False 

    expected_spaces ={
    "original_image",
    "crop_before_resize",
    "network_input_after_960_letterbox",
    }

    if set (scale_protocol ["coordinate_spaces"])!=expected_spaces :
        raise AssertionError (
        scale_protocol ["coordinate_spaces"]
        )

    if (
    context_protocol ["neighbor_retained_rule"]
    !=
    "At least 50% of the original neighbor bbox area "
    "remains inside the crop before resize."
    ):
        raise AssertionError (
        context_protocol ["neighbor_retained_rule"]
        )

    materialization =method_protocol ["materialization"]

    if float (
    materialization [
    "non_target_box_minimum_visible_fraction"
    ]
    )!=0.5 :
        raise AssertionError (materialization )

    if int (
    materialization ["excluded_annotation_id"]
    )!=1005741 :
        raise AssertionError (materialization )

    exposure_summary =json .loads (
    (EXPOSURE_ROOT /"summary.json").read_text ()
    )

    assert (
    exposure_summary ["arms"]["uniform"][
    "training_views"
    ]
    ==70_000 
    )
    assert (
    exposure_summary ["arms"]["rebu_risk"][
    "object_centric_views"
    ]
    ==EXPECTED_CROPS 
    )

    if not RAM_ROOT .is_dir ():
        raise FileNotFoundError (RAM_ROOT )

    rows =load_crop_rows ()

    for index in (0 ,len (rows )//2 ,len (rows )-1 ):
        label_path =(
        RAM_ROOT /rows [index ]["relative_label"]
        )

        if not label_path .is_file ():
            raise FileNotFoundError (label_path )

    images ,annotations ,_ ,_ =load_coco ()

    for row in rows :
        image_id =int (row ["image_id"])
        annotation_id =int (row ["annotation_id"])

        if image_id not in images :
            raise KeyError (image_id )

        if annotation_id not in annotations :
            raise KeyError (annotation_id )

        if (
        int (annotations [annotation_id ]["image_id"])
        !=image_id 
        ):
            raise AssertionError (
            (annotation_id ,image_id )
            )

    if OUTPUT .exists ():
        raise FileExistsError (OUTPUT )

    incomplete =sorted (
    OUTPUT .parent .glob (
    OUTPUT .name +".incomplete-*"
    )
    )

    if incomplete :
        raise FileExistsError (incomplete )

    print ("crops:",len (rows ))
    print ("ram_root:",RAM_ROOT )
    print ("output:",OUTPUT )
    print ("PASS: frozen geometry and context inputs are valid")
    print ("NOTHING CREATED, MODIFIED, TRAINED OR INFERRED")

    return protocol ,materialization 


def summarize_group (rows :list [dict ])->dict :
    neighbor_total =sum (
    int (row ["source_neighbor_count"])
    for row in rows 
    )
    retained_total =sum (
    int (row ["retained_neighbor_count"])
    for row in rows 
    )

    original_small =[
    row 
    for row in rows 
    if row ["original_scale"]=="small"
    ]

    ceased_small =[
    row 
    for row in original_small 
    if row ["crop_network_scale"]!="small"
    ]

    return {
    "targets":len (rows ),
    "original_small_targets":len (original_small ),
    "original_small_ceased_to_be_small":
    len (ceased_small ),
    "original_small_ceased_fraction":(
    len (ceased_small )/len (original_small )
    if original_small 
    else None 
    ),
    "neighbor_objects":neighbor_total ,
    "retained_neighbor_objects":retained_total ,
    "weighted_neighbor_retention_fraction":(
    retained_total /neighbor_total 
    if neighbor_total 
    else None 
    ),
    "crop_area_over_source_area":
    distribution ([
    float (row ["crop_area_over_source_area"])
    for row in rows 
    ]),
    "target_area_over_crop_area":
    distribution ([
    float (row ["target_area_over_crop_area"])
    for row in rows 
    ]),
    "short_side_amplification":
    distribution ([
    float (row ["short_side_amplification"])
    for row in rows 
    ]),
    "area_amplification":
    distribution ([
    float (row ["area_amplification"])
    for row in rows 
    ]),
    "neighbor_retention_fraction":
    distribution ([
    float (row ["neighbor_retention_fraction"])
    for row in rows 
    if row ["neighbor_retention_fraction"]!=""
    ]),
    "minimum_context_margin_over_target_short_side":
    distribution ([
    float (
    row [
    "minimum_context_margin_over_target_short_side"
    ]
    )
    for row in rows 
    ]),
    "objects_per_view_change":
    distribution ([
    float (row ["objects_per_view_change"])
    for row in rows 
    ]),
    }


def execute (
protocol :dict ,
materialization :dict ,
):
    started =time .time ()

    staging =OUTPUT .with_name (
    OUTPUT .name 
    +f".incomplete-{int (time .time ())}-{os .getpid ()}"
    )
    staging .mkdir (parents =True )

    (
    images ,
    annotations ,
    annotations_by_image ,
    category_names ,
    )=load_coco ()

    crop_rows =load_crop_rows ()

    excluded_annotation_id =int (
    materialization ["excluded_annotation_id"]
    )
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
    center_required =bool (
    materialization [
    "non_target_box_center_must_be_inside"
    ]
    )

    per_target_rows =[]
    transition_counts =Counter ()
    grouped_rows =defaultdict (list )
    class_rows =defaultdict (list )

    total_source_neighbors =0 
    total_retained_neighbors =0 
    total_fully_removed =0 
    total_boundary_clipped =0 
    total_supervised_retained =0 

    for index ,crop_row in enumerate (
    crop_rows ,
    start =1 ,
    ):
        image_id =int (crop_row ["image_id"])
        annotation_id =int (crop_row ["annotation_id"])

        image =images [image_id ]
        target =annotations [annotation_id ]

        image_width =float (image ["width"])
        image_height =float (image ["height"])
        source_area =image_width *image_height 

        crop_box =parse_crop_box (
        crop_row ["crop_box_xyxy"]
        )
        x0 ,y0 ,x1 ,y1 =crop_box 
        crop_width =x1 -x0 
        crop_height =y1 -y0 
        crop_area =crop_width *crop_height 

        source_annotations =[
        row 
        for row in annotations_by_image [image_id ]
        if valid_annotation (
        row ,
        excluded_annotation_id ,
        )
        ]

        target_matches =[
        row 
        for row in source_annotations 
        if int (row ["id"])==annotation_id 
        ]

        if len (target_matches )!=1 :
            raise AssertionError (
            (annotation_id ,len (target_matches ))
            )

        target =target_matches [0 ]
        tx ,ty ,tw ,th =map (
        float ,
        target ["bbox"],
        )

        (
        tix0 ,
        tiy0 ,
        tix1 ,
        tiy1 ,
        tiw ,
        tih ,
        target_crop_area ,
        )=intersection (
        target ["bbox"],
        crop_box ,
        )

        original_target_area =tw *th 
        target_visible_fraction =(
        target_crop_area /original_target_area 
        )

        if not math .isclose (
        target_visible_fraction ,
        1.0 ,
        rel_tol =0.0 ,
        abs_tol =1e-9 ,
        ):
            raise AssertionError ({
            "annotation_id":annotation_id ,
            "target_visible_fraction":
            target_visible_fraction ,
            })

        source_neighbor_count =0 
        retained_neighbor_count =0 
        fully_removed_objects =0 
        boundary_clipped_objects =0 
        supervised_retained_count =1 

        for neighbor in source_annotations :
            neighbor_id =int (neighbor ["id"])

            if neighbor_id ==annotation_id :
                continue 

            source_neighbor_count +=1 

            nx ,ny ,nw ,nh =map (
            float ,
            neighbor ["bbox"],
            )
            original_neighbor_area =nw *nh 

            (
            _ ,
            _ ,
            _ ,
            _ ,
            visible_width ,
            visible_height ,
            visible_area ,
            )=intersection (
            neighbor ["bbox"],
            crop_box ,
            )

            visible_fraction =(
            visible_area /original_neighbor_area 
            )

            if visible_fraction >=0.5 :
                retained_neighbor_count +=1 

            if visible_area <=0.0 :
                fully_removed_objects +=1 
            elif visible_fraction <1.0 -1e-12 :
                boundary_clipped_objects +=1 

            center_inside =(
            x0 <=nx +nw /2.0 <=x1 
            and y0 <=ny +nh /2.0 <=y1 
            )

            retained_by_materializer =(
            visible_fraction >=minimum_visible 
            and visible_width >=minimum_pixels 
            and visible_height >=minimum_pixels 
            and (
            center_inside 
            if center_required 
            else True 
            )
            )

            if retained_by_materializer :
                supervised_retained_count +=1 

        label_path =(
        RAM_ROOT /crop_row ["relative_label"]
        )
        actual_label_count =read_label_count (label_path )
        manifest_label_count =int (
        crop_row ["objects_in_view"]
        )

        if actual_label_count !=manifest_label_count :
            raise AssertionError ({
            "label_path":str (label_path ),
            "actual":actual_label_count ,
            "manifest":manifest_label_count ,
            })

        if (
        actual_label_count 
        !=supervised_retained_count 
        ):
            raise AssertionError ({
            "annotation_id":annotation_id ,
            "actual_label_count":
            actual_label_count ,
            "reconstructed_count":
            supervised_retained_count ,
            })

        source_object_count =len (source_annotations )

        original_area =original_target_area 
        crop_native_area =target_crop_area 

        original_short_side =min (tw ,th )
        crop_short_side =min (tiw ,tih )

        original_letterbox_scale =min (
        NETWORK_SIZE /image_width ,
        NETWORK_SIZE /image_height ,
        )
        crop_letterbox_scale =min (
        NETWORK_SIZE /crop_width ,
        NETWORK_SIZE /crop_height ,
        )

        original_network_short_side =(
        original_short_side 
        *original_letterbox_scale 
        )
        crop_network_short_side =(
        crop_short_side 
        *crop_letterbox_scale 
        )

        original_network_area =(
        original_area 
        *original_letterbox_scale **2 
        )
        crop_network_area =(
        crop_native_area 
        *crop_letterbox_scale **2 
        )

        original_scale =scale_name (original_area )
        crop_native_scale =scale_name (crop_native_area )
        original_network_scale =scale_name (
        original_network_area 
        )
        crop_network_scale =scale_name (
        crop_network_area 
        )

        short_side_amplification =(
        crop_network_short_side 
        /original_network_short_side 
        )
        area_amplification =(
        crop_network_area 
        /original_network_area 
        )

        left_margin =tix0 -x0 
        top_margin =tiy0 -y0 
        right_margin =x1 -tix1 
        bottom_margin =y1 -tiy1 
        minimum_margin =min (
        left_margin ,
        top_margin ,
        right_margin ,
        bottom_margin ,
        )

        neighbor_retention_fraction =(
        retained_neighbor_count 
        /source_neighbor_count 
        if source_neighbor_count 
        else None 
        )

        diagnostic_group =crop_row ["diagnostic_group"]
        category_id =int (target ["category_id"])

        result ={
        "slot_index":int (crop_row ["slot_index"]),
        "annotation_id":annotation_id ,
        "image_id":image_id ,
        "category_id":category_id ,
        "class_name":category_names [category_id ],
        "diagnostic_group":diagnostic_group ,
        "risk_score":float (crop_row ["risk_score"]),
        "source_width":image_width ,
        "source_height":image_height ,
        "crop_width":crop_width ,
        "crop_height":crop_height ,
        "crop_area_over_source_area":
        crop_area /source_area ,
        "original_bbox_short_side":
        original_short_side ,
        "original_bbox_area":
        original_area ,
        "original_normalized_bbox_area":
        original_area /source_area ,
        "crop_bbox_short_side":
        crop_short_side ,
        "crop_bbox_area":
        crop_native_area ,
        "crop_normalized_bbox_area":
        crop_native_area /crop_area ,
        "network_bbox_short_side":
        crop_network_short_side ,
        "network_bbox_area":
        crop_network_area ,
        "network_normalized_bbox_area":
        crop_network_area 
        /(NETWORK_SIZE **2 ),
        "full_scene_network_bbox_short_side":
        original_network_short_side ,
        "full_scene_network_bbox_area":
        original_network_area ,
        "short_side_amplification":
        short_side_amplification ,
        "area_amplification":
        area_amplification ,
        "original_scale":original_scale ,
        "crop_native_scale":crop_native_scale ,
        "full_scene_network_scale":
        original_network_scale ,
        "crop_network_scale":
        crop_network_scale ,
        "target_area_over_crop_area":
        crop_native_area /crop_area ,
        "source_object_count":
        source_object_count ,
        "crop_object_count":
        actual_label_count ,
        "source_neighbor_count":
        source_neighbor_count ,
        "retained_neighbor_count":
        retained_neighbor_count ,
        "neighbor_retention_fraction":(
        neighbor_retention_fraction 
        if neighbor_retention_fraction is not None 
        else ""
        ),
        "fully_removed_objects":
        fully_removed_objects ,
        "boundary_clipped_objects":
        boundary_clipped_objects ,
        "materializer_supervised_neighbors":
        supervised_retained_count -1 ,
        "objects_per_view_change":
        actual_label_count -source_object_count ,
        "objects_per_view_change_fraction":
        (
        actual_label_count 
        /source_object_count 
        -1.0 
        ),
        "left_context_margin_px":
        left_margin ,
        "top_context_margin_px":
        top_margin ,
        "right_context_margin_px":
        right_margin ,
        "bottom_context_margin_px":
        bottom_margin ,
        "minimum_context_margin_px":
        minimum_margin ,
        "minimum_context_margin_over_target_short_side":
        minimum_margin /original_short_side ,
        }

        per_target_rows .append (result )
        grouped_rows [diagnostic_group ].append (result )
        class_rows [category_id ].append (result )

        total_source_neighbors +=source_neighbor_count 
        total_retained_neighbors +=retained_neighbor_count 
        total_fully_removed +=fully_removed_objects 
        total_boundary_clipped +=boundary_clipped_objects 
        total_supervised_retained +=actual_label_count 

        transitions =(
        (
        "original_image",
        original_scale ,
        "crop_before_resize",
        crop_native_scale ,
        ),
        (
        "original_image",
        original_scale ,
        "network_input_after_960_letterbox",
        crop_network_scale ,
        ),
        (
        "full_scene_network_input_after_960_letterbox",
        original_network_scale ,
        "network_input_after_960_letterbox",
        crop_network_scale ,
        ),
        )

        for transition in transitions :
            transition_counts [transition ]+=1 

        if index %1000 ==0 :
            print (
            f"audited={index }/{EXPECTED_CROPS }",
            flush =True ,
            )

    if len (per_target_rows )!=EXPECTED_CROPS :
        raise AssertionError (len (per_target_rows ))

    transition_rows =[]

    transition_denominators =Counter ()

    for (
    source_space ,
    source_scale ,
    target_space ,
    target_scale ,
    ),count in transition_counts .items ():
        transition_denominators [
        (
        source_space ,
        source_scale ,
        target_space ,
        )
        ]+=count 

    for key ,count in sorted (
    transition_counts .items ()
    ):
        (
        source_space ,
        source_scale ,
        target_space ,
        target_scale ,
        )=key 

        denominator =transition_denominators [
        (
        source_space ,
        source_scale ,
        target_space ,
        )
        ]

        transition_rows .append ({
        "source_coordinate_space":source_space ,
        "source_scale":source_scale ,
        "target_coordinate_space":target_space ,
        "target_scale":target_scale ,
        "objects":int (count ),
        "fraction_within_source_scale":
        count /denominator ,
        })

    group_summary_rows =[]

    for group ,rows in sorted (grouped_rows .items ()):
        summary =summarize_group (rows )

        group_summary_rows .append ({
        "diagnostic_group":group ,
        "targets":summary ["targets"],
        "original_small_targets":
        summary ["original_small_targets"],
        "original_small_ceased_to_be_small":
        summary [
        "original_small_ceased_to_be_small"
        ],
        "original_small_ceased_fraction":
        summary [
        "original_small_ceased_fraction"
        ],
        "weighted_neighbor_retention_fraction":
        summary [
        "weighted_neighbor_retention_fraction"
        ],
        "mean_crop_area_over_source_area":
        summary [
        "crop_area_over_source_area"
        ]["mean"],
        "mean_short_side_amplification":
        summary [
        "short_side_amplification"
        ]["mean"],
        "mean_area_amplification":
        summary [
        "area_amplification"
        ]["mean"],
        "mean_objects_per_view_change":
        summary [
        "objects_per_view_change"
        ]["mean"],
        })

    class_summary_rows =[]

    for category_id ,rows in sorted (class_rows .items ()):
        summary =summarize_group (rows )

        class_summary_rows .append ({
        "category_id":category_id ,
        "class_name":category_names [category_id ],
        "targets":summary ["targets"],
        "original_small_targets":
        summary ["original_small_targets"],
        "original_small_ceased_to_be_small":
        summary [
        "original_small_ceased_to_be_small"
        ],
        "original_small_ceased_fraction":
        summary [
        "original_small_ceased_fraction"
        ],
        "weighted_neighbor_retention_fraction":
        summary [
        "weighted_neighbor_retention_fraction"
        ],
        "mean_short_side_amplification":
        summary [
        "short_side_amplification"
        ]["mean"],
        "mean_area_amplification":
        summary [
        "area_amplification"
        ]["mean"],
        })

    overall =summarize_group (per_target_rows )

    original_small_rows =[
    row 
    for row in per_target_rows 
    if row ["original_scale"]=="small"
    ]
    original_small_ceased =[
    row 
    for row in original_small_rows 
    if row ["crop_network_scale"]!="small"
    ]

    full_scene_network_small =[
    row 
    for row in per_target_rows 
    if row ["full_scene_network_scale"]=="small"
    ]
    full_scene_network_small_ceased =[
    row 
    for row in full_scene_network_small 
    if row ["crop_network_scale"]!="small"
    ]

    summary ={
    "version":"final_crop_scale_context_audit_v1",
    "status":"complete",
    "created_utc":time .strftime (
    "%Y-%m-%dT%H:%M:%SZ",
    time .gmtime (),
    ),
    "protocol_sha256":sha256_file (PROTOCOL ),
    "exposure_manifest_sha256":
    sha256_file (EXPOSURE_MANIFEST ),
    "targets":EXPECTED_CROPS ,
    "coordinate_spaces":{
    "original_image":(
    "Target bbox in the original BDD100K image."
    ),
    "crop_before_resize":(
    "Target bbox clipped to the materialized crop "
    "before any network resize."
    ),
    "network_input_after_960_letterbox":(
    "Target bbox after aspect-ratio-preserving "
    "resize into the 960x960 network canvas."
    ),
    "full_scene_network_input_after_960_letterbox":(
    "Auxiliary counterfactual coordinate: the same "
    "target after resizing the uncropped full scene "
    "into the 960x960 network canvas."
    ),
    },
    "scale_thresholds":{
    "small":"[0, 1024)",
    "medium":"[1024, 9216)",
    "large":"[9216, infinity)",
    },
    "primary_scale_question":{
    "original_small_targets":
    len (original_small_rows ),
    "original_small_ceased_to_be_small":
    len (original_small_ceased ),
    "fraction":
    (
    len (original_small_ceased )
    /len (original_small_rows )
    if original_small_rows 
    else None 
    ),
    },
    "auxiliary_full_scene_network_comparison":{
    "full_scene_network_small_targets":
    len (full_scene_network_small ),
    "ceased_to_be_small_after_crop":
    len (full_scene_network_small_ceased ),
    "fraction":
    (
    len (full_scene_network_small_ceased )
    /len (full_scene_network_small )
    if full_scene_network_small 
    else None 
    ),
    },
    "scale_shift":{
    "bbox_short_side":
    distribution ([
    row ["network_bbox_short_side"]
    for row in per_target_rows 
    ]),
    "bbox_area":
    distribution ([
    row ["network_bbox_area"]
    for row in per_target_rows 
    ]),
    "normalized_bbox_area":
    distribution ([
    row ["network_normalized_bbox_area"]
    for row in per_target_rows 
    ]),
    "short_side_amplification":
    overall [
    "short_side_amplification"
    ],
    "area_amplification":
    overall ["area_amplification"],
    },
    "context_shift":{
    "source_neighbors":
    total_source_neighbors ,
    "retained_neighbors":
    total_retained_neighbors ,
    "weighted_neighbor_retention_fraction":
    (
    total_retained_neighbors 
    /total_source_neighbors 
    if total_source_neighbors 
    else None 
    ),
    "fully_removed_objects":
    total_fully_removed ,
    "boundary_clipped_objects":
    total_boundary_clipped ,
    "crop_supervised_objects":
    total_supervised_retained ,
    "crop_area_over_source_area":
    overall [
    "crop_area_over_source_area"
    ],
    "target_area_over_crop_area":
    overall [
    "target_area_over_crop_area"
    ],
    "neighbor_retention_fraction":
    overall [
    "neighbor_retention_fraction"
    ],
    "objects_per_view_change":
    overall [
    "objects_per_view_change"
    ],
    "context_margin_around_target":
    overall [
    "minimum_context_margin_over_target_short_side"
    ],
    },
    "interpretation":(
    "This audit quantifies crop-induced scale and "
    "context shifts and their association with "
    "downstream non-transfer. It does not establish "
    "strict causality."
    ),
    "causal_language_allowed":False ,
    "new_training":False ,
    "new_inference":False ,
    "elapsed_minutes":
    (time .time ()-started )/60.0 ,
    }

    write_csv (
    staging /"per_target_scale_context.csv",
    per_target_rows ,
    )
    write_csv (
    staging /"scale_transition_matrix.csv",
    transition_rows ,
    )
    write_csv (
    staging /"diagnostic_group_summary.csv",
    group_summary_rows ,
    )
    write_csv (
    staging /"class_summary.csv",
    class_summary_rows ,
    )
    write_json (
    staging /"summary.json",
    summary ,
    )

    metadata ={
    "version":"final_crop_scale_context_audit_v1",
    "status":"complete",
    "script":str (Path (__file__ ).resolve ()),
    "script_sha256":
    sha256_file (Path (__file__ ).resolve ()),
    "neighbor_definition":(
    "Every valid non-target, non-crowd BDD100K GT "
    "object in the target source image, excluding the "
    "frozen duplicate annotation."
    ),
    "neighbor_retained_rule":(
    "At least 50% of the original neighbor bbox area "
    "remains inside the crop before resize."
    ),
    "materializer_reconstruction_verified":
    True ,
    "target_boxes_fully_retained":
    True ,
    "causal_interpretation_forbidden":
    True ,
    }

    write_json (
    staging /"metadata.json",
    metadata ,
    )

    manifest_path ,manifest_sha =build_manifest (
    staging 
    )

    os .replace (staging ,OUTPUT )

    print ()
    print ("="*100 )
    print ("FINAL CROP SCALE / CONTEXT AUDIT COMPLETE")
    print ("="*100 )
    print (
    "targets:",
    EXPECTED_CROPS ,
    )
    print (
    "original_small:",
    len (original_small_rows ),
    "ceased_small:",
    len (original_small_ceased ),
    "fraction:",
    summary [
    "primary_scale_question"
    ]["fraction"],
    )
    print (
    "neighbor_retention:",
    summary ["context_shift"][
    "weighted_neighbor_retention_fraction"
    ],
    )
    print (
    "fully_removed_objects:",
    total_fully_removed ,
    )
    print (
    "boundary_clipped_objects:",
    total_boundary_clipped ,
    )
    print (
    "mean_short_side_amplification:",
    summary ["scale_shift"][
    "short_side_amplification"
    ]["mean"],
    )
    print ("output:",OUTPUT )
    print (
    "manifest:",
    OUTPUT /manifest_path .name ,
    )
    print ("manifest_sha256:",manifest_sha )
    print ("NOTHING TRAINED OR INFERRED")
    print (
    "NEXT: representative-checkpoint "
    "corruption evaluation"
    )
    print ("="*100 )


def main ():
    args =parse_args ()
    protocol ,materialization =preflight ()

    if args .preflight_only :
        return 

    execute (protocol ,materialization )


if __name__ =="__main__":
    main ()
