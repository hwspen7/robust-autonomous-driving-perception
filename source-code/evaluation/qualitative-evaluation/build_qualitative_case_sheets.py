from __future__ import annotations 

import argparse 
import csv 
import gc 
import hashlib 
import json 
import os 
import shutil 
import time 
from collections import defaultdict 
from datetime import datetime ,timezone 
from pathlib import Path 

import numpy as np 
import torch 
from PIL import Image ,ImageDraw ,ImageFont 
from ultralytics import YOLO 

from controlled_corruptions import apply_corruption 


AUTODRIVE =Path (
"/root/rivermind-data/autodrive"
)
PROJECT =(
AUTODRIVE /"code/"
"robust-autonomous-driving-perception"
)
RESULTS =(
AUTODRIVE /"results/evaluation"
)
FINAL =RESULTS /"final_analysis_v1"
DATASET =(
AUTODRIVE /"datasets/datasets/"
"bdd100k_final"
)

PROTOCOL =(
FINAL /"qualitative_protocol_v1/"
"qualitative_protocol_v1.json"
)
PROTOCOL_MANIFEST =(
FINAL /"qualitative_protocol_v1/"
"artifact_manifest.json"
)

EXPECTED_PROTOCOL_SHA256 =(
"037fbd5e35b27ff9d62e7c41716834a6"
"ee45a70465d41200d66b40c6cdc52b88"
)
EXPECTED_PROTOCOL_MANIFEST_SHA256 =(
"fc95a64af63ab6daaa7c49b63dd203bd"
"dcd9babb09658d6c946d7f944fd79529"
)

OUTPUT =FINAL /"qualitative_cases_v1"
WORK =FINAL /(
"qualitative_cases_v1.incomplete-"
"037fbd5e35b2"
)

VAL_GT =(
DATASET /"coco/annotations/"
"instances_val.json"
)
TRAIN_GT =(
DATASET /"coco/annotations/"
"instances_train.json"
)
FAILURE_SUBSET =(
FINAL /"official_val_failure_subsets_v1/"
"subsets/small_jointly_missed.json"
)
SCALE_CONTEXT_CSV =(
FINAL /"crop_scale_context_audit_v1/"
"per_target_scale_context.csv"
)

SEED =20260816 
CASES_PER_FAMILY =4 
SCORE_THRESHOLD =0.25 
MATCH_IOU =0.50 
TOP_K =100 
PROBE_IMAGES =512 
CORRUPTIONS =("blur","noise")
CORRUPTION_SEVERITY =2 

MODEL_KEYS =(
"baseline_yolo",
"p2_e20",
"uniform_e56",
"risk_e56",
)

CORRUPTION_CHECKPOINTS ={
"standard_e20":(
RESULTS /"architecture_gate/formal_gate_v4/"
"standard/weights/resume_epoch20.pt"
),
"p2_e20":(
RESULTS /"architecture_gate/formal_gate_v4/"
"p2_projected/weights/resume_epoch20.pt"
),
}

FAMILIES =(
"p2_repairs_small_failure",
"p2_remaining_failure",
"risk_short_term_recovery",
"risk_non_transfer_in_original_scene",
"crop_scale_condition_shift",
"crop_context_condition_shift",
"blur_or_noise_success",
"blur_or_noise_failure",
)

CATEGORY_NAMES ={
1 :"pedestrian",
2 :"rider",
3 :"car",
4 :"truck",
5 :"bus",
6 :"train",
7 :"motorcycle",
8 :"bicycle",
9 :"traffic light",
10 :"traffic sign",
}

PANEL_WIDTH =480 
PANEL_HEIGHT =270 
ROW_TEXT_HEIGHT =70 


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


def sha256 (path :Path )->str :
    digest =hashlib .sha256 ()

    with path .open ("rb")as handle :
        while True :
            block =handle .read (
            8 *1024 *1024 
            )
            if not block :
                break 
            digest .update (block )

    return digest .hexdigest ()


def write_json (path :Path ,payload )->None :
    temporary =path .with_suffix (
    path .suffix +".tmp"
    )
    temporary .write_text (
    json .dumps (
    payload ,
    indent =2 ,
    ensure_ascii =False ,
    sort_keys =True ,
    )+"\n",
    encoding ="utf-8",
    )
    os .replace (temporary ,path )


def verify (path :Path ,expected :str )->None :
    if not path .is_file ():
        raise FileNotFoundError (path )

    actual =sha256 (path )

    print (path )
    print (" expected:",expected )
    print (" actual  :",actual )

    if actual !=expected :
        raise AssertionError (path )


def deterministic_rank (
family :str ,
annotation_id :int ,
variant :str ="clean",
)->str :
    text =(
    f"{SEED }|{family }|"
    f"{annotation_id }|{variant }"
    )
    return hashlib .sha256 (
    text .encode ("utf-8")
    ).hexdigest ()


def xywh_to_xyxy (bbox ):
    x ,y ,width ,height =map (
    float ,
    bbox ,
    )
    return [
    x ,
    y ,
    x +width ,
    y +height ,
    ]


def bbox_iou (box_a ,box_b )->float :
    ax0 ,ay0 ,ax1 ,ay1 =box_a 
    bx0 ,by0 ,bx1 ,by1 =box_b 

    ix0 =max (ax0 ,bx0 )
    iy0 =max (ay0 ,by0 )
    ix1 =min (ax1 ,bx1 )
    iy1 =min (ay1 ,by1 )

    iw =max (0.0 ,ix1 -ix0 )
    ih =max (0.0 ,iy1 -iy0 )
    intersection =iw *ih 

    area_a =max (
    0.0 ,
    ax1 -ax0 ,
    )*max (
    0.0 ,
    ay1 -ay0 ,
    )
    area_b =max (
    0.0 ,
    bx1 -bx0 ,
    )*max (
    0.0 ,
    by1 -by0 ,
    )

    union =area_a +area_b -intersection 

    return (
    intersection /union 
    if union >0.0 
    else 0.0 
    )


def detects_target (
annotation :dict ,
predictions :list [dict ],
)->bool :
    target_box =xywh_to_xyxy (
    annotation ["bbox"]
    )
    category_id =int (
    annotation ["category_id"]
    )

    for prediction in predictions :
        if (
        int (prediction ["category_id"])
        !=category_id 
        ):
            continue 

        if float (prediction ["score"])<(
        SCORE_THRESHOLD 
        ):
            continue 

        if bbox_iou (
        target_box ,
        prediction ["bbox_xyxy"],
        )>=MATCH_IOU :
            return True 

    return False 


def load_prediction_index (
key :str ,
record :dict ,
)->dict [int ,list [dict ]]:
    path =Path (record ["path"])
    expected =record ["sha256"]

    verify (path ,expected )

    print ("\nLOADING PREDICTIONS:",key )

    raw =json .loads (
    path .read_text ()
    )

    grouped =defaultdict (list )

    for prediction in raw :
        score =float (
        prediction ["score"]
        )

        if score <SCORE_THRESHOLD :
            continue 

        grouped [
        int (prediction ["image_id"])
        ].append ({
        "category_id":int (
        prediction ["category_id"]
        ),
        "score":score ,
        "bbox_xyxy":xywh_to_xyxy (
        prediction ["bbox"]
        ),
        })

    del raw 

    for image_id ,predictions in grouped .items ():
        predictions .sort (
        key =lambda item :(
        -item ["score"],
        item ["category_id"],
        item ["bbox_xyxy"],
        )
        )
        grouped [image_id ]=(
        predictions [:TOP_K ]
        )

    print (
    "images_with_display_predictions:",
    len (grouped ),
    )

    return dict (grouped )


def choose_diverse (
family :str ,
candidates :list [dict ],
)->list [dict ]:
    ordered =sorted (
    candidates ,
    key =lambda row :(
    row ["selection_rank"],
    row ["annotation_id"],
    ),
    )

    selected =[]
    used_images =set ()
    used_categories =set ()


    for row in ordered :
        if len (selected )>=CASES_PER_FAMILY :
            break 

        image_id =int (row ["image_id"])
        category_id =int (row ["category_id"])

        if image_id in used_images :
            continue 

        if category_id in used_categories :
            continue 

        selected .append (row )
        used_images .add (image_id )
        used_categories .add (category_id )


    for row in ordered :
        if len (selected )>=CASES_PER_FAMILY :
            break 

        image_id =int (row ["image_id"])

        if image_id in used_images :
            continue 

        selected .append (row )
        used_images .add (image_id )

    if len (selected )!=CASES_PER_FAMILY :
        raise RuntimeError ({
        "family":family ,
        "eligible_candidates":
        len (candidates ),
        "selected":len (selected ),
        "required":CASES_PER_FAMILY ,
        })

    for index ,row in enumerate (
    selected ,
    start =1 ,
    ):
        row ["case_index"]=index 

    return selected 


def model_result_predictions (result ):
    if (
    result .boxes is None 
    or len (result .boxes )==0 
    ):
        return []

    xyxy =(
    result .boxes .xyxy 
    .detach ()
    .cpu ()
    .numpy ()
    )
    scores =(
    result .boxes .conf 
    .detach ()
    .cpu ()
    .numpy ()
    )
    classes =(
    result .boxes .cls 
    .detach ()
    .cpu ()
    .numpy ()
    )

    predictions =[]

    for box ,score ,class_id in zip (
    xyxy ,
    scores ,
    classes ,
    ):
        score =float (score )

        if score <SCORE_THRESHOLD :
            continue 

        predictions .append ({
        "category_id":
        int (class_id )+1 ,
        "score":score ,
        "bbox_xyxy":[
        float (value )
        for value in box .tolist ()
        ],
        })

    predictions .sort (
    key =lambda item :(
    -item ["score"],
    item ["category_id"],
    item ["bbox_xyxy"],
    )
    )

    return predictions [:TOP_K ]


def build_corruption_cases (
*,
annotations :dict [int ,dict ],
images :dict [int ,dict ],
subset_annotation_ids :list [int ],
clean_prediction_indexes :dict ,
)->tuple [dict [str ,list [dict ]],dict ]:
    probe_candidates =[]

    for annotation_id in subset_annotation_ids :
        annotation =annotations [
        annotation_id 
        ]

        probe_candidates .append ({
        "annotation_id":
        annotation_id ,
        "image_id":
        int (annotation ["image_id"]),
        "category_id":
        int (annotation ["category_id"]),
        "selection_rank":
        deterministic_rank (
        "corruption_probe",
        annotation_id ,
        ),
        })

    probe_candidates .sort (
    key =lambda row :(
    row ["selection_rank"],
    row ["annotation_id"],
    )
    )

    probe_targets =[]
    used_images =set ()

    for row in probe_candidates :
        image_id =int (row ["image_id"])

        if image_id in used_images :
            continue 

        probe_targets .append (row )
        used_images .add (image_id )

        if len (probe_targets )==PROBE_IMAGES :
            break 

    if len (probe_targets )!=PROBE_IMAGES :
        raise RuntimeError (
        len (probe_targets )
        )

    probe_records =[]

    for row in probe_targets :
        for corruption in CORRUPTIONS :
            variant =(
            f"{corruption }_"
            f"s{CORRUPTION_SEVERITY }"
            )

            probe_records .append ({
            **row ,
            "corruption":corruption ,
            "severity":
            CORRUPTION_SEVERITY ,
            "variant":variant ,
            })

    print ("\n=== CORRUPTION PROBE ===")
    print ("probe_images:",len (probe_targets ))
    print ("probe_views:",len (probe_records ))

    standard_model =YOLO (
    str (
    CORRUPTION_CHECKPOINTS [
    "standard_e20"
    ]
    )
    )
    p2_model =YOLO (
    str (
    CORRUPTION_CHECKPOINTS [
    "p2_e20"
    ]
    )
    )

    batch_size =8 

    for start in range (
    0 ,
    len (probe_records ),
    batch_size ,
    ):
        batch_records =probe_records [
        start :start +batch_size 
        ]
        corrupted_images =[]

        for row in batch_records :
            image_info =images [
            int (row ["image_id"])
            ]
            image_path =(
            DATASET 
            /image_info ["file_name"]
            )

            with Image .open (
            image_path 
            )as image :
                corrupted =apply_corruption (
                image .convert ("RGB"),
                image_key =(
                image_info ["file_name"]
                ),
                corruption =row ["corruption"],
                severity =row ["severity"],
                )

            corrupted_images .append (
            corrupted 
            )

        standard_results =(
        standard_model .predict (
        source =corrupted_images ,
        imgsz =960 ,
        batch =batch_size ,
        conf =0.001 ,
        iou =0.7 ,
        max_det =300 ,
        device =0 ,
        half =True ,
        verbose =False ,
        stream =False ,
        )
        )

        p2_results =p2_model .predict (
        source =corrupted_images ,
        imgsz =960 ,
        batch =batch_size ,
        conf =0.001 ,
        iou =0.7 ,
        max_det =300 ,
        device =0 ,
        half =True ,
        verbose =False ,
        stream =False ,
        )

        if (
        len (standard_results )
        !=len (batch_records )
        or len (p2_results )
        !=len (batch_records )
        ):
            raise AssertionError (start )

        for (
        row ,
        standard_result ,
        p2_result ,
        )in zip (
        batch_records ,
        standard_results ,
        p2_results ,
        ):
            row ["standard_corrupted_predictions"]=(
            model_result_predictions (
            standard_result 
            )
            )
            row ["p2_corrupted_predictions"]=(
            model_result_predictions (
            p2_result 
            )
            )

            annotation =annotations [
            int (row ["annotation_id"])
            ]

            row ["standard_corrupted_detected"]=(
            detects_target (
            annotation ,
            row [
            "standard_corrupted_predictions"
            ],
            )
            )
            row ["p2_corrupted_detected"]=(
            detects_target (
            annotation ,
            row [
            "p2_corrupted_predictions"
            ],
            )
            )

        if (
        start ==0 
        or (
        start +len (batch_records )
        )%128 ==0 
        ):
            print (
            "corruption_probe=",
            min (
            start +len (batch_records ),
            len (probe_records ),
            ),
            "/",
            len (probe_records ),
            )

        for image in corrupted_images :
            image .close ()

        del standard_results 
        del p2_results 
        del corrupted_images 

    del standard_model 
    del p2_model 
    gc .collect ()
    torch .cuda .empty_cache ()

    success_candidates =[]
    failure_candidates =[]

    for row in probe_records :
        annotation_id =int (
        row ["annotation_id"]
        )
        image_id =int (row ["image_id"])
        category_id =int (
        row ["category_id"]
        )

        base ={
        **row ,
        "class_name":
        CATEGORY_NAMES [
        category_id 
        ],
        "selection_rank":
        deterministic_rank (
        "blur_or_noise_success",
        annotation_id ,
        row ["variant"],
        ),
        "clean_statuses":{
        key :detects_target (
        annotations [annotation_id ],
        clean_prediction_indexes [
        key 
        ].get (image_id ,[]),
        )
        for key in MODEL_KEYS 
        },
        }

        if (
        row ["p2_corrupted_detected"]
        and not row [
        "standard_corrupted_detected"
        ]
        ):
            success_row =dict (base )
            success_row ["family"]=(
            "blur_or_noise_success"
            )
            success_candidates .append (
            success_row 
            )

        if (
        not row ["p2_corrupted_detected"]
        and not row [
        "standard_corrupted_detected"
        ]
        ):
            failure_row =dict (base )
            failure_row ["family"]=(
            "blur_or_noise_failure"
            )
            failure_row ["selection_rank"]=(
            deterministic_rank (
            "blur_or_noise_failure",
            annotation_id ,
            row ["variant"],
            )
            )
            failure_candidates .append (
            failure_row 
            )

    selected ={
    "blur_or_noise_success":
    choose_diverse (
    "blur_or_noise_success",
    success_candidates ,
    ),
    "blur_or_noise_failure":
    choose_diverse (
    "blur_or_noise_failure",
    failure_candidates ,
    ),
    }

    summary ={
    "probe_images":len (probe_targets ),
    "probe_views":len (probe_records ),
    "success_candidates":
    len (success_candidates ),
    "failure_candidates":
    len (failure_candidates ),
    "corruptions":
    list (CORRUPTIONS ),
    "severity":
    CORRUPTION_SEVERITY ,
    "new_training":False ,
    "temporary_prediction_files":0 ,
    }

    return selected ,summary 


def load_font (size :int ):
    candidates =[
    Path (
    "/usr/share/fonts/truetype/"
    "dejavu/DejaVuSans.ttf"
    ),
    Path (
    "/usr/share/fonts/dejavu/"
    "DejaVuSans.ttf"
    ),
    ]

    for path in candidates :
        if path .is_file ():
            return ImageFont .truetype (
            str (path ),
            size =size ,
            )

    return ImageFont .load_default ()


FONT_TITLE =load_font (22 )
FONT_TEXT =load_font (16 )
FONT_SMALL =load_font (13 )


def scaled_box (
box ,
source_size ,
target_size ,
):
    source_width ,source_height =(
    source_size 
    )
    target_width ,target_height =(
    target_size 
    )

    sx =target_width /source_width 
    sy =target_height /source_height 

    x0 ,y0 ,x1 ,y1 =box 

    return [
    x0 *sx ,
    y0 *sy ,
    x1 *sx ,
    y1 *sy ,
    ]


def make_detection_panel (
*,
image :Image .Image ,
annotation :dict ,
predictions :list [dict ],
title :str ,
)->Image .Image :
    panel =image .convert ("RGB").resize (
    (PANEL_WIDTH ,PANEL_HEIGHT ),
    Image .Resampling .LANCZOS ,
    )

    draw =ImageDraw .Draw (panel )
    source_size =image .size 
    target_box =xywh_to_xyxy (
    annotation ["bbox"]
    )

    scaled_target =scaled_box (
    target_box ,
    source_size ,
    panel .size ,
    )

    draw .rectangle (
    scaled_target ,
    outline ="yellow",
    width =4 ,
    )

    category_id =int (
    annotation ["category_id"]
    )

    for prediction in predictions :
        if (
        int (prediction ["category_id"])
        !=category_id 
        ):
            continue 

        prediction_box =prediction [
        "bbox_xyxy"
        ]
        matched =(
        bbox_iou (
        target_box ,
        prediction_box ,
        )>=MATCH_IOU 
        )

        color =(
        "lime"
        if matched 
        else "red"
        )

        scaled_prediction =scaled_box (
        prediction_box ,
        source_size ,
        panel .size ,
        )

        draw .rectangle (
        scaled_prediction ,
        outline =color ,
        width =3 if matched else 2 ,
        )

        label =(
        f"{prediction ['score']:.2f}"
        )
        draw .text (
        (
        scaled_prediction [0 ]+2 ,
        max (
        2 ,
        scaled_prediction [1 ]-16 ,
        ),
        ),
        label ,
        fill =color ,
        font =FONT_SMALL ,
        stroke_width =2 ,
        stroke_fill ="black",
        )

    detected =detects_target (
    annotation ,
    predictions ,
    )

    draw .rectangle (
    (0 ,0 ,PANEL_WIDTH ,30 ),
    fill =(0 ,0 ,0 ),
    )
    draw .text (
    (8 ,5 ),
    (
    title 
    if title =="Ground Truth"
    else (
    f"{title } | "
    f"{'DETECTED'if detected else 'MISSED'}"
    )
    ),
    fill ="white",
    font =FONT_TEXT ,
    )

    return panel 


def make_gt_panel (
image :Image .Image ,
annotation :dict ,
)->Image .Image :
    return make_detection_panel (
    image =image ,
    annotation =annotation ,
    predictions =[],
    title ="Ground Truth",
    )


def make_text_panel (
title :str ,
lines :list [str ],
width :int =600 ,
height :int =338 ,
)->Image .Image :
    panel =Image .new (
    "RGB",
    (width ,height ),
    "white",
    )
    draw =ImageDraw .Draw (panel )

    draw .rectangle (
    (0 ,0 ,width ,42 ),
    fill =(30 ,30 ,30 ),
    )
    draw .text (
    (12 ,8 ),
    title ,
    fill ="white",
    font =FONT_TITLE ,
    )

    y =58 

    for line in lines :
        draw .text (
        (14 ,y ),
        line ,
        fill ="black",
        font =FONT_TEXT ,
        )
        y +=29 

    return panel 


def add_row_caption (
row_image :Image .Image ,
text :str ,
)->Image .Image :
    output =Image .new (
    "RGB",
    (
    row_image .width ,
    row_image .height 
    +ROW_TEXT_HEIGHT ,
    ),
    "white",
    )
    output .paste (
    row_image ,
    (0 ,0 ),
    )

    draw =ImageDraw .Draw (output )
    draw .text (
    (10 ,row_image .height +8 ),
    text ,
    fill ="black",
    font =FONT_TEXT ,
    )

    return output 


def render_clean_case (
case :dict ,
annotation :dict ,
image_info :dict ,
prediction_indexes :dict ,
)->Image .Image :
    image_path =(
    DATASET /image_info ["file_name"]
    )

    with Image .open (image_path )as source :
        image =source .convert ("RGB")

    image_id =int (case ["image_id"])

    panels =[
    make_gt_panel (
    image ,
    annotation ,
    ),
    make_detection_panel (
    image =image ,
    annotation =annotation ,
    predictions =(
    prediction_indexes [
    "baseline_yolo"
    ].get (image_id ,[])
    ),
    title ="Baseline YOLO",
    ),
    make_detection_panel (
    image =image ,
    annotation =annotation ,
    predictions =(
    prediction_indexes [
    "p2_e20"
    ].get (image_id ,[])
    ),
    title ="P2-E20",
    ),
    make_detection_panel (
    image =image ,
    annotation =annotation ,
    predictions =(
    prediction_indexes [
    "uniform_e56"
    ].get (image_id ,[])
    ),
    title ="Uniform-E56",
    ),
    make_detection_panel (
    image =image ,
    annotation =annotation ,
    predictions =(
    prediction_indexes [
    "risk_e56"
    ].get (image_id ,[])
    ),
    title ="Risk-E56",
    ),
    ]

    row =Image .new (
    "RGB",
    (
    PANEL_WIDTH *len (panels ),
    PANEL_HEIGHT ,
    ),
    "white",
    )

    for index ,panel in enumerate (panels ):
        row .paste (
        panel ,
        (index *PANEL_WIDTH ,0 ),
        )

    caption =(
    f"family={case ['family']} | "
    f"annotation={case ['annotation_id']} | "
    f"class={case ['class_name']} | "
    f"image={case ['image_id']} | "
    f"score threshold={SCORE_THRESHOLD }"
    )

    return add_row_caption (
    row ,
    caption ,
    )


def reconstruct_crop (
row :dict ,
annotation :dict ,
image :Image .Image ,
):
    tx ,ty ,tw ,th =map (
    float ,
    annotation ["bbox"],
    )

    left =float (
    row ["left_context_margin_px"]
    )
    top =float (
    row ["top_context_margin_px"]
    )
    crop_width =float (
    row ["crop_width"]
    )
    crop_height =float (
    row ["crop_height"]
    )

    x0 =tx -left 
    y0 =ty -top 
    x1 =x0 +crop_width 
    y1 =y0 +crop_height 

    width ,height =image .size 

    x0 =max (0.0 ,min (x0 ,width ))
    y0 =max (0.0 ,min (y0 ,height ))
    x1 =max (x0 ,min (x1 ,width ))
    y1 =max (y0 ,min (y1 ,height ))

    crop_box =(
    int (round (x0 )),
    int (round (y0 )),
    int (round (x1 )),
    int (round (y1 )),
    )

    crop =image .crop (crop_box )

    target_crop_bbox =[
    tx -crop_box [0 ],
    ty -crop_box [1 ],
    tw ,
    th ,
    ]

    return crop ,crop_box ,target_crop_bbox 


def render_crop_case (
case :dict ,
annotation :dict ,
image_info :dict ,
)->Image .Image :
    image_path =(
    DATASET /image_info ["file_name"]
    )

    with Image .open (image_path )as source :
        image =source .convert ("RGB")

    row_data =case ["scale_context"]
    crop ,crop_box ,crop_target =(
    reconstruct_crop (
    row_data ,
    annotation ,
    image ,
    )
    )

    original_panel =image .resize (
    (600 ,338 ),
    Image .Resampling .LANCZOS ,
    )
    draw =ImageDraw .Draw (original_panel )

    target_scaled =scaled_box (
    xywh_to_xyxy (annotation ["bbox"]),
    image .size ,
    original_panel .size ,
    )
    crop_scaled =scaled_box (
    crop_box ,
    image .size ,
    original_panel .size ,
    )

    draw .rectangle (
    target_scaled ,
    outline ="yellow",
    width =4 ,
    )
    draw .rectangle (
    crop_scaled ,
    outline ="cyan",
    width =4 ,
    )
    draw .rectangle (
    (0 ,0 ,600 ,34 ),
    fill ="black",
    )
    draw .text (
    (8 ,6 ),
    "Original scene: GT + crop boundary",
    fill ="white",
    font =FONT_TEXT ,
    )

    crop_annotation ={
    "bbox":crop_target ,
    "category_id":
    annotation ["category_id"],
    }

    crop_panel =crop .resize (
    (600 ,338 ),
    Image .Resampling .LANCZOS ,
    )
    crop_draw =ImageDraw .Draw (crop_panel )

    crop_target_scaled =scaled_box (
    xywh_to_xyxy (crop_target ),
    crop .size ,
    crop_panel .size ,
    )
    crop_draw .rectangle (
    crop_target_scaled ,
    outline ="yellow",
    width =4 ,
    )
    crop_draw .rectangle (
    (0 ,0 ,600 ,34 ),
    fill ="black",
    )
    crop_draw .text (
    (8 ,6 ),
    "Reconstructed object-centric crop",
    fill ="white",
    font =FONT_TEXT ,
    )

    stat_panel =make_text_panel (
    "Frozen scale/context audit",
    [
    f"class: {case ['class_name']}",
    (
    "full scene -> crop scale: "
    f"{row_data ['full_scene_network_scale']} "
    f"-> {row_data ['crop_network_scale']}"
    ),
    (
    "short-side amplification: "
    f"{float (row_data ['short_side_amplification']):.3f}x"
    ),
    (
    "area amplification: "
    f"{float (row_data ['area_amplification']):.3f}x"
    ),
    (
    "neighbor retention: "
    f"{float (row_data ['neighbor_retention_fraction']or 0 ):.3f}"
    ),
    (
    "fully removed objects: "
    f"{row_data ['fully_removed_objects']}"
    ),
    (
    "objects/view change: "
    f"{row_data ['objects_per_view_change']}"
    ),
    (
    "crop/source area: "
    f"{float (row_data ['crop_area_over_source_area']):.3f}"
    ),
    ],
    )

    combined =Image .new (
    "RGB",
    (1800 ,338 ),
    "white",
    )
    combined .paste (
    original_panel ,
    (0 ,0 ),
    )
    combined .paste (
    crop_panel ,
    (600 ,0 ),
    )
    combined .paste (
    stat_panel ,
    (1200 ,0 ),
    )

    caption =(
    f"family={case ['family']} | "
    f"annotation={case ['annotation_id']} | "
    f"image={case ['image_id']} | "
    "crop reconstructed from frozen geometry"
    )

    return add_row_caption (
    combined ,
    caption ,
    )


def render_corruption_case (
case :dict ,
annotation :dict ,
image_info :dict ,
prediction_indexes :dict ,
)->Image .Image :
    image_path =(
    DATASET /image_info ["file_name"]
    )

    with Image .open (image_path )as source :
        image =source .convert ("RGB")
        corrupted =apply_corruption (
        image ,
        image_key =image_info ["file_name"],
        corruption =case ["corruption"],
        severity =int (case ["severity"]),
        )

    image_id =int (case ["image_id"])

    panels =[
    make_gt_panel (
    image ,
    annotation ,
    ),
    make_detection_panel (
    image =corrupted ,
    annotation =annotation ,
    predictions =case [
    "standard_corrupted_predictions"
    ],
    title =(
    "Standard-E20 "
    f"{case ['variant']}"
    ),
    ),
    make_detection_panel (
    image =corrupted ,
    annotation =annotation ,
    predictions =case [
    "p2_corrupted_predictions"
    ],
    title =(
    "P2-E20 "
    f"{case ['variant']}"
    ),
    ),
    make_detection_panel (
    image =image ,
    annotation =annotation ,
    predictions =(
    prediction_indexes [
    "uniform_e56"
    ].get (image_id ,[])
    ),
    title ="Uniform-E56 clean",
    ),
    make_detection_panel (
    image =image ,
    annotation =annotation ,
    predictions =(
    prediction_indexes [
    "risk_e56"
    ].get (image_id ,[])
    ),
    title ="Risk-E56 clean",
    ),
    ]

    combined =Image .new (
    "RGB",
    (
    PANEL_WIDTH *len (panels ),
    PANEL_HEIGHT ,
    ),
    "white",
    )

    for index ,panel in enumerate (panels ):
        combined .paste (
        panel ,
        (index *PANEL_WIDTH ,0 ),
        )

    caption =(
    f"family={case ['family']} | "
    f"variant={case ['variant']} | "
    f"annotation={case ['annotation_id']} | "
    f"class={case ['class_name']} | "
    f"standard={'detected'if case ['standard_corrupted_detected']else 'missed'} | "
    f"p2={'detected'if case ['p2_corrupted_detected']else 'missed'}"
    )

    return add_row_caption (
    combined ,
    caption ,
    )


def save_family_sheet (
family :str ,
rows :list [Image .Image ],
output_path :Path ,
)->None :
    width =max (
    row .width 
    for row in rows 
    )
    header_height =58 
    height =(
    header_height 
    +sum (row .height for row in rows )
    )

    sheet =Image .new (
    "RGB",
    (width ,height ),
    "white",
    )
    draw =ImageDraw .Draw (sheet )

    draw .rectangle (
    (0 ,0 ,width ,header_height ),
    fill =(25 ,25 ,25 ),
    )
    draw .text (
    (16 ,14 ),
    family ,
    fill ="white",
    font =FONT_TITLE ,
    )

    y =header_height 

    for row in rows :
        sheet .paste (
        row ,
        (0 ,y ),
        )
        y +=row .height 

    sheet .save (
    output_path ,
    format ="PNG",
    optimize =True ,
    )


def build_manifest (directory :Path ):
    artifacts =[]

    for path in sorted (
    directory .rglob ("*")
    ):
        if (
        not path .is_file ()
        or path .name 
        =="artifact_manifest.json"
        ):
            continue 

        artifacts .append ({
        "path":str (
        path .relative_to (directory )
        ),
        "bytes":path .stat ().st_size ,
        "sha256":sha256 (path ),
        })

    manifest ={
    "version":
    "final_qualitative_cases_manifest_v1",
    "status":"complete",
    "created_utc":
    datetime .now (
    timezone .utc 
    ).isoformat (),
    "artifacts":artifacts ,
    }

    path =(
    directory 
    /"artifact_manifest.json"
    )
    write_json (
    path ,
    manifest ,
    )

    return path ,sha256 (path )


def preflight ():
    print ("=== FINAL QUALITATIVE PREFLIGHT ===")

    verify (
    PROTOCOL ,
    EXPECTED_PROTOCOL_SHA256 ,
    )
    verify (
    PROTOCOL_MANIFEST ,
    EXPECTED_PROTOCOL_MANIFEST_SHA256 ,
    )

    protocol =json .loads (
    PROTOCOL .read_text ()
    )

    assert protocol ["status"]==(
    "frozen_before_case_selection"
    )
    assert protocol ["seed"]==SEED 
    assert protocol ["cases_per_family"]==(
    CASES_PER_FAMILY 
    )
    assert tuple (
    protocol ["case_families"]
    )==FAMILIES 
    assert protocol ["total_cases"]==32 

    for group_name in (
    "inputs",
    "predictions",
    "checkpoints",
    ):
        for record in protocol [
        group_name 
        ].values ():
            verify (
            Path (record ["path"]),
            record ["sha256"],
            )

    if OUTPUT .exists ():
        raise FileExistsError (OUTPUT )

    if WORK .exists ():
        raise FileExistsError (WORK )

    free_gib =(
    shutil .disk_usage (
    "/root/rivermind-data"
    ).free 
    /1024 **3 
    )

    if free_gib <1.0 :
        raise RuntimeError (
        f"Insufficient storage: "
        f"{free_gib :.3f} GiB"
        )

    if not torch .cuda .is_available ():
        raise RuntimeError (
        "CUDA unavailable"
        )

    print ("free_GiB:",round (free_gib ,3 ))
    print (
    "GPU:",
    torch .cuda .get_device_name (0 ),
    )
    print ("families:",len (FAMILIES ))
    print (
    "cases_per_family:",
    CASES_PER_FAMILY ,
    )
    print ("total_cases:",32 )
    print ("corruption_probe_images:",PROBE_IMAGES )
    print (
    "PASS: qualitative inputs and "
    "selection rules are frozen"
    )
    print (
    "NOTHING SELECTED, RENDERED, "
    "INFERRED OR TRAINED"
    )

    return protocol 


def execute (protocol :dict ):
    WORK .mkdir (
    parents =True ,
    exist_ok =False ,
    )
    figure_dir =WORK /"figures"
    figure_dir .mkdir ()

    started =time .time ()

    val_data =json .loads (
    VAL_GT .read_text ()
    )
    train_data =json .loads (
    TRAIN_GT .read_text ()
    )
    subset =json .loads (
    FAILURE_SUBSET .read_text ()
    )

    val_images ={
    int (row ["id"]):row 
    for row in val_data ["images"]
    }
    val_annotations ={
    int (row ["id"]):row 
    for row in val_data [
    "annotations"
    ]
    }
    train_images ={
    int (row ["id"]):row 
    for row in train_data ["images"]
    }
    train_annotations ={
    int (row ["id"]):row 
    for row in train_data [
    "annotations"
    ]
    }

    prediction_indexes ={}

    for key in MODEL_KEYS :
        prediction_indexes [key ]=(
        load_prediction_index (
        key ,
        protocol [
        "predictions"
        ][key ],
        )
        )

    subset_ids =[
    int (value )
    for value in subset [
    "annotation_ids"
    ]
    ]

    clean_candidates ={
    family :[]
    for family in FAMILIES [:4 ]
    }

    print ("\n=== BUILDING CLEAN CASE POOLS ===")

    for annotation_id in subset_ids :
        annotation =val_annotations [
        annotation_id 
        ]
        image_id =int (
        annotation ["image_id"]
        )
        category_id =int (
        annotation ["category_id"]
        )

        statuses ={
        key :detects_target (
        annotation ,
        prediction_indexes [
        key 
        ].get (image_id ,[]),
        )
        for key in MODEL_KEYS 
        }

        base ={
        "annotation_id":
        annotation_id ,
        "image_id":
        image_id ,
        "category_id":
        category_id ,
        "class_name":
        CATEGORY_NAMES [
        category_id 
        ],
        "statuses":statuses ,
        }

        conditions ={
        "p2_repairs_small_failure":(
        not statuses ["baseline_yolo"]
        and statuses ["p2_e20"]
        ),
        "p2_remaining_failure":(
        not statuses ["baseline_yolo"]
        and not statuses ["p2_e20"]
        ),
        "risk_short_term_recovery":(
        statuses ["risk_e56"]
        and not statuses [
        "uniform_e56"
        ]
        ),
        "risk_non_transfer_in_original_scene":(
        statuses ["uniform_e56"]
        and not statuses ["risk_e56"]
        ),
        }

        for family ,eligible in (
        conditions .items ()
        ):
            if not eligible :
                continue 

            clean_candidates [
            family 
            ].append ({
            **base ,
            "family":family ,
            "selection_rank":
            deterministic_rank (
            family ,
            annotation_id ,
            ),
            })

    selected_cases ={}

    for family in FAMILIES [:4 ]:
        print (
        family ,
        "eligible=",
        len (clean_candidates [family ]),
        )
        selected_cases [family ]=(
        choose_diverse (
        family ,
        clean_candidates [family ],
        )
        )

    print ("\n=== BUILDING CROP CASE POOLS ===")

    with SCALE_CONTEXT_CSV .open (
    newline ="",
    encoding ="utf-8",
    )as handle :
        crop_rows =list (
        csv .DictReader (handle )
        )

    crop_candidates ={
    "crop_scale_condition_shift":[],
    "crop_context_condition_shift":[],
    }

    for row in crop_rows :
        annotation_id =int (
        row ["annotation_id"]
        )
        image_id =int (row ["image_id"])
        category_id =int (
        row ["category_id"]
        )

        base ={
        "annotation_id":
        annotation_id ,
        "image_id":
        image_id ,
        "category_id":
        category_id ,
        "class_name":
        row ["class_name"],
        "scale_context":
        row ,
        }

        if (
        row [
        "full_scene_network_scale"
        ]=="small"
        and row [
        "crop_network_scale"
        ]in ("medium","large")
        ):
            family =(
            "crop_scale_condition_shift"
            )
            crop_candidates [
            family 
            ].append ({
            **base ,
            "family":family ,
            "selection_rank":
            deterministic_rank (
            family ,
            annotation_id ,
            ),
            })

        neighbor_value =(
        row [
        "neighbor_retention_fraction"
        ]
        )
        neighbor_retention =(
        float (neighbor_value )
        if neighbor_value !=""
        else 1.0 
        )

        if (
        neighbor_retention <=0.50 
        or int (
        row [
        "fully_removed_objects"
        ]
        )>=1 
        ):
            family =(
            "crop_context_condition_shift"
            )
            crop_candidates [
            family 
            ].append ({
            **base ,
            "family":family ,
            "selection_rank":
            deterministic_rank (
            family ,
            annotation_id ,
            ),
            })

    for family in (
    "crop_scale_condition_shift",
    "crop_context_condition_shift",
    ):
        print (
        family ,
        "eligible=",
        len (crop_candidates [family ]),
        )
        selected_cases [family ]=(
        choose_diverse (
        family ,
        crop_candidates [family ],
        )
        )

    corruption_selected ,corruption_summary =(
    build_corruption_cases (
    annotations =val_annotations ,
    images =val_images ,
    subset_annotation_ids =subset_ids ,
    clean_prediction_indexes =
    prediction_indexes ,
    )
    )
    selected_cases .update (
    corruption_selected 
    )

    required_case_fields ={
    "family",
    "case_index",
    "annotation_id",
    "image_id",
    "category_id",
    "class_name",
    "selection_rank",
    }

    for family in FAMILIES :
        if (
        len (selected_cases [family ])
        !=CASES_PER_FAMILY 
        ):
            raise AssertionError (family )

        for case in selected_cases [family ]:
            missing =(
            required_case_fields 
            -set (case )
            )

            if missing :
                raise AssertionError ({
                "family":family ,
                "annotation_id":
                case .get (
                "annotation_id"
                ),
                "missing_fields":
                sorted (missing ),
                })

            if case ["family"]!=family :
                raise AssertionError ({
                "expected_family":family ,
                "actual_family":
                case ["family"],
                "annotation_id":
                case ["annotation_id"],
                })

    print ("\n=== RENDERING FAMILY SHEETS ===")

    case_rows =[]

    for family in FAMILIES :
        rendered_rows =[]

        for case in selected_cases [family ]:
            annotation_id =int (
            case ["annotation_id"]
            )

            if family .startswith ("crop_"):
                annotation =(
                train_annotations [
                annotation_id 
                ]
                )
                image_info =(
                train_images [
                int (case ["image_id"])
                ]
                )
                rendered =render_crop_case (
                case ,
                annotation ,
                image_info ,
                )

            elif family .startswith (
            "blur_or_noise_"
            ):
                annotation =(
                val_annotations [
                annotation_id 
                ]
                )
                image_info =(
                val_images [
                int (case ["image_id"])
                ]
                )
                rendered =(
                render_corruption_case (
                case ,
                annotation ,
                image_info ,
                prediction_indexes ,
                )
                )

            else :
                annotation =(
                val_annotations [
                annotation_id 
                ]
                )
                image_info =(
                val_images [
                int (case ["image_id"])
                ]
                )
                rendered =render_clean_case (
                case ,
                annotation ,
                image_info ,
                prediction_indexes ,
                )

            rendered_rows .append (rendered )

            case_rows .append ({
            "family":family ,
            "case_index":
            case ["case_index"],
            "annotation_id":
            case ["annotation_id"],
            "image_id":
            case ["image_id"],
            "category_id":
            case ["category_id"],
            "class_name":
            case ["class_name"],
            "variant":
            case .get (
            "variant",
            "clean",
            ),
            "selection_rank":
            case ["selection_rank"],
            })

        sheet_path =(
        figure_dir /f"{family }.png"
        )
        save_family_sheet (
        family ,
        rendered_rows ,
        sheet_path ,
        )

        print (
        "PASS:",
        family ,
        sheet_path ,
        )

    write_json (
    WORK /"case_selection.json",
    {
    "version":
    "deterministic_qualitative_selection_v1",
    "status":"complete",
    "protocol_sha256":
    EXPECTED_PROTOCOL_SHA256 ,
    "families":selected_cases ,
    },
    )

    with (
    WORK /"case_selection.csv"
    ).open (
    "w",
    newline ="",
    encoding ="utf-8",
    )as handle :
        writer =csv .DictWriter (
        handle ,
        fieldnames =[
        "family",
        "case_index",
        "annotation_id",
        "image_id",
        "category_id",
        "class_name",
        "variant",
        "selection_rank",
        ],
        )
        writer .writeheader ()
        writer .writerows (case_rows )

    write_json (
    WORK 
    /"corruption_probe_summary.json",
    corruption_summary ,
    )

    metadata ={
    "version":
    "final_qualitative_cases_v1",
    "status":"complete",
    "created_utc":
    datetime .now (
    timezone .utc 
    ).isoformat (),
    "script":str (
    Path (__file__ ).resolve ()
    ),
    "script_sha256":sha256 (
    Path (__file__ ).resolve ()
    ),
    "protocol":str (PROTOCOL ),
    "protocol_sha256":
    EXPECTED_PROTOCOL_SHA256 ,
    "protocol_manifest_sha256":
    EXPECTED_PROTOCOL_MANIFEST_SHA256 ,
    "seed":SEED ,
    "families":len (FAMILIES ),
    "cases_per_family":
    CASES_PER_FAMILY ,
    "total_cases":len (case_rows ),
    "manual_selection":False ,
    "new_training":False ,
    "new_inference":(
    "Only Standard-E20 and P2-E20 "
    "on the frozen 512-image "
    "blur_s2/noise_s2 probe."
    ),
    "ram_crops_required":False ,
    "elapsed_minutes":
    (time .time ()-started )
    /60.0 ,
    }

    write_json (
    WORK /"metadata.json",
    metadata ,
    )

    manifest_path ,manifest_sha =(
    build_manifest (WORK )
    )

    os .replace (
    WORK ,
    OUTPUT ,
    )

    print ("\n"+"="*100 )
    print (
    "FINAL DETERMINISTIC QUALITATIVE "
    "CASES COMPLETE"
    )
    print ("="*100 )

    for family in FAMILIES :
        print (
        family ,
        "cases=",
        len (selected_cases [family ]),
        )

    print ("total_cases:",len (case_rows ))
    print (
    "corruption_probe_images:",
    corruption_summary [
    "probe_images"
    ],
    )
    print (
    "corruption_success_candidates:",
    corruption_summary [
    "success_candidates"
    ],
    )
    print (
    "corruption_failure_candidates:",
    corruption_summary [
    "failure_candidates"
    ],
    )
    print ("output:",OUTPUT )
    print (
    "manifest:",
    OUTPUT /manifest_path .name ,
    )
    print (
    "manifest_sha256:",
    manifest_sha ,
    )
    print ("NEW TRAINING: False")
    print (
    "NEXT: paper tables, markdown "
    "results and final manifest"
    )
    print ("="*100 )


def main ():
    args =parse_args ()

    protocol =preflight ()

    if args .preflight_only :
        return 

    execute (protocol )


if __name__ =="__main__":
    main ()
