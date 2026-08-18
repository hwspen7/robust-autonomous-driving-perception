from __future__ import annotations 

import argparse 
import csv 
import hashlib 
import json 
import math 
from collections import Counter ,defaultdict 
from pathlib import Path 
from typing import Any 


EXPECTED_GT_OBJECTS =185_523 

STRONG_THRESHOLD =0.25 
MODERATE_THRESHOLD =0.10 
WEAK_THRESHOLD =0.05 


def parse_args ()->argparse .Namespace :
    parser =argparse .ArgumentParser (
    description ="Finalize graded BDD100K failure taxonomy."
    )

    parser .add_argument (
    "--diagnosis",
    type =Path ,
    required =True ,
    help ="gt_failure_diagnosis.csv from the Seg threshold=0.05 run.",
    )

    parser .add_argument (
    "--output-dir",
    type =Path ,
    required =True ,
    )

    return parser .parse_args ()


def sha256_file (path :Path )->str :
    digest =hashlib .sha256 ()

    with path .open ("rb")as f :
        while True :
            block =f .read (1024 *1024 )

            if not block :
                break 

            digest .update (block )

    return digest .hexdigest ()


def parse_bool (value :Any )->bool :
    if isinstance (value ,bool ):
        return value 

    text =str (value ).strip ().lower ()

    if text in {"true","1","yes"}:
        return True 

    if text in {"false","0","no"}:
        return False 

    raise ValueError (
    f"Cannot parse bool value: {value !r }"
    )


def parse_optional_bool (
value :Any ,
)->bool |None :
    text =str (value ).strip ().lower ()

    if text in {"","none","nan"}:
        return None 

    return parse_bool (value )


def parse_optional_float (
value :Any ,
)->float |None :
    text =str (value ).strip ()

    if text =="":
        return None 

    number =float (text )

    if not math .isfinite (number ):
        return None 

    return number 


def write_csv (
path :Path ,
rows :list [dict [str ,Any ]],
fields :list [str ],
)->None :
    path .parent .mkdir (
    parents =True ,
    exist_ok =True ,
    )

    with path .open (
    "w",
    newline ="",
    encoding ="utf-8",
    )as f :
        writer =csv .DictWriter (
        f ,
        fieldnames =fields ,
        )

        writer .writeheader ()
        writer .writerows (rows )


def write_json (
path :Path ,
data :Any ,
)->None :
    path .parent .mkdir (
    parents =True ,
    exist_ok =True ,
    )

    with path .open (
    "w",
    encoding ="utf-8",
    )as f :
        json .dump (
        data ,
        f ,
        indent =2 ,
        ensure_ascii =False ,
        allow_nan =False ,
        )


def classify_seg_evidence (
*,
applicable :bool ,
support :bool |None ,
score :float |None ,
)->str :
    if not applicable :
        return "not_applicable"

    if support is not True :
        return "none"

    if score is None :
        raise RuntimeError (
        "Seg support=True but Seg score is missing."
        )

    if score >=STRONG_THRESHOLD :
        return "strong"

    if score >=MODERATE_THRESHOLD :
        return "moderate"

    if score >=WEAK_THRESHOLD :
        return "weak"

    raise RuntimeError (
    "Seg support=True with score below export threshold: "
    f"{score }"
    )


def classify_failure_taxonomy (
*,
yolo_detected :bool ,
dfine_detected :bool ,
seg_evidence :str ,
)->str :
    has_seg_evidence =(
    seg_evidence 
    in {
    "strong",
    "moderate",
    "weak",
    }
    )

    no_seg_evidence =(
    seg_evidence 
    =="not_applicable"
    )


    if yolo_detected and dfine_detected :
        if no_seg_evidence :
            return (
            "stable_detection_no_seg_evidence"
            )

        if has_seg_evidence :
            return (
            "stable_detection_with_region_evidence"
            )

        return (
        "detection_success_without_seg_support"
        )


    if (
    not yolo_detected 
    and dfine_detected 
    ):
        if no_seg_evidence :
            return (
            "yolo_architecture_sensitive_candidate_"
            "no_seg_evidence"
            )

        if has_seg_evidence :
            return (
            "yolo_specific_detection_failure_candidate"
            )

        return (
        "yolo_architecture_sensitive_candidate_"
        "without_seg_support"
        )





    if (
    not yolo_detected 
    and not dfine_detected 
    ):
        if no_seg_evidence :
            return (
            "shared_detection_difficulty_"
            "no_seg_evidence"
            )

        if has_seg_evidence :
            return (
            "shared_detection_difficulty_"
            "with_region_evidence"
            )

        return (
        "shared_visual_hard_sample_candidate"
        )


    if no_seg_evidence :
        return (
        "dfine_architecture_sensitive_candidate_"
        "no_seg_evidence"
        )

    if has_seg_evidence :
        return (
        "dfine_specific_detection_failure_candidate"
        )

    return (
    "dfine_architecture_sensitive_candidate_"
    "without_seg_support"
    )


def load_rows (
path :Path ,
)->tuple [
list [dict [str ,Any ]],
list [str ],
]:
    rows =[]

    with path .open (
    "r",
    encoding ="utf-8",
    newline ="",
    )as f :
        reader =csv .DictReader (f )

        original_fields =list (
        reader .fieldnames or []
        )

        for raw in reader :
            row =dict (raw )

            row ["annotation_id"]=int (
            row ["annotation_id"]
            )

            row ["image_id"]=int (
            row ["image_id"]
            )

            row ["category_id"]=int (
            row ["category_id"]
            )

            row ["yolo_detected"]=(
            parse_bool (
            row ["yolo_detected"]
            )
            )

            row ["dfine_detected"]=(
            parse_bool (
            row ["dfine_detected"]
            )
            )

            row ["seg_applicable"]=(
            parse_bool (
            row ["seg_applicable"]
            )
            )

            row ["seg_support"]=(
            parse_optional_bool (
            row ["seg_support"]
            )
            )

            row ["seg_score"]=(
            parse_optional_float (
            row ["seg_score"]
            )
            )

            row ["seg_bbox_iou"]=(
            parse_optional_float (
            row ["seg_bbox_iou"]
            )
            )

            row [
            "seg_mask_inside_gt_ratio"
            ]=parse_optional_float (
            row [
            "seg_mask_inside_gt_ratio"
            ]
            )

            rows .append (row )

    if len (rows )!=EXPECTED_GT_OBJECTS :
        raise RuntimeError (
        "Unexpected GT row count.\n"
        f"Expected: {EXPECTED_GT_OBJECTS }\n"
        f"Found   : {len (rows )}"
        )

    annotation_ids =[
    row ["annotation_id"]
    for row in rows 
    ]

    if len (annotation_ids )!=len (
    set (annotation_ids )
    ):
        raise RuntimeError (
        "Duplicate annotation_id detected."
        )

    return rows ,original_fields 


def build_group_keys (
row :dict [str ,Any ],
)->list [
tuple [str ,str ]
]:
    timeofday =str (
    row ["timeofday"]
    )

    weather =str (
    row ["weather"]
    )

    scene =str (
    row ["scene"]
    )

    scale =str (
    row ["scale"]
    )

    category =str (
    row ["category"]
    )

    return [
    (
    "overall",
    "all",
    ),

    (
    "timeofday",
    timeofday ,
    ),

    (
    "weather",
    weather ,
    ),

    (
    "scene",
    scene ,
    ),

    (
    "scale",
    scale ,
    ),

    (
    "category",
    category ,
    ),

    (
    "timeofday_x_scale",
    f"{timeofday }|{scale }",
    ),

    (
    "weather_x_scale",
    f"{weather }|{scale }",
    ),

    (
    "scene_x_scale",
    f"{scene }|{scale }",
    ),

    (
    "category_x_scale",
    f"{category }|{scale }",
    ),

    (
    "timeofday_x_category",
    f"{timeofday }|{category }",
    ),

    (
    "weather_x_category",
    f"{weather }|{category }",
    ),

    (
    "scene_x_category",
    f"{scene }|{category }",
    ),
    ]


def main ()->None :
    args =parse_args ()

    diagnosis_path =(
    args .diagnosis 
    .expanduser ()
    .resolve ()
    )

    output_dir =(
    args .output_dir 
    .expanduser ()
    .resolve ()
    )

    if not diagnosis_path .is_file ():
        raise FileNotFoundError (
        diagnosis_path 
        )

    output_dir .mkdir (
    parents =True ,
    exist_ok =True ,
    )

    print ("="*100 )
    print (
    "FINALIZING GRADED FAILURE TAXONOMY"
    )
    print ("="*100 )

    rows ,original_fields =(
    load_rows (
    diagnosis_path 
    )
    )

    taxonomy_counter =Counter ()
    evidence_counter =Counter ()

    for row in rows :
        evidence =(
        classify_seg_evidence (
        applicable =(
        row [
        "seg_applicable"
        ]
        ),
        support =(
        row [
        "seg_support"
        ]
        ),
        score =(
        row [
        "seg_score"
        ]
        ),
        )
        )

        taxonomy =(
        classify_failure_taxonomy (
        yolo_detected =(
        row [
        "yolo_detected"
        ]
        ),
        dfine_detected =(
        row [
        "dfine_detected"
        ]
        ),
        seg_evidence =(
        evidence 
        ),
        )
        )

        row [
        "seg_evidence"
        ]=evidence 

        row [
        "final_failure_taxonomy"
        ]=taxonomy 

        row [
        "primary_yolo_failure"
        ]=(
        not row [
        "yolo_detected"
        ]
        )

        taxonomy_counter [
        taxonomy 
        ]+=1 

        evidence_counter [
        evidence 
        ]+=1 

    output_fields =list (
    original_fields 
    )

    for field in [
    "seg_evidence",
    "final_failure_taxonomy",
    "primary_yolo_failure",
    ]:
        if field not in output_fields :
            output_fields .append (
            field 
            )

    write_csv (
    output_dir 
    /"gt_failure_taxonomy.csv",
    rows ,
    output_fields ,
    )

    grouped_rows =defaultdict (
    list 
    )

    for row in rows :
        for group_key in build_group_keys (
        row 
        ):
            grouped_rows [
            group_key 
            ].append (row )

    failure_rate_rows =[]
    taxonomy_summary_rows =[]
    evidence_summary_rows =[]

    for (
    group_type ,
    group_value ,
    ),group in sorted (
    grouped_rows .items ()
    ):
        num_gt =len (group )

        num_images =len (
        {
        row [
        "image_id"
        ]
        for row in group 
        }
        )

        yolo_failures =[
        row 
        for row in group 
        if not row [
        "yolo_detected"
        ]
        ]

        dfine_failures =[
        row 
        for row in group 
        if not row [
        "dfine_detected"
        ]
        ]

        both_failures =[
        row 
        for row in group 
        if (
        not row [
        "yolo_detected"
        ]
        and not row [
        "dfine_detected"
        ]
        )
        ]

        yolo_fail_dfine_ok =[
        row 
        for row in group 
        if (
        not row [
        "yolo_detected"
        ]
        and row [
        "dfine_detected"
        ]
        )
        ]

        yolo_ok_dfine_fail =[
        row 
        for row in group 
        if (
        row [
        "yolo_detected"
        ]
        and not row [
        "dfine_detected"
        ]
        )
        ]

        failure_rate_rows .append (
        {
        "group_type":(
        group_type 
        ),
        "group_value":(
        group_value 
        ),
        "num_images":(
        num_images 
        ),
        "num_gt":(
        num_gt 
        ),

        "yolo_failures":len (
        yolo_failures 
        ),
        "yolo_failure_rate":(
        len (
        yolo_failures 
        )
        /num_gt 
        ),

        "dfine_failures":len (
        dfine_failures 
        ),
        "dfine_failure_rate":(
        len (
        dfine_failures 
        )
        /num_gt 
        ),

        "both_failures":len (
        both_failures 
        ),
        "both_failure_rate":(
        len (
        both_failures 
        )
        /num_gt 
        ),

        "yolo_fail_dfine_ok":len (
        yolo_fail_dfine_ok 
        ),
        "yolo_fail_dfine_ok_rate":(
        len (
        yolo_fail_dfine_ok 
        )
        /num_gt 
        ),

        "yolo_ok_dfine_fail":len (
        yolo_ok_dfine_fail 
        ),
        "yolo_ok_dfine_fail_rate":(
        len (
        yolo_ok_dfine_fail 
        )
        /num_gt 
        ),
        }
        )

        taxonomy_counts =Counter (
        row [
        "final_failure_taxonomy"
        ]
        for row in group 
        )

        yolo_failure_count =len (
        yolo_failures 
        )

        for (
        taxonomy ,
        count ,
        )in sorted (
        taxonomy_counts .items ()
        ):
            yolo_failure_taxonomy_count =sum (
            1 
            for row in yolo_failures 
            if row [
            "final_failure_taxonomy"
            ]==taxonomy 
            )

            taxonomy_summary_rows .append (
            {
            "group_type":(
            group_type 
            ),
            "group_value":(
            group_value 
            ),
            "num_images":(
            num_images 
            ),
            "num_gt":(
            num_gt 
            ),
            "yolo_failure_count":(
            yolo_failure_count 
            ),
            "taxonomy":(
            taxonomy 
            ),
            "count":count ,
            "share_of_gt":(
            count /num_gt 
            ),
            "count_within_yolo_failures":(
            yolo_failure_taxonomy_count 
            ),
            "share_of_yolo_failures":(
            (
            yolo_failure_taxonomy_count 
            /yolo_failure_count 
            )
            if yolo_failure_count 
            >0 
            else None 
            ),
            }
            )

        applicable_group =[
        row 
        for row in group 
        if row [
        "seg_applicable"
        ]
        ]

        evidence_counts =Counter (
        row [
        "seg_evidence"
        ]
        for row in group 
        )

        for (
        evidence ,
        count ,
        )in sorted (
        evidence_counts .items ()
        ):
            applicable_denominator =(
            len (
            applicable_group 
            )
            )

            evidence_summary_rows .append (
            {
            "group_type":(
            group_type 
            ),
            "group_value":(
            group_value 
            ),
            "num_gt":(
            num_gt 
            ),
            "seg_applicable_gt":(
            applicable_denominator 
            ),
            "seg_evidence":(
            evidence 
            ),
            "count":(
            count 
            ),
            "share_of_gt":(
            count /num_gt 
            ),
            "share_of_seg_applicable_gt":(
            (
            count 
            /applicable_denominator 
            )
            if (
            evidence 
            !="not_applicable"
            and applicable_denominator 
            >0 
            )
            else None 
            ),
            }
            )





    write_csv (
    output_dir 
    /"failure_rates.csv",
    failure_rate_rows ,
    [
    "group_type",
    "group_value",
    "num_images",
    "num_gt",
    "yolo_failures",
    "yolo_failure_rate",
    "dfine_failures",
    "dfine_failure_rate",
    "both_failures",
    "both_failure_rate",
    "yolo_fail_dfine_ok",
    "yolo_fail_dfine_ok_rate",
    "yolo_ok_dfine_fail",
    "yolo_ok_dfine_fail_rate",
    ],
    )

    write_csv (
    output_dir 
    /"failure_taxonomy_summary.csv",
    taxonomy_summary_rows ,
    [
    "group_type",
    "group_value",
    "num_images",
    "num_gt",
    "yolo_failure_count",
    "taxonomy",
    "count",
    "share_of_gt",
    "count_within_yolo_failures",
    "share_of_yolo_failures",
    ],
    )

    write_csv (
    output_dir 
    /"seg_evidence_summary.csv",
    evidence_summary_rows ,
    [
    "group_type",
    "group_value",
    "num_gt",
    "seg_applicable_gt",
    "seg_evidence",
    "count",
    "share_of_gt",
    "share_of_seg_applicable_gt",
    ],
    )









    diagnostic_hard_samples =[
    row 
    for row in rows 
    if row [
    "final_failure_taxonomy"
    ]==(
    "shared_visual_hard_sample_candidate"
    )
    ]

    write_csv (
    output_dir 
    /"diagnostic_shared_hard_samples.csv",
    diagnostic_hard_samples ,
    output_fields ,
    )





    metadata ={
    "dataset":"BDD100K",
    "split":"val",

    "source_diagnosis":str (
    diagnosis_path 
    ),

    "source_sha256":(
    sha256_file (
    diagnosis_path 
    )
    ),

    "num_gt_objects":(
    len (rows )
    ),

    "seg_evidence_policy":{
    "source_prediction_threshold":(
    WEAK_THRESHOLD 
    ),
    "strong":(
    "seg_support=True and score >= 0.25"
    ),
    "moderate":(
    "seg_support=True and "
    "0.10 <= score < 0.25"
    ),
    "weak":(
    "seg_support=True and "
    "0.05 <= score < 0.10"
    ),
    "none":(
    "Seg applicable but no accepted "
    "region support"
    ),
    "not_applicable":(
    "No reliable COCO-to-BDD Seg mapping"
    ),
    },

    "seg_evidence_counts":dict (
    evidence_counter 
    ),

    "failure_taxonomy_counts":dict (
    taxonomy_counter 
    ),

    "research_boundary":{
    "seg_is_diagnostic_evidence":True ,
    "seg_mask_correctness_claim":False ,
    "internal_causal_claim":False ,
    "validation_failures_used_for_training":False ,
    "training_hard_sample_bank_source":(
    "BDD100K train split only"
    ),
    },
    }

    write_json (
    output_dir 
    /"metadata.json",
    metadata ,
    )





    print ()
    print ("="*100 )
    print (
    "FINAL FAILURE TAXONOMY COMPLETE"
    )
    print ("="*100 )

    print (
    f"GT objects: {len (rows )}"
    )

    print ()
    print (
    "Seg evidence:"
    )

    for evidence in [
    "strong",
    "moderate",
    "weak",
    "none",
    "not_applicable",
    ]:
        count =evidence_counter [
        evidence 
        ]

        print (
        f"  {evidence :<20} "
        f"{count :7d}  "
        f"{100.0 *count /len (rows ):6.2f}%"
        )

    print ()
    print (
    "Final taxonomy:"
    )

    for (
    taxonomy ,
    count ,
    )in sorted (
    taxonomy_counter .items (),
    key =lambda item :(
    -item [1 ],
    item [0 ],
    ),
    ):
        print (
        f"  {taxonomy :<65} "
        f"{count :7d}"
        )

    print ()

    overall =next (
    row 
    for row in failure_rate_rows 
    if (
    row ["group_type"]
    =="overall"
    and row ["group_value"]
    =="all"
    )
    )

    print (
    "Overall object failure rates:"
    )

    print (
    f"  YOLO11m : "
    f"{overall ['yolo_failures']:7d} / "
    f"{overall ['num_gt']:7d} = "
    f"{100 *overall ['yolo_failure_rate']:.2f}%"
    )

    print (
    f"  D-FINE-M: "
    f"{overall ['dfine_failures']:7d} / "
    f"{overall ['num_gt']:7d} = "
    f"{100 *overall ['dfine_failure_rate']:.2f}%"
    )

    print (
    f"  Both fail: "
    f"{overall ['both_failures']:7d} / "
    f"{overall ['num_gt']:7d} = "
    f"{100 *overall ['both_failure_rate']:.2f}%"
    )

    print ()
    print (
    "IMPORTANT:"
    )

    print (
    "  diagnostic_shared_hard_samples.csv "
    "comes from VAL and is diagnostic only."
    )

    print (
    "  Do NOT feed these exact validation samples "
    "back into training."
    )

    print ()

    print (
    f"Results: {output_dir }"
    )

    print ("="*100 )


if __name__ =="__main__":
    main ()
