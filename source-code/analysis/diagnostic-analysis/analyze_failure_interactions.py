from __future__ import annotations 

import argparse 
import csv 
import json 
from collections import defaultdict 
from pathlib import Path 


def parse_args ():
    parser =argparse .ArgumentParser ()
    parser .add_argument ("--taxonomy",type =Path ,required =True )
    parser .add_argument ("--output-dir",type =Path ,required =True )
    parser .add_argument ("--min-gt",type =int ,default =100 )
    return parser .parse_args ()


def parse_bool (value ):
    return str (value ).strip ().lower ()in {"true","1","yes"}


def write_csv (path ,rows ,fields ):
    path .parent .mkdir (parents =True ,exist_ok =True )
    with path .open ("w",encoding ="utf-8",newline ="")as f :
        writer =csv .DictWriter (f ,fieldnames =fields )
        writer .writeheader ()
        writer .writerows (rows )


def main ():
    args =parse_args ()

    taxonomy_path =args .taxonomy .expanduser ().resolve ()
    output_dir =args .output_dir .expanduser ().resolve ()
    output_dir .mkdir (parents =True ,exist_ok =True )

    if not taxonomy_path .is_file ():
        raise FileNotFoundError (taxonomy_path )

    with taxonomy_path .open ("r",encoding ="utf-8",newline ="")as f :
        rows =list (csv .DictReader (f ))

    print ("="*100 )
    print ("THREE-WAY FAILURE INTERACTION ANALYSIS")
    print ("="*100 )
    print ("GT objects:",len (rows ))

    group_specs ={
    "timeofday_x_scale_x_category":
    ("timeofday","scale","category"),

    "scene_x_scale_x_category":
    ("scene","scale","category"),

    "weather_x_scale_x_category":
    ("weather","scale","category"),
    }

    grouped =defaultdict (list )

    for row in rows :
        for group_type ,dimensions in group_specs .items ():
            value ="|".join (
            str (row [d ])
            for d in dimensions 
            )

            grouped [
            (group_type ,value )
            ].append (row )

    result_rows =[]

    for (group_type ,group_value ),group in grouped .items ():
        num_gt =len (group )

        yolo_failures =sum (
        not parse_bool (row ["yolo_detected"])
        for row in group 
        )

        dfine_failures =sum (
        not parse_bool (row ["dfine_detected"])
        for row in group 
        )

        both_failures =sum (
        (
        not parse_bool (row ["yolo_detected"])
        and not parse_bool (row ["dfine_detected"])
        )
        for row in group 
        )

        yolo_fail_dfine_ok =sum (
        (
        not parse_bool (row ["yolo_detected"])
        and parse_bool (row ["dfine_detected"])
        )
        for row in group 
        )

        images ={
        int (row ["image_id"])
        for row in group 
        }

        result_rows .append (
        {
        "group_type":group_type ,
        "group_value":group_value ,
        "num_images":len (images ),
        "num_gt":num_gt ,

        "yolo_failures":yolo_failures ,
        "yolo_failure_rate":yolo_failures /num_gt ,

        "dfine_failures":dfine_failures ,
        "dfine_failure_rate":dfine_failures /num_gt ,

        "both_failures":both_failures ,
        "both_failure_rate":both_failures /num_gt ,

        "yolo_fail_dfine_ok":yolo_fail_dfine_ok ,
        "yolo_fail_dfine_ok_rate":
        yolo_fail_dfine_ok /num_gt ,
        }
        )

    fields =[
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
    ]


    result_rows .sort (
    key =lambda row :(
    row ["group_type"],
    -row ["yolo_failure_rate"],
    row ["group_value"],
    )
    )

    write_csv (
    output_dir /"interaction_failure_rates.csv",
    result_rows ,
    fields ,
    )


    priority_rows =[
    row 
    for row in result_rows 
    if row ["num_gt"]>=args .min_gt 
    ]

    grouped_priority =defaultdict (list )

    for row in priority_rows :
        grouped_priority [
        row ["group_type"]
        ].append (row )

    final_priority =[]

    for group_type ,group in grouped_priority .items ():
        group .sort (
        key =lambda row :row ["yolo_failure_rate"],
        reverse =True ,
        )

        for rank ,row in enumerate (group ,start =1 ):
            item =dict (row )
            item ["priority_rank_within_group"]=rank 
            final_priority .append (item )

    write_csv (
    output_dir /"interaction_priority_analysis.csv",
    final_priority ,
    fields +["priority_rank_within_group"],
    )

    summary ={
    "source":str (taxonomy_path ),
    "num_gt_objects":len (rows ),
    "minimum_gt_for_priority":args .min_gt ,
    "num_interaction_groups":len (result_rows ),
    "num_priority_groups":len (final_priority ),
    "groups":list (group_specs ),
    "validation_only":True ,
    "training_leakage_allowed":False ,
    }

    with (
    output_dir /"interaction_summary.json"
    ).open ("w",encoding ="utf-8")as f :
        json .dump (
        summary ,
        f ,
        indent =2 ,
        ensure_ascii =False ,
        )

    print ()
    print ("="*120 )
    print ("TOP THREE-WAY FAILURE INTERACTIONS")
    print ("="*120 )

    for group_type in group_specs :
        candidates =[
        row 
        for row in final_priority 
        if row ["group_type"]==group_type 
        ]

        candidates .sort (
        key =lambda row :row ["yolo_failure_rate"],
        reverse =True ,
        )

        print ()
        print (group_type .upper ())
        print ("-"*120 )

        print (
        f"{'group_value':<55}"
        f"{'GT':>9}"
        f"{'YOLO':>10}"
        f"{'Y-rate':>10}"
        f"{'D-rate':>10}"
        f"{'Both':>10}"
        f"{'Yx/Dok':>10}"
        )

        for row in candidates [:20 ]:
            print (
            f"{row ['group_value']:<55}"
            f"{row ['num_gt']:>9d}"
            f"{row ['yolo_failures']:>10d}"
            f"{row ['yolo_failure_rate']:>10.4f}"
            f"{row ['dfine_failure_rate']:>10.4f}"
            f"{row ['both_failure_rate']:>10.4f}"
            f"{row ['yolo_fail_dfine_ok_rate']:>10.4f}"
            )

    print ()
    print ("="*100 )
    print ("INTERACTION ANALYSIS COMPLETE")
    print ("Results:",output_dir )
    print ("="*100 )


if __name__ =="__main__":
    main ()
