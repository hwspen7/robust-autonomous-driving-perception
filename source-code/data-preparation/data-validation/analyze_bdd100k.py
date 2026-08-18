import argparse 
import ast 
import csv 
import json 
import os 
import sys 
import tempfile 
import textwrap 
from collections import Counter 
from pathlib import Path 
from typing import Any 

PROJECT_ROOT =Path (__file__ ).resolve ().parents [1 ]

if str (PROJECT_ROOT )not in sys .path :
    sys .path .insert (0 ,str (PROJECT_ROOT ))

os .environ .setdefault (
"MPLCONFIGDIR",
str (Path (tempfile .gettempdir ())/"matplotlib"),
)

import matplotlib 

matplotlib .use ("Agg")
import matplotlib .pyplot as plt 
from tqdm import tqdm 

from configs .data_taxonomy import SOURCE_TO_CANONICAL 


def project_path (path :Path )->Path :
    if path .is_absolute ():
        return path 
    return PROJECT_ROOT /path 


def parse_args ()->argparse .Namespace :
    parser =argparse .ArgumentParser (description ="Analyze BDD100K Supervisely annotations.")
    parser .add_argument (
    "--data-root",
    type =Path ,
    default =Path ("datasets/downloads/bdd100k/bdd100k:-images-100k"),
    help ="Root directory of the BDD100K Supervisely dataset.",
    )
    parser .add_argument (
    "--split",
    type =str ,
    default ="train",
    help ="Dataset split to analyze (e.g., train, val, or test).",
    )
    parser .add_argument (
    "--output-dir",
    type =Path ,
    default =Path ("reports/bdd100k_statistics"),
    help ="Directory to write analysis results.",
    )
    parser .add_argument (
    "--max-files",
    type =int ,
    default =None ,
    help ="Maximum number of annotation files to analyze; default analyzes all.",
    )
    return parser .parse_args ()

def load_json (
path :Path ,
)->dict [str ,Any ]:
    with path .open ("r",encoding ="utf-8")as file :
        data =json .load (file )

    if not isinstance (data ,dict ):
        raise TypeError (
        f"The input data is not a dict: {path }"
        )
    return data 

def parse_image_tags (
tags :list [dict [str ,Any ]],
)->dict [str ,str ]:
    result ={
    "weather":"undefined",
    "scene":"undefined",
    "timeofday":"undefined",
    }
    for tag in tags :
        name =tag .get ("name")
        value =tag .get ("value")

        if name in result and value is not None :
            result [name ]=str (value )
    return result 

def parse_object_attributes (
tags :list [dict [str ,Any ]],
)->dict [str ,Any ]:
    for tag in tags :
        if tag .get ("name")!="attributes":
            continue 

        value =tag .get ("value",{})

        if isinstance (value ,dict ):
            return value 

        if isinstance (value ,str ):
            try :
                parsed =ast .literal_eval (value )
            except (ValueError ,SyntaxError ):
                return {}
            if isinstance (parsed ,dict ):
                return parsed 
    return {}

def parse_rectangle (
obj :dict [str ,Any ],
image_width :int ,
image_height :int ,
)->tuple [float ,float ,float ,float ]|None :
    exterior =obj .get ("points",{}).get ("exterior",[])
    if len (exterior )<2 :
        return None 

    try :
        x1 ,y1 =map (float ,exterior [0 ])
        x2 ,y2 =map (float ,exterior [1 ])
    except (ValueError ,TypeError ):
        return None 

    left =max (0.0 ,min (x1 ,x2 ))
    top =max (0.0 ,min (y1 ,y2 ))
    right =min (float (image_width ),max (x1 ,x2 ))
    bottom =min (float (image_height ),max (y1 ,y2 ))

    width =right -left 
    height =bottom -top 

    if width <=0 or height <=0 :
        return None 

    return left ,top ,width ,height 

def classify_coco_size (
area :float 
)->str :




    if area <32 **2 :
        return "small"
    if area <96 **2 :
        return "medium"
    return "large"

def save_counter_csv (
counter :Counter ,
path :Path ,
)->None :
    path .parent .mkdir (parents =True ,exist_ok =True )
    with open (path ,"w",encoding ="utf-8",newline ="")as file :
        writer =csv .writer (file )
        writer .writerow (["name","count"])
        for name ,count in counter .most_common ():
            writer .writerow ([name ,count ])

def save_bar_chart (
counter :Counter ,
path :Path ,
title :str ,
horizontal :bool =False ,
)->None :
    if not counter :
        return 

    items =counter .most_common ()
    names =[str (name )for name ,_ in items ]
    values =[count for _ ,count in items ]
    positions =list (range (len (names )))

    fig ,ax =plt .subplots (figsize =(10 ,6 ))

    if horizontal :
        ax .barh (positions ,values [::-1 ])
        ax .set_yticks (positions )
        ax .set_yticklabels (names [::-1 ])
        ax .set_xlabel ("Count")
    else :
        labels =[
        "\n".join (textwrap .wrap (name ,width =12 ,break_long_words =False ))
        for name in names 
        ]
        ax .bar (positions ,values ,align ="center")
        ax .set_xticks (positions )
        ax .set_xticklabels (labels ,rotation =0 ,ha ="center")
        ax .tick_params (axis ="x",pad =8 )
        ax .set_ylabel ("Count")

    ax .set_title (title )
    fig .tight_layout ()
    fig .savefig (path ,dpi =200 )
    plt .close (fig )

def analyze_dataset (
annotation_dir :Path ,
image_dir :Path ,
max_files :int |None ,
)->dict [str ,Any ]:
    annotation_files =sorted (annotation_dir .glob ("*.json"))
    if max_files is not None :
        annotation_files =annotation_files [:max_files ]

    class_counter :Counter [str ]=Counter ()
    source_class_counter :Counter [str ]=Counter ()
    geometry_counter :Counter [str ]=Counter ()
    weather_counter :Counter [str ]=Counter ()
    scene_counter :Counter [str ]=Counter ()
    timeofday_counter :Counter [str ]=Counter ()
    size_counter :Counter [str ]=Counter ()

    total_objects =0 
    total_detection_objects =0 
    empty_detection_images =0 
    missing_images =0 
    invalid_boxes =0 
    occluded_objects =0 
    truncated_objects =0 

    for annotation_path in tqdm (annotation_files ,desc ="Analyzing BDD100K"):
        data =load_json (annotation_path )

        image_name =annotation_path .name .removesuffix (".json")
        image_path =image_dir /image_name 

        if not image_path .exists ():
            missing_images +=1 

        image_tags =parse_image_tags (data .get ("tags",[]))
        weather_counter [image_tags ["weather"]]+=1 
        scene_counter [image_tags ["scene"]]+=1 
        timeofday_counter [image_tags ["timeofday"]]+=1 

        size =data .get ("size",{})
        image_width =int (size .get ("width",0 ))
        image_height =int (size .get ("height",0 ))

        detection_count =0 

        for obj in data .get ("objects",[]):
            total_objects +=1 

            source_class =str (obj .get ("classTitle","unknown"))
            geometry_type =str (obj .get ("geometryType","unknown"))

            source_class_counter [source_class ]+=1 
            geometry_counter [geometry_type ]+=1 

            if geometry_type !="rectangle":
                continue 

            canonical_class =SOURCE_TO_CANONICAL .get (source_class )
            if canonical_class is None :
                continue 

            bbox =parse_rectangle (
            obj ,
            image_width ,
            image_height ,
            )

            if bbox is None :
                invalid_boxes +=1 
                continue 

            _ ,_ ,width ,height =bbox 
            area =width *height 

            class_counter [canonical_class ]+=1 
            size_counter [classify_coco_size (area )]+=1 

            total_detection_objects +=1 
            detection_count +=1 

            attributes =parse_object_attributes (obj .get ("tags",[]))

            if bool (attributes .get ("occluded",False )):
                occluded_objects +=1 

            if bool (attributes .get ("truncated",False )):
                truncated_objects +=1 

        if detection_count ==0 :
            empty_detection_images +=1 

    num_images =len (annotation_files )

    return {
    "num_images":num_images ,
    "num_existing_images":num_images -missing_images ,
    "num_missing_images":missing_images ,
    "num_all_objects":total_objects ,
    "num_detection_objects":total_detection_objects ,
    "average_detection_objects_per_image":total_detection_objects /num_images if num_images >0 else 0 ,
    "num_empty_detection_images":empty_detection_images ,
    "num_invalid_boxes":invalid_boxes ,
    "num_occluded_objects":occluded_objects ,
    "num_truncated_objects":truncated_objects ,
    "class_counts":dict (class_counter ),
    "source_class_counts":dict (source_class_counter ),
    "geometry_counts":dict (geometry_counter ),
    "weather_counts":dict (weather_counter ),
    "scene_counts":dict (scene_counter ),
    "timeofday_counts":dict (timeofday_counter ),
    "bbox_size_counts":dict (size_counter ),
    }

def main ()->None :
    args =parse_args ()

    data_root =project_path (args .data_root )
    output_root =project_path (args .output_dir )

    split_root =data_root /args .split 
    annotation_dir =split_root /"ann"
    image_dir =split_root /"img"

    if not annotation_dir .exists ():
        raise FileNotFoundError (
        f"{annotation_dir } doesn't exist. "
        f"Current working directory: {Path .cwd ()}. "
        f"Project root: {PROJECT_ROOT }."
        )

    if not image_dir .exists ():
        raise FileNotFoundError (
        f"{image_dir } doesn't exist. "
        f"Current working directory: {Path .cwd ()}. "
        f"Project root: {PROJECT_ROOT }."
        )

    output_dir =output_root /args .split 
    output_dir .mkdir (parents =True ,exist_ok =True )

    summary =analyze_dataset (
    annotation_dir ,
    image_dir ,
    max_files =args .max_files ,
    )

    summary_path =output_dir /"summary.json"

    with summary_path .open ("w",encoding ="utf-8")as file :
        json .dump (summary ,file ,ensure_ascii =False ,indent =2 )

    summary_counters ={
    "class_counts":Counter (summary ["class_counts"]),
    "source_class_counts":Counter (summary ["source_class_counts"]),
    "geometry_counts":Counter (summary ["geometry_counts"]),
    "weather_counts":Counter (summary ["weather_counts"]),
    "scene_counts":Counter (summary ["scene_counts"]),
    "timeofday_counts":Counter (summary ["timeofday_counts"]),
    "bbox_size_counts":Counter (summary ["bbox_size_counts"]),
    }

    for name ,counter in summary_counters .items ():
        save_counter_csv (
        counter ,
        path =output_dir /f"{name }.csv",
        )

    save_bar_chart (
    summary_counters ["class_counts"],
    output_dir /"class_counts.png",
    "BDD100K Detection Class Distribution",
    horizontal =True ,
    )

    save_bar_chart (
    summary_counters ["weather_counts"],
    output_dir /"weather_counts.png",
    "BDD100K Weather Distribution",
    )

    save_bar_chart (
    summary_counters ["scene_counts"],
    output_dir /"scene_counts.png",
    "BDD100K Scene Distribution",
    )

    save_bar_chart (
    summary_counters ["timeofday_counts"],
    output_dir /"timeofday_counts.png",
    "BDD100K Time of Day Distribution",
    )

    save_bar_chart (
    summary_counters ["bbox_size_counts"],
    output_dir /"bbox_size_counts.png",
    "BDD100K Bounding Box Distribution",
    )

    print ("BDD100K Analysis")
    print (f"Split：{args .split }")
    print (f"Images：{summary ['num_images']}")
    print (f"All objects：{summary ['num_all_objects']}")
    print (f"Detection objects：{summary ['num_detection_objects']}")
    print (f"Empty detection images：{summary ['num_empty_detection_images']}")
    print (f"Invalid boxes：{summary ['num_invalid_boxes']}")
    print (f"Occluded objects：{summary ['num_occluded_objects']}")
    print (f"Truncated objects：{summary ['num_truncated_objects']}")
    print (f"Output：{output_dir .resolve ()}")

if __name__ =='__main__':
    main ()
