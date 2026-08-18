
import argparse 
import os 
from pathlib import Path 

import torch 
from ultralytics import YOLO 


PROJECT_ROOT =Path (__file__ ).resolve ().parents [1 ]
DEFAULT_DATA_YAML =PROJECT_ROOT /"datasets"/"yolo"/"bdd100k"/"data.yaml"
DEFAULT_OUTPUT_ROOT =PROJECT_ROOT /"results"/"training"/"yolo"


def project_path (
path :Path ,
)->Path :
    expanded =path .expanduser ()

    if not expanded .is_absolute ():
        expanded =PROJECT_ROOT /expanded 

    return expanded .resolve (
    strict =False ,
    )


def default_data_yaml ()->Path :
    for env_name in (
    "AUTODRIVE_YOLO_DATA",
    "BDD100K_YOLO_DATA",
    "YOLO_DATA",
    ):
        configured_path =os .getenv (
        env_name 
        )

        if configured_path :
            return project_path (
            Path (configured_path )
            )

    return DEFAULT_DATA_YAML 


def parse_args ()->argparse .Namespace :
    parser =argparse .ArgumentParser (
    description ="Train the BDD100K YOLO11 baseline."
    )
    parser .add_argument (
    "--profile",
    choices =("smoke","full"),
    default ="smoke",
    help ="smoke validates the pipeline; full runs the baseline.",
    )
    parser .add_argument (
    "--data",
    type =Path ,
    default =default_data_yaml (),
    help ="Path to the YOLO data.yaml file.",
    )
    parser .add_argument (
    "--model",
    type =str ,
    default ="yolo11n.pt",
    help ="Ultralytics model name or weights path.",
    )
    parser .add_argument (
    "--resume",
    nargs ="?",
    const ="true",
    default =None ,
    metavar ="CHECKPOINT",
    help =(
    "Resume training. Pass without a value for Ultralytics auto-resume, "
    "or pass a checkpoint path such as runs/train/exp/weights/last.pt."
    ),
    )
    parser .add_argument (
    "--epochs",
    type =int ,
    default =None ,
    help ="Override default epoch count.",
    )
    parser .add_argument (
    "--batch",
    type =int ,
    default =None ,
    help ="Override batch size.",
    )
    parser .add_argument (
    "--workers",
    type =int ,
    default =None ,
    help ="Override dataloader workers.",
    )
    parser .add_argument (
    "--device",
    type =str ,
    default =None ,
    help ="Device, for example cpu, mps, 0, or 0,1.",
    )
    parser .add_argument (
    "--name",
    type =str ,
    default =None ,
    help ="Experiment name.",
    )
    parser .add_argument (
    "--project",
    type =Path ,
    default =DEFAULT_OUTPUT_ROOT ,
    help ="Output project directory.",
    )
    return parser .parse_args ()


def print_gpu_info ()->None :
    if torch .cuda .is_available ():
        print (f"CUDA GPU count: {torch .cuda .device_count ()}")

        for device_id in range (torch .cuda .device_count ()):
            print (f"GPU {device_id }: {torch .cuda .get_device_name (device_id )}")
        return 

    if (
    hasattr (torch .backends ,"mps")
    and torch .backends .mps .is_available ()
    ):
        print ("Device: Apple MPS")
        return 

    print ("Device: CPU")


def resolve_device (
requested_device :str |None ,
)->str :
    if requested_device is not None :
        return requested_device 

    if torch .cuda .is_available ():
        return "0"

    if (
    hasattr (torch .backends ,"mps")
    and torch .backends .mps .is_available ()
    ):
        return "mps"

    return "cpu"


def get_default_batch (
device :str ,
)->int :
    if device not in {
    "cpu",
    "mps",
    }:
        return 16 

    if device =="mps":
        return 4 

    return 2 


def resolve_resume (
model :str ,
resume :str |None ,
)->tuple [str ,bool ]:
    if resume is None :
        return model ,False 

    normalized_resume =resume .lower ()

    if normalized_resume in {
    "0",
    "false",
    "no",
    "off",
    }:
        return model ,False 

    if normalized_resume in {
    "1",
    "true",
    "yes",
    "on",
    }:
        return model ,True 

    checkpoint_path =project_path (
    Path (resume )
    )

    if not checkpoint_path .exists ():
        raise FileNotFoundError (
        f"Resume checkpoint does not exist: {checkpoint_path }"
        )

    return str (checkpoint_path ),True 


def validate_args (
args :argparse .Namespace ,
)->Path :
    data_yaml =project_path (
    args .data 
    )

    if not data_yaml .exists ():
        raise FileNotFoundError (
        f"Unable to find YOLO data file: {data_yaml }"
        )

    return data_yaml 


def build_train_args (
args :argparse .Namespace ,
data_yaml :Path ,
device :str ,
resume_training :bool ,
)->dict :
    model_name =Path (args .model ).stem 
    project_dir =project_path (
    args .project 
    )

    if args .profile =="smoke":
        train_args ={
        "data":str (data_yaml ),
        "epochs":args .epochs if args .epochs is not None else 1 ,
        "imgsz":640 ,
        "batch":args .batch if args .batch is not None else 4 ,
        "device":device ,
        "workers":args .workers if args .workers is not None else 0 ,
        "fraction":0.01 ,
        "val":False ,
        "plots":False ,
        "save":True ,
        "project":str (project_dir ),
        "name":args .name or f"{model_name }_smoke",
        "exist_ok":True ,
        "seed":42 ,
        "deterministic":True ,
        "verbose":True ,
        }
    else :
        train_args ={
        "data":str (data_yaml ),
        "epochs":args .epochs if args .epochs is not None else 50 ,
        "imgsz":640 ,
        "batch":(
        args .batch 
        if args .batch is not None 
        else get_default_batch (device )
        ),
        "device":device ,
        "workers":args .workers if args .workers is not None else 8 ,
        "fraction":1.0 ,
        "val":True ,
        "plots":True ,
        "save":True ,
        "save_period":5 ,
        "project":str (project_dir ),
        "name":args .name or f"{model_name }_baseline",
        "exist_ok":False ,
        "seed":42 ,
        "deterministic":True ,
        "patience":10 ,
        "verbose":True ,
        }

    if resume_training :
        train_args ["resume"]=True 

    return train_args 


def main ()->None :
    args =parse_args ()

    print ("\n========== GPU Information ==========")
    print_gpu_info ()

    data_yaml =validate_args (
    args 
    )
    device =resolve_device (
    args .device 
    )
    model_path ,resume_training =resolve_resume (
    model =args .model ,
    resume =args .resume ,
    )
    train_args =build_train_args (
    args =args ,
    data_yaml =data_yaml ,
    device =device ,
    resume_training =resume_training ,
    )

    print ("========== YOLO11 Baseline ==========")
    print (f"Profile: {args .profile }")
    print (f"Model: {model_path }")
    print (f"Resume: {resume_training }")
    print (f"Device: {device }")
    print (f"Data: {data_yaml }")
    print (f"Epochs: {train_args ['epochs']}")
    print (f"Batch: {train_args ['batch']}")
    print (f"Workers: {train_args ['workers']}")
    print (f"Output: {Path (train_args ['project'])/train_args ['name']}")

    model =YOLO (
    model_path 
    )
    model .train (
    **train_args 
    )


if __name__ =="__main__":
    main ()
