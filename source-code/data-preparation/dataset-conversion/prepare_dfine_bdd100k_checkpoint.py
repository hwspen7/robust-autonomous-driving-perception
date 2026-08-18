from __future__ import annotations 

import argparse 
import copy 
import sys 
from pathlib import Path 

import torch 





BDD_TO_COCO ={
0 :0 ,
2 :2 ,
3 :7 ,
4 :5 ,
5 :6 ,
6 :3 ,
7 :1 ,
8 :9 ,
}


def build_parser ()->argparse .ArgumentParser :
    parser =argparse .ArgumentParser (
    description ="Adapt D-FINE-M COCO classification heads to initialize a BDD100K 10-class checkpoint."
    )
    parser .add_argument ("--dfine-root",type =Path ,required =True )
    parser .add_argument ("--config",type =Path ,required =True )
    parser .add_argument ("--checkpoint",type =Path ,required =True )
    parser .add_argument ("--output",type =Path ,required =True )
    return parser 


def map_classifier (
target :torch .Tensor ,
source :torch .Tensor ,
name :str ,
)->torch .Tensor :

    if target .ndim !=source .ndim or target .shape [1 :]!=source .shape [1 :]:
        raise ValueError(
        f"{name}: shape mismatch except for the class dimension: "
        f"target={tuple(target.shape)}, source={tuple(source.shape)}"
        )

    result = target.clone()

    for bdd_id ,coco_id in BDD_TO_COCO .items ():
        result [bdd_id ].copy_ (source [coco_id ])

    return result 


def map_denoising_embedding (
target :torch .Tensor ,
source :torch .Tensor ,
)->torch .Tensor :


    if target .ndim !=2 or source .ndim !=2 :
        raise ValueError ("denoising_class_embed must be a 2-D tensor")

    if target .shape [1 ]!=source .shape [1 ]:
        raise ValueError (
        "denoising embedding hidden_dim mismatch: "
        f"target={tuple (target .shape )}, source={tuple (source .shape )}"
        )

    result =target .clone ()

    for bdd_id ,coco_id in BDD_TO_COCO .items ():
        result [bdd_id ].copy_ (source [coco_id ])

    result [-1 ].copy_ (source [-1 ])

    return result 


def main ()->None :
    args =build_parser ().parse_args ()

    dfine_root =args .dfine_root .expanduser ().resolve ()
    config_path =args .config .expanduser ().resolve ()
    checkpoint_path =args .checkpoint .expanduser ().resolve ()
    output_path =args .output .expanduser ().resolve ()

    if not dfine_root .exists ():
        raise FileNotFoundError (dfine_root )
    if not config_path .exists ():
        raise FileNotFoundError (config_path )
    if not checkpoint_path .exists ():
        raise FileNotFoundError (checkpoint_path )

    sys .path .insert (0 ,str (dfine_root ))

    from src .core import YAMLConfig 



    cfg =YAMLConfig (str (config_path ))
    if "HGNetv2"in cfg .yaml_cfg :
        cfg .yaml_cfg ["HGNetv2"]["pretrained"]=False 

    target_state =cfg .model .state_dict ()

    checkpoint =torch .load (
    checkpoint_path ,
    map_location ="cpu",
    weights_only =False ,
    )

    if "ema"in checkpoint :
        source_state =checkpoint ["ema"]["module"]
    elif "model"in checkpoint :
        source_state =checkpoint ["model"]
    else :
        raise KeyError("checkpoint has no 'ema' or 'model' state_dict")

    adapted_state =copy .deepcopy (source_state )
    mapped =[]


    for suffix in ("weight","bias"):
        name =f"decoder.enc_score_head.{suffix }"
        adapted_state [name ]=map_classifier (
        target_state [name ],
        source_state [name ],
        name ,
        )
        mapped .append (name )


    decoder_head_names =sorted (
    name 
    for name in source_state 
    if name .startswith ("decoder.dec_score_head.")
    and (name .endswith (".weight")or name .endswith (".bias"))
    )

    for name in decoder_head_names :
        adapted_state [name ]=map_classifier (
        target_state [name ],
        source_state [name ],
        name ,
        )
        mapped .append (name )

    dn_name ="decoder.denoising_class_embed.weight"
    adapted_state [dn_name ]=map_denoising_embedding (
    target_state [dn_name ],
    source_state [dn_name ],
    )
    mapped .append (dn_name )



    output_checkpoint =copy .deepcopy (checkpoint )

    if "model"in output_checkpoint :
        output_checkpoint ["model"]=copy .deepcopy (adapted_state )

    if "ema"in output_checkpoint :
        output_checkpoint ["ema"]["module"]=copy .deepcopy (adapted_state )


    if "model"not in output_checkpoint :
        output_checkpoint ["model"]=copy .deepcopy (adapted_state )

    output_checkpoint ["bdd100k_class_mapping"]={
    str (k ):v for k ,v in BDD_TO_COCO .items ()
    }

    output_path .parent .mkdir (parents =True ,exist_ok =True )
    torch .save (output_checkpoint ,output_path )

    print ("===== D-FINE COCO -> BDD100K checkpoint =====")
    print ("source :",checkpoint_path )
    print ("config :",config_path )
    print ("output :",output_path )
    print ("mapped classes:",BDD_TO_COCO )
    print ("mapped parameters:")
    for name in mapped :
        print (" -",name ,tuple (adapted_state [name ].shape ))

    print ("\nUnmapped BDD classes:")
    print (" - 1 rider")
    print (" - 9 traffic sign")
    print ("\nDone.")


if __name__ =="__main__":
    main ()
