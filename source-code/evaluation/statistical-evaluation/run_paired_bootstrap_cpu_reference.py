from __future__ import annotations 

import argparse 
import csv 
import gc 
import hashlib 
import json 
import math 
import multiprocessing as mp 
import os 
import time 
from collections import defaultdict 
from pathlib import Path 

os .environ .setdefault ("OMP_NUM_THREADS","1")
os .environ .setdefault ("OPENBLAS_NUM_THREADS","1")
os .environ .setdefault ("MKL_NUM_THREADS","1")
os .environ .setdefault ("NUMEXPR_NUM_THREADS","1")

import numpy as np 
from pycocotools .coco import COCO 
from pycocotools .cocoeval import COCOeval 


ROOT =Path (
"/root/rivermind-data/autodrive/results/evaluation/"
"final_analysis_v1"
)

PROTOCOL =(
ROOT /"frozen_protocol/final_testing_protocol_v1.json"
)
CANONICAL_ROOT =ROOT /"canonical_clean_v1"
CANONICAL_MANIFEST =(
CANONICAL_ROOT /"artifact_manifest.json"
)
CANONICAL_METRICS =(
CANONICAL_ROOT /"canonical_clean_metrics.csv"
)
FAILURE_MANIFEST =(
ROOT 
/"official_val_failure_subsets_v1/"
"artifact_manifest.json"
)
FAILURE_TABLE =(
ROOT 
/"official_val_failure_subsets_v1/"
"failure_object_table.csv"
)
STAGE3_MANIFEST =(
ROOT 
/"threshold_sensitivity_v1/"
"artifact_manifest.json"
)
STAGE4_MANIFEST =(
ROOT 
/"training_dynamics_attribution_v1/"
"artifact_manifest.json"
)

GT =Path (
"/root/rivermind-data/autodrive/datasets/datasets/"
"bdd100k_final/coco/annotations/instances_val.json"
)

OUTPUT =ROOT /"paired_bootstrap_v1"

EXPECTED_SHA256 ={
PROTOCOL :
"693c39375cc1eefc4f3ff9f957a8355fb947552a19dfef228cae6cfc21e3dcfe",
CANONICAL_MANIFEST :
"88d2bace7a2f69e26c34e7eed4e2814b12e8317980602968b407d1b94282338b",
FAILURE_MANIFEST :
"79029af794f214d45bdecb6cb00d9b64155aa8ace04df97ca6524e1f40f2c1dc",
STAGE3_MANIFEST :
"de9536176d1d171b60ac647ed4d834faf55f21598bae24e19440f4ef5f2a9460",
STAGE4_MANIFEST :
"22ce36eed8f85d1e7cbc013b6763bb8c8535146fc6dd55c408c2f31b5356d2d9",
GT :
"7be8a02e147743123a4a62b2c92ed2e4a5712d261c8868223e6325bad6e346d6",
}

PREDICTIONS ={
"baseline_yolo":Path (
"/root/rivermind-data/autodrive/results/evaluation/"
"unified_detection/canonical_predictions/YOLO11m.json"
),
"standard_e20":Path (
"/root/rivermind-data/autodrive/results/evaluation/"
"architecture_gate/borderline_adjudication_v5_1/"
"predictions/standard_epoch20/predictions.json"
),
"p2_e20":Path (
"/root/rivermind-data/autodrive/results/evaluation/"
"architecture_gate/borderline_adjudication_v5_1/"
"predictions/p2_epoch20/predictions.json"
),
}

for key in (
"common40",
"gate_uniform",
"gate_scalar",
"gate_typed",
"risk_e56",
"uniform_e56",
"risk_e100",
"uniform_e100",
):
    PREDICTIONS [key ]=(
    CANONICAL_ROOT 
    /f"predictions/{key }/predictions.json"
    )

COMPARISONS ={
"p2_e20_minus_standard_e20":
("p2_e20","standard_e20"),
"gate_scalar_minus_gate_uniform":
("gate_scalar","gate_uniform"),
"gate_typed_minus_gate_uniform":
("gate_typed","gate_uniform"),
"gate_typed_minus_gate_scalar":
("gate_typed","gate_scalar"),
"rebu_risk_e56_minus_uniform_e56":
("risk_e56","uniform_e56"),
"risk_e100_minus_uniform_e100":
("risk_e100","uniform_e100"),
"common40_minus_baseline_YOLO11m":
("common40","baseline_yolo"),
}

GATE_MODELS ={
"gate_uniform",
"gate_scalar",
"gate_typed",
}

FAILURE_SUBSETS =(
"small_jointly_missed",
"dfine_supported_yolo_failure",
"yolo_low_confidence",
"yolo_localization_failure",
"yolo_classification_failure",
)

CLEAN_METRICS =(
"AP",
"AP50",
"AP75",
"AP_small",
"AP_medium",
"AP_large",
"AR100",
"AR_small",
)

FAILURE_METRICS =(
"failure_macro_AR100",
"small_jointly_missed_recall50",
)

REPLICATES =2000 
SEED =20260816 
EXPECTED_IMAGES =10_000 
EXPECTED_OBJECTS =185_523 

GLOBAL_CLEAN_CACHES =None 
GLOBAL_FAILURE_CACHES =None 
GLOBAL_WEIGHTS =None 


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


def verify (path :Path ,expected :str )->None :
    if not path .is_file ():
        raise FileNotFoundError (path )

    actual =sha256_file (path )

    print (path )
    print (" expected:",expected )
    print (" actual  :",actual )

    if actual !=expected :
        raise AssertionError (path )


def load_canonical_rows ()->dict [str ,dict ]:
    with CANONICAL_METRICS .open (newline ="")as handle :
        rows =list (csv .DictReader (handle ))

    result ={
    row ["model_key"]:row 
    for row in rows 
    }

    for key in PREDICTIONS :
        if key not in result :
            raise KeyError (key )

    return result 


def load_failure_ids ()->dict [str ,set [int ]]:
    result ={
    subset :set ()
    for subset in FAILURE_SUBSETS 
    }

    rows =0 

    with FAILURE_TABLE .open (newline ="")as handle :
        reader =csv .DictReader (handle )

        for rows ,row in enumerate (reader ,start =1 ):
            annotation_id =int (row ["annotation_id"])

            for subset in FAILURE_SUBSETS :
                if int (row [f"in_{subset }"]):
                    result [subset ].add (annotation_id )

    if rows !=EXPECTED_OBJECTS :
        raise AssertionError (rows )

    return result 


def preflight ():
    print ("=== PAIRED BOOTSTRAP PREFLIGHT ===")

    for path ,expected in EXPECTED_SHA256 .items ():
        verify (path ,expected )

    protocol =json .loads (PROTOCOL .read_text ())
    rules =protocol ["paired_bootstrap"]

    assert rules ["enabled"]is True 
    assert rules ["replicates"]==REPLICATES 
    assert rules ["seed"]==SEED 
    assert rules ["confidence_level"]==0.95 
    assert rules ["interval"]=="percentile"
    assert rules ["paired_images_for_both_models"]is True 
    assert rules ["naive_mean_per_image_AP_forbidden"]is True 
    assert set (rules ["primary_comparisons"])==set (
    COMPARISONS 
    )

    canonical_rows =load_canonical_rows ()

    for key ,path in PREDICTIONS .items ():
        if not path .is_file ():
            raise FileNotFoundError (path )

        expected =canonical_rows [key ][
        "prediction_sha256"
        ]
        actual =sha256_file (path )

        print (
        "prediction:",
        key ,
        "size_MiB=",
        round (path .stat ().st_size /1024 **2 ,3 ),
        )
        print (" expected:",expected )
        print (" actual  :",actual )

        if actual !=expected :
            raise AssertionError (path )

    failure_ids =load_failure_ids ()

    for subset ,ids in failure_ids .items ():
        if not ids :
            raise AssertionError (subset )
        print (subset ,"objects=",len (ids ))

    if OUTPUT .exists ():
        raise FileExistsError (OUTPUT )

    incomplete =sorted (
    OUTPUT .parent .glob (
    OUTPUT .name +".incomplete-*"
    )
    )

    if incomplete :
        raise AssertionError (incomplete )

    free_gib =(
    os .statvfs (OUTPUT .parent ).f_bavail 
    *os .statvfs (OUTPUT .parent ).f_frsize 
    /1024 **3 
    )

    print ("logical_cpus:",os .cpu_count ())
    print ("parallel_workers:",min (64 ,os .cpu_count ()or 1 ))
    print ("free_GiB:",round (free_gib ,3 ))

    if free_gib <1.0 :
        raise RuntimeError ("Insufficient output space.")

    print ("PASS: Bootstrap protocol and inputs are frozen")

    return protocol ,canonical_rows ,failure_ids 


def create_coco (dataset :dict )->COCO :
    coco =COCO ()
    coco .dataset =dataset 
    coco .createIndex ()
    return coco 


def compact_evaluator (
evaluator :COCOeval ,
retained_area_labels :tuple [str ,...],
)->dict :
    params =evaluator .params 

    image_ids =list (params .imgIds )
    category_ids =list (params .catIds )
    area_labels =list (params .areaRngLbl )

    image_count =len (image_ids )
    category_count =len (category_ids )
    area_count =len (area_labels )
    threshold_count =len (params .iouThrs )

    cache ={
    "iou_thresholds":
    np .asarray (params .iouThrs ,dtype =np .float64 ),
    "recall_thresholds":
    np .asarray (params .recThrs ,dtype =np .float64 ),
    "areas":{},
    }

    for area_label in retained_area_labels :
        area_index =area_labels .index (area_label )
        category_cells =[]

        for category_index in range (category_count ):
            score_parts =[]
            image_parts =[]
            match_parts =[]
            ignore_parts =[]
            gt_counts =np .zeros (
            image_count ,
            dtype =np .int32 ,
            )

            for image_index in range (image_count ):
                flat_index =(
                category_index 
                *area_count 
                *image_count 
                +area_index *image_count 
                +image_index 
                )

                record =evaluator .evalImgs [flat_index ]

                if record is None :
                    continue 

                gt_ignore =np .asarray (
                record ["gtIgnore"],
                dtype =bool ,
                )

                gt_counts [image_index ]=int (
                np .count_nonzero (~gt_ignore )
                )

                scores =np .asarray (
                record ["dtScores"],
                dtype =np .float64 ,
                )

                if len (scores )==0 :
                    continue 

                matches =(
                np .asarray (
                record ["dtMatches"],
                )>0 
                )
                ignores =np .asarray (
                record ["dtIgnore"],
                dtype =bool ,
                )

                if matches .shape !=(
                threshold_count ,
                len (scores ),
                ):
                    raise AssertionError (
                    matches .shape 
                    )

                score_parts .append (scores )
                image_parts .append (
                np .full (
                len (scores ),
                image_index ,
                dtype =np .int32 ,
                )
                )
                match_parts .append (matches )
                ignore_parts .append (ignores )

            if score_parts :
                scores =np .concatenate (score_parts )
                image_indices =np .concatenate (image_parts )
                matches =np .concatenate (
                match_parts ,
                axis =1 ,
                )
                ignores =np .concatenate (
                ignore_parts ,
                axis =1 ,
                )

                order =np .argsort (
                -scores ,
                kind ="mergesort",
                )

                scores =scores [order ]
                image_indices =image_indices [order ]
                matches =matches [:,order ]
                ignores =ignores [:,order ]
            else :
                scores =np .empty (0 ,dtype =np .float64 )
                image_indices =np .empty (
                0 ,
                dtype =np .int32 ,
                )
                matches =np .empty (
                (threshold_count ,0 ),
                dtype =bool ,
                )
                ignores =np .empty (
                (threshold_count ,0 ),
                dtype =bool ,
                )

            category_cells .append ({
            "category_id":
            int (category_ids [category_index ]),
            "scores":scores ,
            "image_indices":image_indices ,
            "matches":matches ,
            "ignores":ignores ,
            "gt_counts":gt_counts ,
            })

        cache ["areas"][area_label ]=category_cells 

    return cache 


def build_evaluation_cache (
dataset :dict ,
predictions :list [dict ],
retained_area_labels :tuple [str ,...],
)->dict :
    coco_gt =create_coco (dataset )
    coco_dt =coco_gt .loadRes (predictions )

    evaluator =COCOeval (
    coco_gt ,
    coco_dt ,
    "bbox",
    )

    evaluator .params .imgIds =sorted (
    int (image ["id"])
    for image in dataset ["images"]
    )
    evaluator .params .catIds =sorted (
    int (category ["id"])
    for category in dataset ["categories"]
    )
    evaluator .params .maxDets =[1 ,10 ,100 ]

    evaluator .evaluate ()

    cache =compact_evaluator (
    evaluator ,
    retained_area_labels ,
    )

    del evaluator 
    del coco_dt 
    del coco_gt 
    gc .collect ()

    return cache 


def accumulate_area (
cache :dict ,
area_label :str ,
image_weights :np .ndarray ,
)->dict :
    recall_thresholds =cache ["recall_thresholds"]
    iou_thresholds =cache ["iou_thresholds"]

    precision_blocks =[]
    recall_blocks =[]

    micro_tp50 =0.0 
    micro_gt =0.0 

    for cell in cache ["areas"][area_label ]:
        gt_total =float (np .dot (
        cell ["gt_counts"].astype (np .float64 ),
        image_weights ,
        ))

        if gt_total <=0 :
            continue 

        micro_gt +=gt_total 

        detection_weights =image_weights [
        cell ["image_indices"]
        ].astype (np .float64 )

        active =~cell ["ignores"]

        true_positive =(
        cell ["matches"]&active 
        ).astype (np .float64 )

        false_positive =(
        (~cell ["matches"])&active 
        ).astype (np .float64 )

        true_positive *=detection_weights [None ,:]
        false_positive *=detection_weights [None ,:]

        tp_cumulative =np .cumsum (
        true_positive ,
        axis =1 ,
        )
        fp_cumulative =np .cumsum (
        false_positive ,
        axis =1 ,
        )

        q =np .zeros (
        (
        len (iou_thresholds ),
        len (recall_thresholds ),
        ),
        dtype =np .float64 ,
        )
        final_recall =np .zeros (
        len (iou_thresholds ),
        dtype =np .float64 ,
        )

        if tp_cumulative .shape [1 ]>0 :
            for threshold_index in range (
            len (iou_thresholds )
            ):
                tp =tp_cumulative [threshold_index ]
                fp =fp_cumulative [threshold_index ]

                recall =tp /gt_total 
                precision =tp /(
                tp +fp +np .spacing (1 )
                )

                for index in range (
                len (precision )-1 ,
                0 ,
                -1 ,
                ):
                    if precision [index ]>precision [index -1 ]:
                        precision [index -1 ]=precision [index ]

                indices =np .searchsorted (
                recall ,
                recall_thresholds ,
                side ="left",
                )

                valid =indices <len (precision )
                q [threshold_index ,valid ]=(
                precision [indices [valid ]]
                )

                final_recall [threshold_index ]=(
                recall [-1 ]
                )

            micro_tp50 +=float (
            tp_cumulative [0 ,-1 ]
            )

        precision_blocks .append (q )
        recall_blocks .append (final_recall )

    if not precision_blocks :
        return {
        "AP":float ("nan"),
        "AP50":float ("nan"),
        "AP75":float ("nan"),
        "AR100":float ("nan"),
        "recall50_micro":float ("nan"),
        }

    precision =np .stack (
    precision_blocks ,
    axis =0 ,
    )
    recall =np .stack (
    recall_blocks ,
    axis =0 ,
    )

    iou75_index =int (np .argmin (
    np .abs (iou_thresholds -0.75 )
    ))

    return {
    "AP":float (np .mean (precision )),
    "AP50":float (np .mean (
    precision [:,0 ,:]
    )),
    "AP75":float (np .mean (
    precision [:,iou75_index ,:]
    )),
    "AR100":float (np .mean (recall )),
    "recall50_micro":(
    micro_tp50 /micro_gt 
    if micro_gt >0 
    else float ("nan")
    ),
    }


def clean_metrics_from_cache (
cache :dict ,
image_weights :np .ndarray ,
)->np .ndarray :
    all_metrics =accumulate_area (
    cache ,
    "all",
    image_weights ,
    )
    small_metrics =accumulate_area (
    cache ,
    "small",
    image_weights ,
    )
    medium_metrics =accumulate_area (
    cache ,
    "medium",
    image_weights ,
    )
    large_metrics =accumulate_area (
    cache ,
    "large",
    image_weights ,
    )

    values ={
    "AP":all_metrics ["AP"],
    "AP50":all_metrics ["AP50"],
    "AP75":all_metrics ["AP75"],
    "AP_small":small_metrics ["AP"],
    "AP_medium":medium_metrics ["AP"],
    "AP_large":large_metrics ["AP"],
    "AR100":all_metrics ["AR100"],
    "AR_small":small_metrics ["AR100"],
    }

    return np .asarray (
    [values [key ]for key in CLEAN_METRICS ],
    dtype =np .float64 ,
    )


def failure_metrics_from_caches (
caches :dict [str ,dict ],
image_weights :np .ndarray ,
)->np .ndarray :
    subset_results ={
    subset :accumulate_area (
    cache ,
    "all",
    image_weights ,
    )
    for subset ,cache in caches .items ()
    }

    failure_macro =float (np .mean ([
    subset_results [subset ]["AR100"]
    for subset in FAILURE_SUBSETS 
    ]))

    small_joint_recall =subset_results [
    "small_jointly_missed"
    ]["recall50_micro"]

    return np .asarray (
    [
    failure_macro ,
    small_joint_recall ,
    ],
    dtype =np .float64 ,
    )


def bootstrap_worker (task ):
    model ,replicate_index =task 

    weights =GLOBAL_WEIGHTS [replicate_index ]

    clean =clean_metrics_from_cache (
    GLOBAL_CLEAN_CACHES [model ],
    weights ,
    )

    failure =None 

    if model in GLOBAL_FAILURE_CACHES :
        failure =failure_metrics_from_caches (
        GLOBAL_FAILURE_CACHES [model ],
        weights ,
        )

    return (
    model ,
    replicate_index ,
    clean ,
    failure ,
    )


def empirical_two_sided_p (
values :np .ndarray ,
)->float :
    count =len (values )

    nonpositive =(
    np .count_nonzero (values <=0.0 )+1 
    )/(count +1 )

    nonnegative =(
    np .count_nonzero (values >=0.0 )+1 
    )/(count +1 )

    return min (
    1.0 ,
    2.0 *min (nonpositive ,nonnegative ),
    )


def evidence_role (
comparison :str ,
metric :str ,
)->str :
    if comparison =="p2_e20_minus_standard_e20":
        if metric in {"AP_small","AR_small"}:
            return "primary_architecture"
        if metric in {
        "AP",
        "AP75",
        "AP_medium",
        "AP_large",
        }:
            return "prespecified_guardrail"
        return "secondary"

    if comparison .startswith ("gate_"):
        if metric in FAILURE_METRICS :
            return "primary_targeted_recovery"
        if metric in {"AP","AP_small","AP75"}:
            return "prespecified_clean_generalization"
        return "secondary"

    if metric in {"AP","AP_small","AP75"}:
        return "primary_clean_generalization"

    return "secondary"


def apply_bh (rows :list [dict ])->None :
    grouped =defaultdict (list )

    for index ,row in enumerate (rows ):
        if row ["evidence_role"]=="secondary":
            grouped [row ["comparison"]].append (index )

    for indices in grouped .values ():
        ordered =sorted (
        indices ,
        key =lambda index :rows [index ]["p_value"],
        )

        count =len (ordered )
        adjusted =[1.0 ]*count 
        running =1.0 

        for reverse_position in range (
        count -1 ,
        -1 ,
        -1 ,
        ):
            row_index =ordered [reverse_position ]
            rank =reverse_position +1 

            candidate =min (
            1.0 ,
            rows [row_index ]["p_value"]
            *count /rank ,
            )

            running =min (running ,candidate )
            adjusted [reverse_position ]=running 

        for position ,row_index in enumerate (ordered ):
            rows [row_index ]["bh_q_value"]=(
            adjusted [position ]
            )
            rows [row_index ]["fdr_significant_q05"]=(
            adjusted [position ]<0.05 
            )

    for row in rows :
        if "bh_q_value"not in row :
            row ["bh_q_value"]=""
            row ["fdr_significant_q05"]=""


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
    "created_utc":time .strftime (
    "%Y-%m-%dT%H:%M:%SZ",
    time .gmtime (),
    ),
    "artifacts":artifacts ,
    }

    path =directory /"artifact_manifest.json"
    write_json (path ,manifest )

    return path ,sha256_file (path )


def execute (
protocol :dict ,
canonical_rows :dict [str ,dict ],
failure_ids :dict [str ,set [int ]],
):
    global GLOBAL_CLEAN_CACHES 
    global GLOBAL_FAILURE_CACHES 
    global GLOBAL_WEIGHTS 

    started =time .time ()

    staging =OUTPUT .with_name (
    OUTPUT .name 
    +f".incomplete-{int (time .time ())}-{os .getpid ()}"
    )
    staging .mkdir (parents =True )

    gt_data =json .loads (GT .read_text ())

    image_ids =sorted (
    int (image ["id"])
    for image in gt_data ["images"]
    )

    if len (image_ids )!=EXPECTED_IMAGES :
        raise AssertionError (len (image_ids ))

    annotations_by_subset ={
    subset :[
    annotation 
    for annotation in gt_data ["annotations"]
    if int (annotation ["id"])in ids 
    ]
    for subset ,ids in failure_ids .items ()
    }

    subset_datasets ={}

    for subset ,annotations in annotations_by_subset .items ():
        subset_datasets [subset ]={
        "images":gt_data ["images"],
        "categories":gt_data ["categories"],
        "annotations":annotations ,
        "info":gt_data .get ("info",{}),
        "licenses":gt_data .get ("licenses",[]),
        }

    GLOBAL_CLEAN_CACHES ={}
    GLOBAL_FAILURE_CACHES ={}

    print ("=== BUILDING COCO EVALUATION CACHES ===")

    for model ,prediction_path in PREDICTIONS .items ():
        print ("\n"+"="*100 )
        print ("CACHE MODEL:",model )
        print ("="*100 )

        predictions =json .loads (
        prediction_path .read_text ()
        )

        GLOBAL_CLEAN_CACHES [model ]=(
        build_evaluation_cache (
        gt_data ,
        predictions ,
        (
        "all",
        "small",
        "medium",
        "large",
        ),
        )
        )

        original_weights =np .ones (
        EXPECTED_IMAGES ,
        dtype =np .uint16 ,
        )

        reconstructed =clean_metrics_from_cache (
        GLOBAL_CLEAN_CACHES [model ],
        original_weights ,
        )

        for metric_index ,metric in enumerate (
        CLEAN_METRICS 
        ):
            expected =float (
            canonical_rows [model ][metric ]
            )
            actual =float (
            reconstructed [metric_index ]
            )

            if not math .isclose (
            actual ,
            expected ,
            rel_tol =0.0 ,
            abs_tol =2e-10 ,
            ):
                raise AssertionError (
                (
                model ,
                metric ,
                expected ,
                actual ,
                )
                )

        print (
        "PASS canonical reconstruction:",
        model ,
        )

        if model in GATE_MODELS :
            GLOBAL_FAILURE_CACHES [model ]={}

            for subset in FAILURE_SUBSETS :
                print (
                "failure cache:",
                model ,
                subset ,
                )

                GLOBAL_FAILURE_CACHES [model ][
                subset 
                ]=build_evaluation_cache (
                subset_datasets [subset ],
                predictions ,
                ("all",),
                )

        del predictions 
        gc .collect ()

    print ("\n=== GENERATING PAIRED IMAGE WEIGHTS ===")

    rng =np .random .default_rng (SEED )

    GLOBAL_WEIGHTS =np .empty (
    (REPLICATES ,EXPECTED_IMAGES ),
    dtype =np .uint16 ,
    )

    for replicate in range (REPLICATES ):
        sampled =rng .integers (
        0 ,
        EXPECTED_IMAGES ,
        size =EXPECTED_IMAGES ,
        dtype =np .int32 ,
        )

        GLOBAL_WEIGHTS [replicate ]=np .bincount (
        sampled ,
        minlength =EXPECTED_IMAGES ,
        ).astype (np .uint16 )

        if (replicate +1 )%200 ==0 :
            print (
            f"weights={replicate +1 }/{REPLICATES }"
            )

    weights_sha256 =hashlib .sha256 (
    GLOBAL_WEIGHTS .tobytes ()
    ).hexdigest ()

    clean_distributions ={
    model :np .empty (
    (REPLICATES ,len (CLEAN_METRICS )),
    dtype =np .float64 ,
    )
    for model in PREDICTIONS 
    }

    failure_distributions ={
    model :np .empty (
    (REPLICATES ,len (FAILURE_METRICS )),
    dtype =np .float64 ,
    )
    for model in GATE_MODELS 
    }

    tasks =[
    (model ,replicate )
    for model in PREDICTIONS 
    for replicate in range (REPLICATES )
    ]

    workers =min (64 ,os .cpu_count ()or 1 )

    print ("\n=== RUNNING PAIRED BOOTSTRAP ===")
    print ("tasks:",len (tasks ))
    print ("workers:",workers )

    context =mp .get_context ("fork")
    completed =0 

    with context .Pool (
    processes =workers ,
    maxtasksperchild =200 ,
    )as pool :
        for (
        model ,
        replicate ,
        clean ,
        failure ,
        )in pool .imap_unordered (
        bootstrap_worker ,
        tasks ,
        chunksize =1 ,
        ):
            clean_distributions [model ][
            replicate 
            ]=clean 

            if failure is not None :
                failure_distributions [model ][
                replicate 
                ]=failure 

            completed +=1 

            if completed %200 ==0 :
                print (
                f"bootstrap_tasks={completed }/{len (tasks )}",
                flush =True ,
                )

    result_rows =[]

    for comparison ,(
    positive_model ,
    reference_model ,
    )in COMPARISONS .items ():
        metric_distributions ={
        metric :(
        clean_distributions [
        positive_model 
        ][:,metric_index ]
        -clean_distributions [
        reference_model 
        ][:,metric_index ]
        )
        for metric_index ,metric in enumerate (
        CLEAN_METRICS 
        )
        }

        if (
        positive_model in GATE_MODELS 
        and reference_model in GATE_MODELS 
        ):
            for metric_index ,metric in enumerate (
            FAILURE_METRICS 
            ):
                metric_distributions [metric ]=(
                failure_distributions [
                positive_model 
                ][:,metric_index ]
                -failure_distributions [
                reference_model 
                ][:,metric_index ]
                )

        for metric ,distribution in (
        metric_distributions .items ()
        ):
            if metric in CLEAN_METRICS :
                point_estimate =(
                float (
                canonical_rows [
                positive_model 
                ][metric ]
                )
                -float (
                canonical_rows [
                reference_model 
                ][metric ]
                )
                )
            else :
                positive_point =(
                failure_metrics_from_caches (
                GLOBAL_FAILURE_CACHES [
                positive_model 
                ],
                np .ones (
                EXPECTED_IMAGES ,
                dtype =np .uint16 ,
                ),
                )
                )
                reference_point =(
                failure_metrics_from_caches (
                GLOBAL_FAILURE_CACHES [
                reference_model 
                ],
                np .ones (
                EXPECTED_IMAGES ,
                dtype =np .uint16 ,
                ),
                )
                )

                metric_index =FAILURE_METRICS .index (
                metric 
                )
                point_estimate =float (
                positive_point [metric_index ]
                -reference_point [metric_index ]
                )

            lower ,upper =np .quantile (
            distribution ,
            [0.025 ,0.975 ],
            )

            role =evidence_role (
            comparison ,
            metric ,
            )

            result_rows .append ({
            "comparison":comparison ,
            "positive_model":positive_model ,
            "reference_model":reference_model ,
            "metric":metric ,
            "evidence_role":role ,
            "point_estimate":point_estimate ,
            "bootstrap_mean":
            float (np .mean (distribution )),
            "bootstrap_median":
            float (np .median (distribution )),
            "bootstrap_standard_error":
            float (
            np .std (
            distribution ,
            ddof =1 ,
            )
            ),
            "ci95_lower":float (lower ),
            "ci95_upper":float (upper ),
            "ci_crosses_zero":
            bool (lower <=0.0 <=upper ),
            "stable_positive":
            bool (lower >0.0 ),
            "stable_negative":
            bool (upper <0.0 ),
            "p_value":
            empirical_two_sided_p (
            distribution 
            ),
            "replicates":REPLICATES ,
            "seed":SEED ,
            })

    apply_bh (result_rows )

    write_csv (
    staging /"paired_bootstrap_intervals.csv",
    result_rows ,
    )

    npz_payload ={}

    for model ,values in clean_distributions .items ():
        npz_payload [
        f"clean__{model }"
        ]=values 

    for model ,values in failure_distributions .items ():
        npz_payload [
        f"failure__{model }"
        ]=values 

    np .savez_compressed (
    staging /"bootstrap_model_metrics.npz",
    **npz_payload ,
    )

    stable_primary =[
    row for row in result_rows 
    if row ["evidence_role"].startswith ("primary")
    and not row ["ci_crosses_zero"]
    ]

    summary ={
    "version":"paired_bootstrap_v1",
    "status":"complete",
    "protocol_sha256":sha256_file (PROTOCOL ),
    "replicates":REPLICATES ,
    "seed":SEED ,
    "confidence_level":0.95 ,
    "interval":"percentile",
    "resampling_unit":"official_val_image",
    "paired_resampling":True ,
    "weights_sha256":weights_sha256 ,
    "comparisons":list (COMPARISONS ),
    "clean_metrics":list (CLEAN_METRICS ),
    "targeted_recovery_metrics":
    list (FAILURE_METRICS ),
    "stable_primary_results":stable_primary ,
    "interpretation_rule":(
    "Intervals crossing zero are reported as "
    "no stable detected difference. This does "
    "not establish equivalence or proof of no "
    "effect."
    ),
    "scope_warning":(
    "Image bootstrap quantifies validation-image "
    "sampling uncertainty and does not replace "
    "training-seed replication."
    ),
    "elapsed_minutes":
    (time .time ()-started )/60.0 ,
    }

    write_json (
    staging /"summary.json",
    summary ,
    )

    metadata ={
    "version":"paired_bootstrap_v1",
    "status":"complete",
    "created_utc":time .strftime (
    "%Y-%m-%dT%H:%M:%SZ",
    time .gmtime (),
    ),
    "script":str (Path (__file__ ).resolve ()),
    "script_sha256":
    sha256_file (Path (__file__ ).resolve ()),
    "numpy_version":np .__version__ ,
    "logical_cpus":os .cpu_count (),
    "workers":workers ,
    "new_training":False ,
    "new_gpu_inference":False ,
    "method":(
    "Image-cluster bootstrap using multiplicity "
    "weights and complete COCO precision-recall "
    "re-accumulation."
    ),
    }

    write_json (
    staging /"metadata.json",
    metadata ,
    )

    manifest_path ,manifest_sha256 =(
    build_manifest (staging )
    )

    os .replace (staging ,OUTPUT )

    print ("\n"+"="*100 )
    print ("PAIRED BOOTSTRAP COMPLETE")
    print ("="*100 )

    for row in result_rows :
        if row ["evidence_role"]!="secondary":
            print (
            row ["comparison"],
            row ["metric"],
            "delta=",
            round (row ["point_estimate"],6 ),
            "CI95=(",
            round (row ["ci95_lower"],6 ),
            ",",
            round (row ["ci95_upper"],6 ),
            ")",
            "stable_positive=",
            row ["stable_positive"],
            "stable_negative=",
            row ["stable_negative"],
            )

    print ("output:",OUTPUT )
    print (
    "manifest:",
    OUTPUT /manifest_path .name ,
    )
    print ("manifest_sha256:",manifest_sha256 )
    print ("NOTHING TRAINED OR GPU-INFERRED")
    print ("NEXT: exposure-budget audit")


def main ():
    args =parse_args ()

    (
    protocol ,
    canonical_rows ,
    failure_ids ,
    )=preflight ()

    if args .preflight_only :
        print ("\nPASS: Stage-5 preflight only")
        print ("NOTHING CREATED OR RESAMPLED")
        return 

    execute (
    protocol ,
    canonical_rows ,
    failure_ids ,
    )


if __name__ =="__main__":
    main ()
