from __future__ import annotations 

from collections import defaultdict 
from pathlib import Path 
import argparse 
import csv 
import hashlib 
import json 
import os 

import numpy as np 


ROOT =Path (
"/root/rivermind-data/autodrive/results/evaluation"
)
METHOD_ROOT =ROOT /"method_gate"

PROTOCOL =(
METHOD_ROOT /"frozen_manifests/method_gate_protocol_v1.json"
)
RISK_TABLE =(
ROOT /"train_risk_labels/object_risk_v1/object_risk_table.csv"
)
RISK_FREEZE =(
ROOT /"train_risk_labels/frozen_manifests/"
"object_risk_v1_freeze.json"
)
GT_PATH =(
ROOT /"architecture_gate/prepared_inputs_v3/"
"instances_train_dev_effective.json"
)
CLEAN_ROOT =METHOD_ROOT /"evaluation_v2_clean"
CLEAN_MANIFEST =CLEAN_ROOT /"clean_artifact_manifest.json"
CLEAN_COMPARISON =CLEAN_ROOT /"clean_comparison.json"

OUTPUT =METHOD_ROOT /"evaluation_v3_failures"

EXPECTED ={
PROTOCOL :
"6be7ae25550360e38195bb785a064bd19cb9c4a65b4625b30b5698861cc1b00a",
RISK_TABLE :
"83e03b29ff5b796257fed1a3fb284fc22597e6e748315169e25fa481f9b1ba38",
RISK_FREEZE :
"b5e1934ee9108e29548288e6f98cc611bb424c2f6d5b0ab99329b490699ed9d6",
GT_PATH :
"5e13ec651d0582032e5eacc1a46d885279fabe278463bd7c2be58b2429796faf",
CLEAN_MANIFEST :
"29dc8882d383c37c6e9eb5a8cf826b9d562a3c48e9d0ef8d5a23be5b78350f97",
}

PREDICTIONS ={
name :(
CLEAN_ROOT /"evaluations"
/f"{name }_epoch10/predictions.json"
)
for name in ("uniform","scalar_risk","typed_cafr")
}

GROUPS =(
"small_joint_not_detected",
"dfine_supported_yolo_failure",
"yolo_geometric_failure",
"yolo_low_confidence",
"yolo_classification_failure",
)

THRESHOLDS =np .arange (0.50 ,0.96 ,0.05 )
EXCLUDED_ANNOTATION_ID =1005741 
EXCLUDED_CATEGORY_ID =6 
BOOTSTRAP_RESAMPLES =1000 
SEED =20260809 


def sha256 (path :Path )->str :
    digest =hashlib .sha256 ()
    with path .open ("rb")as handle :
        while block :=handle .read (8 *1024 *1024 ):
            digest .update (block )
    return digest .hexdigest ()


def verify (path :Path ,expected :str )->None :
    if not path .is_file ():
        raise FileNotFoundError (path )

    actual =sha256 (path )
    print (path )
    print (" expected:",expected )
    print (" actual  :",actual )
    if actual !=expected :
        raise AssertionError (path )


def as_bool (value :str )->bool :
    return value .strip ().lower ()in {"1","true","yes","y"}


def primary_group (row :dict [str ,str ])->str :
    small =row ["scale"].strip ().lower ()=="small"
    tier =row ["risk_tier"].strip ().lower ()
    status =row ["yolo_status"].strip ().lower ()
    yolo =as_bool (row ["yolo_detected"])
    dfine =as_bool (row ["dfine_detected"])

    if small and not yolo and not dfine :
        return "small_joint_not_detected"
    if dfine and not yolo :
        return "dfine_supported_yolo_failure"
    if status =="low_confidence_candidate":
        return "yolo_low_confidence"
    if small and tier in {"high","critical"}:
        return "small_high_or_critical"
    if status in {
    "localization_error_candidate",
    "assignment_conflict_candidate",
    }:
        return "yolo_geometric_failure"
    if status =="classification_error_candidate":
        return "yolo_classification_failure"
    if not yolo and not dfine :
        return "other_joint_not_detected"
    if tier in {"high","critical"}:
        return "other_high_or_critical"
    return "standard_support"


def iou_matrix (predictions ,ground_truth ):
    p =np .asarray (predictions ,dtype =np .float64 )
    g =np .asarray (ground_truth ,dtype =np .float64 )

    px1 ,py1 =p [:,0 ],p [:,1 ]
    px2 ,py2 =px1 +p [:,2 ],py1 +p [:,3 ]
    gx1 ,gy1 =g [:,0 ],g [:,1 ]
    gx2 ,gy2 =gx1 +g [:,2 ],gy1 +g [:,3 ]

    ix1 =np .maximum (px1 [:,None ],gx1 [None ,:])
    iy1 =np .maximum (py1 [:,None ],gy1 [None ,:])
    ix2 =np .minimum (px2 [:,None ],gx2 [None ,:])
    iy2 =np .minimum (py2 [:,None ],gy2 [None ,:])

    intersection =(
    np .maximum (0.0 ,ix2 -ix1 )
    *np .maximum (0.0 ,iy2 -iy1 )
    )
    p_area =np .maximum (0.0 ,px2 -px1 )*np .maximum (0.0 ,py2 -py1 )
    g_area =np .maximum (0.0 ,gx2 -gx1 )*np .maximum (0.0 ,gy2 -gy1 )

    return intersection /np .maximum (
    p_area [:,None ]+g_area [None ,:]-intersection ,
    1e-12 ,
    )


def match_counts (predictions ,ground_truth ):
    if not ground_truth :
        return np .zeros (len (THRESHOLDS ),dtype =np .int16 )
    if not predictions :
        return np .zeros (len (THRESHOLDS ),dtype =np .int16 )

    matrix =iou_matrix (predictions ,ground_truth )
    output =np .zeros (len (THRESHOLDS ),dtype =np .int16 )

    for threshold_index ,threshold in enumerate (THRESHOLDS ):
        used =np .zeros (len (ground_truth ),dtype =bool )
        matched =0 

        for prediction_index in range (len (predictions )):
            candidates =np .argsort (
            matrix [prediction_index ],
            kind ="stable",
            )[::-1 ]

            for gt_index in candidates :
                if used [gt_index ]:
                    continue 
                if matrix [prediction_index ,gt_index ]<threshold :
                    break 

                used [gt_index ]=True 
                matched +=1 
                break 

        output [threshold_index ]=matched 

    return output 


def metric_summary (gt_counts ,matched_counts ):
    gt_by_class =gt_counts .sum (axis =0 )
    matched_by_class_threshold =matched_counts .sum (axis =0 )

    valid =gt_by_class >0 
    recalls =(
    matched_by_class_threshold [valid ]
    /gt_by_class [valid ,None ]
    )

    per_class ={}
    for class_index in np .flatnonzero (valid ):
        class_recalls =(
        matched_by_class_threshold [class_index ]
        /gt_by_class [class_index ]
        )
        per_class [str (class_index +1 )]={
        "objects":int (gt_by_class [class_index ]),
        "AR100":float (class_recalls .mean ()),
        "recall_IoU50_maxDet100":float (class_recalls [0 ]),
        }

    return {
    "objects":int (gt_by_class .sum ()),
    "categories_with_gt":int (valid .sum ()),
    "subset_AR100":float (recalls .mean ()),
    "subset_recall_IoU50_maxDet100":float (
    matched_by_class_threshold [:,0 ].sum ()
    /gt_by_class .sum ()
    ),
    "macro_recall_IoU50_maxDet100":float (
    recalls [:,0 ].mean ()
    ),
    "per_class":per_class ,
    }


def preflight ():
    print ("=== FAILURE-SUBSET EVALUATION PREFLIGHT ===")

    for path ,expected in EXPECTED .items ():
        verify (path ,expected )

    clean_manifest =json .loads (CLEAN_MANIFEST .read_text ())
    assert clean_manifest ["status"]=="clean_evaluation_complete"

    for relative ,record in clean_manifest ["files"].items ():
        path =CLEAN_ROOT /relative 
        verify (path ,record ["sha256"])
        assert path .stat ().st_size ==int (record ["bytes"])

    for path in PREDICTIONS .values ():
        assert path .is_file (),path 

    protocol =json .loads (PROTOCOL .read_text ())
    assert tuple (
    protocol ["evaluation"]["core_failure_groups"]
    )==GROUPS 
    assert protocol ["evaluation"]["images"]==5000 
    assert protocol ["evaluation"]["effective_gt_objects"]==92392 
    assert (
    protocol ["evaluation"]["paired_bootstrap"]["resamples"]
    ==BOOTSTRAP_RESAMPLES 
    )
    assert (
    protocol ["evaluation"]["paired_bootstrap"]["seed"]
    ==SEED 
    )

    if OUTPUT .exists ():
        raise FileExistsError (
        f"Refusing to overwrite: {OUTPUT }"
        )

    incomplete =list (
    OUTPUT .parent .glob (f"{OUTPUT .name }.incomplete-*")
    )
    if incomplete :
        raise FileExistsError (incomplete )

    print ("PASS: frozen risk labels, clean predictions and rules are valid")
    print ("NOTHING EVALUATED OR CREATED")


def execute ():
    preflight ()

    staging =OUTPUT .with_name (
    f"{OUTPUT .name }.incomplete-{os .getpid ()}"
    )
    staging .mkdir (parents =False ,exist_ok =False )

    gt =json .loads (GT_PATH .read_text ())
    images =sorted (int (row ["id"])for row in gt ["images"])
    image_index ={
    image_id :index 
    for index ,image_id in enumerate (images )
    }

    annotations ={
    int (row ["id"]):row 
    for row in gt ["annotations"]
    }
    assert len (annotations )==92392 
    assert all (
    int (row .get ("iscrowd",0 ))==0 
    for row in annotations .values ()
    )

    membership ={}
    dev_annotation_ids =set ()

    print ("=== BUILDING FROZEN FAILURE MEMBERSHIP ===")
    with RISK_TABLE .open (newline ="",encoding ="utf-8")as handle :
        reader =csv .DictReader (handle )

        for row_index ,row in enumerate (reader ,start =1 ):
            if row ["split"]!="train_dev":
                continue 

            annotation_id =int (row ["annotation_id"])
            if annotation_id ==EXCLUDED_ANNOTATION_ID :
                continue 

            dev_annotation_ids .add (annotation_id )
            assert annotation_id in annotations 

            category_id =int (row ["category_id"])
            if category_id ==EXCLUDED_CATEGORY_ID :
                continue 

            group =primary_group (row )
            if group in GROUPS :
                membership [annotation_id ]=group 

            if row_index %250000 ==0 :
                print (f"risk_rows={row_index }")

    assert dev_annotation_ids ==set (annotations )

    gt_boxes ={
    group :defaultdict (list )
    for group in GROUPS 
    }
    gt_counts ={
    group :np .zeros (
    (len (images ),10 ),
    dtype =np .int16 ,
    )
    for group in GROUPS 
    }

    membership_rows =[]
    for annotation_id ,group in sorted (membership .items ()):
        annotation =annotations [annotation_id ]
        image_id =int (annotation ["image_id"])
        category_id =int (annotation ["category_id"])

        gt_boxes [group ][(image_id ,category_id )].append (
        (annotation_id ,list (map (float ,annotation ["bbox"])))
        )
        gt_counts [group ][image_index [image_id ],category_id -1 ]+=1 

        membership_rows .append ({
        "annotation_id":annotation_id ,
        "image_id":image_id ,
        "category_id":category_id ,
        "failure_group":group ,
        })

    membership_path =staging /"failure_membership.csv"
    with membership_path .open (
    "x",
    newline ="",
    encoding ="utf-8",
    )as handle :
        writer =csv .DictWriter (
        handle ,
        fieldnames =[
        "annotation_id",
        "image_id",
        "category_id",
        "failure_group",
        ],
        )
        writer .writeheader ()
        writer .writerows (membership_rows )

    print ("group_counts:",{
    group :int (gt_counts [group ].sum ())
    for group in GROUPS 
    })

    matched ={
    model :{
    group :np .zeros (
    (len (images ),10 ,len (THRESHOLDS )),
    dtype =np .int16 ,
    )
    for group in GROUPS 
    }
    for model in PREDICTIONS 
    }

    target_keys =set ()
    for group in GROUPS :
        target_keys .update (gt_boxes [group ])

    for model_name ,prediction_path in PREDICTIONS .items ():
        print ("\n"+"="*88 )
        print ("MATCHING:",model_name )
        print ("="*88 )

        prediction_data =json .loads (prediction_path .read_text ())
        predictions_by_key =defaultdict (list )

        for sequence ,row in enumerate (prediction_data ):
            key =(
            int (row ["image_id"]),
            int (row ["category_id"]),
            )
            if key not in target_keys :
                continue 

            predictions_by_key [key ].append ((
            -float (row ["score"]),
            sequence ,
            list (map (float ,row ["bbox"])),
            ))

        for rows in predictions_by_key .values ():
            rows .sort (key =lambda item :(item [0 ],item [1 ]))

        processed =0 
        for group in GROUPS :
            for (image_id ,category_id ),targets in gt_boxes [group ].items ():
                targets .sort (key =lambda item :item [0 ])
                prediction_boxes =[
                item [2 ]
                for item in predictions_by_key .get (
                (image_id ,category_id ),
                [],
                )[:100 ]
                ]
                target_boxes =[
                item [1 ]
                for item in targets 
                ]

                matched [model_name ][group ][
                image_index [image_id ],
                category_id -1 ,
                ]=match_counts (
                prediction_boxes ,
                target_boxes ,
                )
                processed +=1 

                if processed %5000 ==0 :
                    print (f"matched_keys={processed }")

        del prediction_data 
        del predictions_by_key 

    metrics ={
    model :{
    group :metric_summary (
    gt_counts [group ],
    matched [model ][group ],
    )
    for group in GROUPS 
    }
    for model in PREDICTIONS 
    }

    for model in metrics :
        metrics [model ]["failure_macro_AR100"]=float (
        np .mean ([
        metrics [model ][group ]["subset_AR100"]
        for group in GROUPS 
        ])
        )

    comparisons ={}
    for left ,right in (
    ("scalar_risk","uniform"),
    ("typed_cafr","uniform"),
    ("typed_cafr","scalar_risk"),
    ):
        row ={
        "delta_failure_macro_AR100":(
        metrics [left ]["failure_macro_AR100"]
        -metrics [right ]["failure_macro_AR100"]
        )
        }

        for group in GROUPS :
            row [f"delta_{group }_AR100"]=(
            metrics [left ][group ]["subset_AR100"]
            -metrics [right ][group ]["subset_AR100"]
            )
            row [f"delta_{group }_recall50"]=(
            metrics [left ][group ][
            "subset_recall_IoU50_maxDet100"
            ]
            -metrics [right ][group ][
            "subset_recall_IoU50_maxDet100"
            ]
            )

        comparisons [f"{left }_minus_{right }"]=row 

    clean =json .loads (CLEAN_COMPARISON .read_text ())
    clean_metrics =clean ["metrics"]

    typed_uniform =comparisons ["typed_cafr_minus_uniform"]
    typed_scalar =comparisons ["typed_cafr_minus_scalar_risk"]

    clean_delta_typed_uniform =(
    clean_metrics ["typed_cafr"]["AP"]
    -clean_metrics ["uniform"]["AP"]
    )
    clean_ap75_delta =(
    clean_metrics ["typed_cafr"]["AP75"]
    -clean_metrics ["uniform"]["AP75"]
    )
    clean_large_delta =(
    clean_metrics ["typed_cafr"]["AP_large"]
    -clean_metrics ["uniform"]["AP_large"]
    )
    clean_small_delta =(
    clean_metrics ["typed_cafr"]["AP_small"]
    -clean_metrics ["uniform"]["AP_small"]
    )
    clean_typed_scalar =(
    clean_metrics ["typed_cafr"]["AP"]
    -clean_metrics ["scalar_risk"]["AP"]
    )

    rule_checks ={
    "clean_guardrails":{
    "delta_AP":clean_delta_typed_uniform >=-0.002 ,
    "delta_AP75":clean_ap75_delta >=-0.003 ,
    "delta_AP_large":clean_large_delta >=-0.005 ,
    },
    "failure_progress":{
    "delta_AP_small":clean_small_delta >=0.002 ,
    "delta_failure_macro_AR100":(
    typed_uniform ["delta_failure_macro_AR100"]>=0.005 
    ),
    "delta_small_joint_recall50":(
    typed_uniform [
    "delta_small_joint_not_detected_recall50"
    ]>=0.005 
    ),
    },
    "typed_advantage_over_scalar":{
    "clean_delta_AP":clean_typed_scalar >=-0.002 ,
    "failure_rule":(
    typed_scalar ["delta_failure_macro_AR100"]>=0.002 
    or (
    typed_scalar [
    "delta_small_joint_not_detected_recall50"
    ]>=0.003 
    and typed_scalar [
    "delta_yolo_geometric_failure_AR100"
    ]>=0.0 
    )
    ),
    },
    }


    rng =np .random .default_rng (SEED )
    bootstrap ={
    "typed_minus_uniform_failure_macro_micro_AR100":[],
    "typed_minus_scalar_failure_macro_micro_AR100":[],
    }

    gt_per_image ={
    group :gt_counts [group ].sum (axis =1 ).astype (np .float64 )
    for group in GROUPS 
    }
    matched_per_image ={
    model :{
    group :matched [model ][group ].sum (axis =1 ).astype (np .float64 )
    for group in GROUPS 
    }
    for model in PREDICTIONS 
    }

    for _ in range (BOOTSTRAP_RESAMPLES ):
        indices =rng .integers (
        0 ,
        len (images ),
        size =len (images ),
        )

        model_macro ={}
        for model in PREDICTIONS :
            group_values =[]

            for group in GROUPS :
                denominator =gt_per_image [group ][indices ].sum ()
                numerator =(
                matched_per_image [model ][group ][indices ]
                .sum (axis =0 )
                )
                group_values .append (
                float ((numerator /denominator ).mean ())
                )

            model_macro [model ]=float (np .mean (group_values ))

        bootstrap [
        "typed_minus_uniform_failure_macro_micro_AR100"
        ].append (
        model_macro ["typed_cafr"]-model_macro ["uniform"]
        )
        bootstrap [
        "typed_minus_scalar_failure_macro_micro_AR100"
        ].append (
        model_macro ["typed_cafr"]-model_macro ["scalar_risk"]
        )

    bootstrap_summary ={}
    for key ,values in bootstrap .items ():
        array =np .asarray (values )
        bootstrap_summary [key ]={
        "estimate_mean":float (array .mean ()),
        "ci95_low":float (np .quantile (array ,0.025 )),
        "ci95_high":float (np .quantile (array ,0.975 )),
        }

    result ={
    "version":"method_gate_failure_evaluation_v1",
    "status":"failure_evaluation_complete",
    "group_definition":(
    "primary_failure_group from frozen sampling audit v3"
    ),
    "excluded_annotation_id":EXCLUDED_ANNOTATION_ID ,
    "excluded_target_category_id":EXCLUDED_CATEGORY_ID ,
    "thresholds":THRESHOLDS .tolist (),
    "max_detections_per_image_category":100 ,
    "metrics":metrics ,
    "comparisons":comparisons ,
    "rule_checks":rule_checks ,
    "bootstrap_diagnostic":{
    "resamples":BOOTSTRAP_RESAMPLES ,
    "seed":SEED ,
    "role":"reported diagnostic; not a gate threshold",
    "summary":bootstrap_summary ,
    },
    "corruption_evaluation_pending":True ,
    "full_gate_decision_allowed":False ,
    "frozen_gate_already_blocked_by_AP_small":(
    not rule_checks ["failure_progress"]["delta_AP_small"]
    ),
    }

    result_path =staging /"failure_evaluation.json"
    result_path .write_text (
    json .dumps (
    result ,
    ensure_ascii =False ,
    indent =2 ,
    sort_keys =True ,
    )+"\n"
    )

    arrays ={"image_ids":np .asarray (images )}
    for group in GROUPS :
        arrays [f"gt_{group }"]=gt_counts [group ]
        for model in PREDICTIONS :
            arrays [f"match_{model }_{group }"]=matched [model ][group ]

    np .savez_compressed (
    staging /"paired_failure_counts.npz",
    **arrays ,
    )

    files ={}
    for path in sorted (staging .iterdir ()):
        if path .is_file ():
            files [path .name ]={
            "bytes":path .stat ().st_size ,
            "sha256":sha256 (path ),
            }

    manifest ={
    "version":"method_gate_failure_artifacts_v1",
    "status":"failure_evaluation_complete",
    "files":files ,
    }
    manifest_path =staging /"artifact_manifest.json"
    manifest_path .write_text (
    json .dumps (
    manifest ,
    ensure_ascii =False ,
    indent =2 ,
    sort_keys =True ,
    )+"\n"
    )

    os .replace (staging ,OUTPUT )

    print ("\n"+"="*100 )
    print ("METHOD-GATE FAILURE EVALUATION COMPLETE")
    print ("="*100 )

    for model in PREDICTIONS :
        print (
        model ,
        "failure_macro_AR100=",
        round (metrics [model ]["failure_macro_AR100"],6 ),
        "small_joint_recall50=",
        round (
        metrics [model ]["small_joint_not_detected"][
        "subset_recall_IoU50_maxDet100"
        ],
        6 ,
        ),
        "geometric_AR100=",
        round (
        metrics [model ]["yolo_geometric_failure"][
        "subset_AR100"
        ],
        6 ,
        ),
        )

    print (json .dumps (
    comparisons ,
    ensure_ascii =False ,
    indent =2 ,
    sort_keys =True ,
    ))
    print ("rule_checks:",json .dumps (
    rule_checks ,
    ensure_ascii =False ,
    indent =2 ,
    sort_keys =True ,
    ))
    print ("output:",OUTPUT )
    print (
    "manifest_sha256:",
    sha256 (OUTPUT /"artifact_manifest.json"),
    )
    print ("NEXT: frozen corruption evaluation")


def main ():
    parser =argparse .ArgumentParser ()
    parser .add_argument ("--execute",action ="store_true")
    args =parser .parse_args ()

    if args .execute :
        execute ()
    else :
        preflight ()


if __name__ =="__main__":
    main ()
