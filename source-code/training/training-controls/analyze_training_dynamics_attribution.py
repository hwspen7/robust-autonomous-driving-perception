from __future__ import annotations 

import argparse 
import csv 
import hashlib 
import json 
import math 
import os 
import time 
from collections import defaultdict 
from pathlib import Path 

import numpy as np 


FINAL_ROOT =Path (
"/root/rivermind-data/autodrive/results/evaluation/"
"final_analysis_v1"
)

FORMAL_ROOT =Path (
"/root/rivermind-data/autodrive/results/evaluation/"
"formal_training"
)

PROTOCOL =(
FINAL_ROOT 
/"frozen_protocol/final_testing_protocol_v1.json"
)

TRAJECTORY_ROOT =(
FORMAL_ROOT /"trajectory_evaluation_v1"
)

TRAJECTORY_MANIFEST =(
TRAJECTORY_ROOT /"artifact_manifest.json"
)
TRAJECTORY_METRICS =(
TRAJECTORY_ROOT /"trajectory_metrics.csv"
)
MATCHED_EXISTING =(
TRAJECTORY_ROOT /"matched_rebu_minus_uniform.csv"
)
TRAJECTORY_PROTOCOL =(
TRAJECTORY_ROOT /"trajectory_protocol.json"
)
TRAJECTORY_SUMMARY =(
TRAJECTORY_ROOT /"trajectory_summary.json"
)

COMMON_MANIFEST =(
FORMAL_ROOT 
/"training_v1/common_uniform_e1_40/"
"artifact_manifest.json"
)
FORMAL_BRANCH_MANIFEST =(
FORMAL_ROOT 
/"training_v2/formal_branches_manifest.json"
)

RAW_RESULTS ={
"common":(
FORMAL_ROOT 
/"training_v1/common_uniform_e1_40/"
"results.csv"
),
"uniform":(
FORMAL_ROOT 
/"training_v2/uniform_e41_100/"
"results.csv"
),
"rebu_risk":(
FORMAL_ROOT 
/"training_v2/rebu_risk_e41_100/"
"results.csv"
),
}

OUTPUT =(
FINAL_ROOT 
/"training_dynamics_attribution_v1"
)

EXPECTED_SHA256 ={
PROTOCOL :
"693c39375cc1eefc4f3ff9f957a8355fb947552a19dfef228cae6cfc21e3dcfe",
TRAJECTORY_MANIFEST :
"922f27c7b573e500f449809591af7e7886449b62e9f662ba50609ac1db695485",
TRAJECTORY_METRICS :
"79819bc2b7d5e4a033ed4371885d4a69f6e2641b20e8e42855e42d4854c36057",
MATCHED_EXISTING :
"52b5a63e7996dee1364c658c942623f622461054955e4e6c3a89dcf14b364523",
TRAJECTORY_PROTOCOL :
"7dc023019c6c6945f837a1608847ec67f80efa4e6b505b0d11a48d2963e68afb",
TRAJECTORY_SUMMARY :
"b0ee1f494b315fcda084b5ac994a68ef7acaf52848333a023f6d5aaad978fab7",
COMMON_MANIFEST :
"5fff0d57aa336cf36074cf8da3aff1b3c0d58a169f5bc924d5ebaf978a6f77b7",
FORMAL_BRANCH_MANIFEST :
"64dd9fdfc15db3fae8881a63f86254899187f6f6570b6bfa89129799bb589cd0",
RAW_RESULTS ["common"]:
"08146eede95380b4c47a9592622cf04286905ee27b7a278596ac8991a6c1dd64",
}

METRICS =(
"AP",
"AP50",
"AP75",
"AP_small",
"AP_medium",
"AP_large",
"AR100",
"AR_small",
)

PRIMARY_METRICS =(
"AP",
"AP_small",
"AP75",
)

EXPECTED_MATCHED_EPOCHS =(
41 ,46 ,51 ,56 ,61 ,66 ,71 ,
76 ,81 ,86 ,91 ,96 ,100 ,
)


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


def verify (path :Path ,expected :str )->None :
    if not path .is_file ():
        raise FileNotFoundError (path )

    actual =sha256_file (path )

    print (path )
    print (" expected:",expected )
    print (" actual  :",actual )

    if actual !=expected :
        raise AssertionError (path )


def load_csv (path :Path )->list [dict ]:
    with path .open (newline ="")as handle :
        return list (csv .DictReader (handle ))


def numeric_row (row :dict )->dict :
    result =dict (row )

    for key ,value in row .items ():
        try :
            result [key ]=float (value )
        except (TypeError ,ValueError ):
            pass 

    return result 


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


def normalized_auc (
epochs :list [int ],
values :list [float ],
)->float :
    if len (epochs )<2 :
        return float (values [0 ])

    area =float (np .trapz (
    np .asarray (values ,dtype =np .float64 ),
    np .asarray (epochs ,dtype =np .float64 ),
    ))

    span =float (epochs [-1 ]-epochs [0 ])

    if span <=0 :
        raise ValueError (epochs )

    return area /span 


def linear_slope (values :list [float ])->float :
    if len (values )<2 :
        return 0.0 

    x =np .arange (len (values ),dtype =np .float64 )
    y =np .asarray (values ,dtype =np .float64 )

    return float (np .polyfit (x ,y ,1 )[0 ])


def training_loss_summary (
rows :list [dict ],
)->dict :
    fields =(
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    )

    result ={
    "epochs":len (rows ),
    "first_epoch":int (float (rows [0 ]["epoch"])),
    "last_epoch":int (float (rows [-1 ]["epoch"])),
    "losses":{},
    }

    for field in fields :
        values =[
        float (row [field ])
        for row in rows 
        ]

        result ["losses"][field ]={
        "first":values [0 ],
        "last":values [-1 ],
        "first5_mean":float (
        np .mean (values [:5 ])
        ),
        "last5_mean":float (
        np .mean (values [-5 :])
        ),
        "last10_slope":linear_slope (
        values [-10 :]
        ),
        }

    total =[
    sum (float (row [field ])for field in fields )
    for row in rows 
    ]

    result ["total_train_loss"]={
    "first":total [0 ],
    "last":total [-1 ],
    "first5_mean":float (np .mean (total [:5 ])),
    "last5_mean":float (np .mean (total [-5 :])),
    "last10_slope":linear_slope (total [-10 :]),
    }

    result ["learning_rate"]={
    field :{
    "first":float (rows [0 ][field ]),
    "last":float (rows [-1 ][field ]),
    }
    for field in (
    "lr/pg0",
    "lr/pg1",
    "lr/pg2",
    )
    }

    result ["final_validation_losses"]={
    field :float (rows [-1 ][field ])
    for field in (
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    )
    }

    nonzero_validation_rows =0 

    for row in rows :
        values =[
        float (row [field ])
        for field in (
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
        )
        ]
        if any (value !=0.0 for value in values ):
            nonzero_validation_rows +=1 

    result ["nonzero_validation_rows"]=(
    nonzero_validation_rows 
    )

    return result 


def preflight ():
    print ("=== TRAINING-DYNAMICS ATTRIBUTION PREFLIGHT ===")

    for path ,expected in EXPECTED_SHA256 .items ():
        verify (path ,expected )

    for label ,path in RAW_RESULTS .items ():
        if not path .is_file ():
            raise FileNotFoundError (path )

        print (
        "PASS raw results:",
        label ,
        path ,
        "sha256=",
        sha256_file (path ),
        )

    protocol =json .loads (PROTOCOL .read_text ())
    trajectory_rules =protocol ["trajectory_analysis"]

    required_metrics =set (
    protocol ["primary_evidence"][
    "long_horizon_persistence"
    ]["metrics"]
    )

    expected_required ={
    "matched_epoch_delta_AP",
    "matched_epoch_delta_AP_small",
    "matched_epoch_delta_AP75",
    "trajectory_mean",
    "trajectory_AUC",
    "positive_delta_checkpoint_fraction",
    }

    if required_metrics !=expected_required :
        raise AssertionError (required_metrics )

    rows =[
    numeric_row (row )
    for row in load_csv (TRAJECTORY_METRICS )
    ]

    if len (rows )!=27 :
        raise AssertionError (len (rows ))

    keys =[str (row ["key"])for row in rows ]

    if len (set (keys ))!=27 :
        raise AssertionError ("Duplicate trajectory key.")

    if OUTPUT .exists ():
        raise FileExistsError (OUTPUT )

    incomplete =sorted (
    OUTPUT .parent .glob (
    OUTPUT .name +".incomplete-*"
    )
    )

    if incomplete :
        raise AssertionError (incomplete )

    print ("trajectory_nodes:",len (rows ))
    print ("trajectory_rules:")
    print (json .dumps (
    trajectory_rules ,
    indent =2 ,
    ensure_ascii =False ,
    sort_keys =True ,
    ))
    print ("PASS: Stage-4 rules and inputs are valid")

    return protocol ,rows 


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
trajectory_rows :list [dict ],
):
    started =time .time ()

    staging =OUTPUT .with_name (
    OUTPUT .name 
    +f".incomplete-{int (time .time ())}-{os .getpid ()}"
    )
    staging .mkdir (parents =True )

    common_rows =[
    row for row in trajectory_rows 
    if str (row ["arm"])=="common"
    ]
    uniform_rows =[
    row for row in trajectory_rows 
    if str (row ["arm"])=="uniform"
    ]
    risk_rows =[
    row for row in trajectory_rows 
    if str (row ["arm"])=="rebu_risk"
    ]

    if len (common_rows )!=1 :
        raise AssertionError (len (common_rows ))
    if len (uniform_rows )!=13 :
        raise AssertionError (len (uniform_rows ))
    if len (risk_rows )!=13 :
        raise AssertionError (len (risk_rows ))

    common =common_rows [0 ]

    uniform_by_epoch ={
    int (row ["global_epoch"]):row 
    for row in uniform_rows 
    }
    risk_by_epoch ={
    int (row ["global_epoch"]):row 
    for row in risk_rows 
    }

    matched_epochs =tuple (sorted (
    set (uniform_by_epoch )
    &set (risk_by_epoch )
    ))

    if matched_epochs !=EXPECTED_MATCHED_EPOCHS :
        raise AssertionError (matched_epochs )

    attribution_rows =[]

    for epoch in matched_epochs :
        uniform =uniform_by_epoch [epoch ]
        risk =risk_by_epoch [epoch ]

        row ={
        "global_epoch":epoch ,
        "uniform_key":uniform ["key"],
        "risk_key":risk ["key"],
        }

        for metric in METRICS :
            common_value =float (common [metric ])
            uniform_value =float (uniform [metric ])
            risk_value =float (risk [metric ])

            shared_schedule_effect =(
            uniform_value -common_value 
            )
            risk_specific_residual =(
            risk_value -uniform_value 
            )
            total_risk_effect =(
            risk_value -common_value 
            )

            identity_error =abs (
            total_risk_effect 
            -(
            shared_schedule_effect 
            +risk_specific_residual 
            )
            )

            if identity_error >1e-12 :
                raise AssertionError (
                (epoch ,metric ,identity_error )
                )

            row [f"common_{metric }"]=common_value 
            row [f"uniform_{metric }"]=uniform_value 
            row [f"risk_{metric }"]=risk_value 
            row [
            f"shared_schedule_effect_{metric }"
            ]=shared_schedule_effect 
            row [
            f"risk_specific_residual_{metric }"
            ]=risk_specific_residual 
            row [
            f"total_risk_effect_{metric }"
            ]=total_risk_effect 

        attribution_rows .append (row )

    aggregate_rows =[]

    for metric in METRICS :
        epochs =list (matched_epochs )

        uniform_values =[
        float (uniform_by_epoch [epoch ][metric ])
        for epoch in epochs 
        ]
        risk_values =[
        float (risk_by_epoch [epoch ][metric ])
        for epoch in epochs 
        ]
        deltas =[
        risk -uniform 
        for risk ,uniform in zip (
        risk_values ,
        uniform_values ,
        )
        ]

        aggregate_rows .append ({
        "metric":metric ,
        "common_value":float (common [metric ]),
        "uniform_trajectory_mean":
        float (np .mean (uniform_values )),
        "risk_trajectory_mean":
        float (np .mean (risk_values )),
        "delta_trajectory_mean":
        float (np .mean (deltas )),
        "uniform_normalized_AUC":
        normalized_auc (
        epochs ,
        uniform_values ,
        ),
        "risk_normalized_AUC":
        normalized_auc (
        epochs ,
        risk_values ,
        ),
        "delta_normalized_AUC":
        normalized_auc (
        epochs ,
        deltas ,
        ),
        "positive_delta_checkpoints":sum (
        value >0.0 for value in deltas 
        ),
        "zero_delta_checkpoints":sum (
        math .isclose (
        value ,
        0.0 ,
        abs_tol =1e-12 ,
        )
        for value in deltas 
        ),
        "negative_delta_checkpoints":sum (
        value <0.0 for value in deltas 
        ),
        "positive_delta_checkpoint_fraction":
        sum (value >0.0 for value in deltas )
        /len (deltas ),
        "minimum_delta":min (deltas ),
        "maximum_delta":max (deltas ),
        "terminal_delta":deltas [-1 ],
        })

    raw_rows ={
    label :load_csv (path )
    for label ,path in RAW_RESULTS .items ()
    }

    loss_audit ={
    label :training_loss_summary (rows )
    for label ,rows in raw_rows .items ()
    }

    common_final_lr =(
    loss_audit ["common"]["learning_rate"]
    )
    uniform_first_lr =(
    loss_audit ["uniform"]["learning_rate"]
    )
    risk_first_lr =(
    loss_audit ["rebu_risk"]["learning_rate"]
    )

    restart_ratios ={}

    for group in (
    "lr/pg0",
    "lr/pg1",
    "lr/pg2",
    ):
        denominator =common_final_lr [group ]["last"]

        restart_ratios [group ]={
        "uniform_branch_restart_ratio":
        uniform_first_lr [group ]["first"]
        /denominator ,
        "risk_branch_restart_ratio":
        risk_first_lr [group ]["first"]
        /denominator ,
        }

    trajectory_summary =json .loads (
    TRAJECTORY_SUMMARY .read_text ()
    )

    aggregate_by_metric ={
    row ["metric"]:row 
    for row in aggregate_rows 
    }

    terminal =attribution_rows [-1 ]

    summary ={
    "version":
    "training_dynamics_attribution_v1",
    "status":"complete",
    "protocol_sha256":
    sha256_file (PROTOCOL ),
    "trajectory_manifest_sha256":
    sha256_file (TRAJECTORY_MANIFEST ),
    "matched_epochs":list (matched_epochs ),
    "common_anchor":{
    metric :float (common [metric ])
    for metric in METRICS 
    },
    "frozen_decline_records":{
    "uniform":
    trajectory_summary [
    "uniform_decline"
    ],
    "rebu_risk":
    trajectory_summary [
    "rebu_decline"
    ],
    },
    "learning_rate_restart_ratios":
    restart_ratios ,
    "terminal_epoch_100_attribution":{
    metric :{
    "shared_schedule_effect":
    terminal [
    f"shared_schedule_effect_{metric }"
    ],
    "risk_specific_residual":
    terminal [
    f"risk_specific_residual_{metric }"
    ],
    "total_risk_effect":
    terminal [
    f"total_risk_effect_{metric }"
    ],
    }
    for metric in PRIMARY_METRICS 
    },
    "long_horizon_persistence":{
    metric :{
    "delta_trajectory_mean":
    aggregate_by_metric [metric ][
    "delta_trajectory_mean"
    ],
    "delta_normalized_AUC":
    aggregate_by_metric [metric ][
    "delta_normalized_AUC"
    ],
    "positive_delta_checkpoint_fraction":
    aggregate_by_metric [metric ][
    "positive_delta_checkpoint_fraction"
    ],
    "minimum_delta":
    aggregate_by_metric [metric ][
    "minimum_delta"
    ],
    "maximum_delta":
    aggregate_by_metric [metric ][
    "maximum_delta"
    ],
    "terminal_delta":
    aggregate_by_metric [metric ][
    "terminal_delta"
    ],
    }
    for metric in PRIMARY_METRICS 
    },
    "attribution_conclusion":{
    "shared_schedule_degradation_present":
    terminal [
    "shared_schedule_effect_AP"
    ]<0.0 ,
    "risk_specific_terminal_penalty_present":
    terminal [
    "risk_specific_residual_AP"
    ]<0.0 ,
    "persistent_small_AP_advantage_established":
    (
    aggregate_by_metric [
    "AP_small"
    ][
    "positive_delta_checkpoint_fraction"
    ]
    >0.5 
    and aggregate_by_metric [
    "AP_small"
    ]["delta_normalized_AUC"]
    >0.0 
    ),
    },
    "interpretation":(
    "Shared schedule effects and risk-specific "
    "residuals are descriptive decompositions. "
    "They do not constitute a fully randomized "
    "causal identification of optimization and "
    "data-intervention mechanisms."
    ),
    "elapsed_minutes":
    (time .time ()-started )/60.0 ,
    }

    schedule_audit ={
    "version":"optimization_schedule_audit_v1",
    "canonical_metrics_source":
    str (TRAJECTORY_METRICS ),
    "raw_training_metrics_source":{
    label :str (path )
    for label ,path in RAW_RESULTS .items ()
    },
    "raw_results_sha256":{
    label :sha256_file (path )
    for label ,path in RAW_RESULTS .items ()
    },
    "loss_audit":loss_audit ,
    "learning_rate_restart_ratios":
    restart_ratios ,
    "metric_separation_rule":(
    "Canonical COCO trajectory metrics are used "
    "for AP/AR conclusions. Raw Ultralytics "
    "metrics are used only for optimization, "
    "learning-rate and loss diagnostics."
    ),
    }

    write_csv (
    staging /"matched_epoch_attribution.csv",
    attribution_rows ,
    )
    write_csv (
    staging /"trajectory_aggregate.csv",
    aggregate_rows ,
    )
    write_json (
    staging /"optimization_schedule_audit.json",
    schedule_audit ,
    )
    write_json (
    staging /"summary.json",
    summary ,
    )

    metadata ={
    "version":
    "training_dynamics_attribution_v1",
    "status":"complete",
    "created_utc":time .strftime (
    "%Y-%m-%dT%H:%M:%SZ",
    time .gmtime (),
    ),
    "script":
    str (Path (__file__ ).resolve ()),
    "script_sha256":
    sha256_file (Path (__file__ ).resolve ()),
    "new_training":False ,
    "new_inference":False ,
    "new_model_selection":False ,
    "saved_checkpoint_grid":
    list (matched_epochs ),
    "grid_limitation":(
    "Unsaved epochs between the frozen "
    "checkpoint nodes cannot be reconstructed."
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
    print ("TRAINING-DYNAMICS ATTRIBUTION COMPLETE")
    print ("="*100 )

    print (json .dumps (
    summary [
    "terminal_epoch_100_attribution"
    ],
    indent =2 ,
    ensure_ascii =False ,
    sort_keys =True ,
    ))

    print ("\nLONG-HORIZON PERSISTENCE")
    print (json .dumps (
    summary ["long_horizon_persistence"],
    indent =2 ,
    ensure_ascii =False ,
    sort_keys =True ,
    ))

    print ("\nCONCLUSION")
    print (json .dumps (
    summary ["attribution_conclusion"],
    indent =2 ,
    ensure_ascii =False ,
    sort_keys =True ,
    ))

    print ("output:",OUTPUT )
    print (
    "manifest:",
    OUTPUT /manifest_path .name ,
    )
    print ("manifest_sha256:",manifest_sha256 )
    print ("NOTHING TRAINED OR INFERRED")
    print ("NEXT: paired bootstrap confidence intervals")


def main ():
    args =parse_args ()
    protocol ,trajectory_rows =preflight ()

    if args .preflight_only :
        print ("\nPASS: Stage-4 preflight only")
        print ("NOTHING CREATED OR ANALYZED")
        return 

    execute (protocol ,trajectory_rows )


if __name__ =="__main__":
    main ()
