from __future__ import annotations 

import argparse 
import csv 
import hashlib 
import json 
import os 
import shutil 
from collections import Counter 
from datetime import datetime ,timezone 
from pathlib import Path 


RESULTS =Path ("/root/rivermind-data/autodrive/results/evaluation")
FINAL =RESULTS /"final_analysis_v1"
FORMAL =RESULTS /"formal_training"
PROTOCOL =FINAL /"frozen_protocol/final_testing_protocol_v1.json"
OUTPUT =FINAL /"final_paper_bundle_v1"
WORK =OUTPUT .with_name (OUTPUT .name +".incomplete-693c39375cc1")

DIRS ={
"canonical":FINAL /"canonical_clean_v1",
"failures":FINAL /"official_val_failure_subsets_v1",
"threshold":FINAL /"threshold_sensitivity_v1",
"dynamics":FINAL /"training_dynamics_attribution_v1",
"bootstrap":FINAL /"paired_bootstrap_v1",
"exposure":FINAL /"exposure_audit_v1",
"crop":FINAL /"crop_scale_context_audit_v1",
"corruptions":FINAL /"representative_corruptions_v1",
"efficiency":FINAL /"efficiency_evaluation_v1",
"qualitative":FINAL /"qualitative_cases_v1",
"trajectory":FORMAL /"trajectory_evaluation_v1",
}

PROTOCOL_SHA256 =(
"693c39375cc1eefc4f3ff9f957a8355fb947552a19dfef228cae6cfc21e3dcfe"
)

MANIFEST_HASHES ={
FINAL /"frozen_protocol/artifact_manifest.json":
"bc6bac887694d7e4e09210e6996df22d1cf7006b8c44957011bd4fcf9112adce",
DIRS ["canonical"]/"artifact_manifest.json":
"88d2bace7a2f69e26c34e7eed4e2814b12e8317980602968b407d1b94282338b",
DIRS ["failures"]/"artifact_manifest.json":
"79029af794f214d45bdecb6cb00d9b64155aa8ace04df97ca6524e1f40f2c1dc",
DIRS ["threshold"]/"artifact_manifest.json":
"de9536176d1d171b60ac647ed4d834faf55f21598bae24e19440f4ef5f2a9460",
DIRS ["dynamics"]/"artifact_manifest.json":
"22ce36eed8f85d1e7cbc013b6763bb8c8535146fc6dd55c408c2f31b5356d2d9",
DIRS ["bootstrap"]/"artifact_manifest.json":
"35b07a65bffb2e1e72a31dcee9cc568c1dea36d181f7bdf68e51401c594c0115",
DIRS ["exposure"]/"artifact_manifest.json":
"94b87f98f4c320da3bc0bf1d87b9c5820682d9dbd1a3443eb4f2f42cdb78d33f",
DIRS ["crop"]/"artifact_manifest.json":
"30a8c5043acab642bd025802ffd318e31515422f9671a5a54a56cbd820de558d",
DIRS ["corruptions"]/"artifact_manifest.json":
"b2be2be57d06b103317ef7c80934d010c9171cd2d21ca8536e44cf21a25b85b3",
DIRS ["efficiency"]/"artifact_manifest.json":
"bee305e01b57dca02fe3b66ee7802596204dac811494c40a2d52e70dc99257cc",
DIRS ["qualitative"]/"artifact_manifest.json":
"a1b13f5e699481c946fb03de62f657ee0b56c3fe3efd83122676fcb7759cf2d3",
DIRS ["trajectory"]/"artifact_manifest.json":
"922f27c7b573e500f449809591af7e7886449b62e9f662ba50609ac1db695485",
}

FILES ={
"canonical_summary":DIRS ["canonical"]/"canonical_clean_summary.json",
"canonical_csv":DIRS ["canonical"]/"canonical_clean_metrics.csv",
"failure_csv":DIRS ["failures"]/"subset_summary.csv",
"threshold_summary":DIRS ["threshold"]/"summary.json",
"recovery":DIRS ["threshold"]/"primary_recovery_metrics.json",
"threshold_csv":DIRS ["threshold"]/"primary_effect_direction_stability.csv",
"dynamics":DIRS ["dynamics"]/"summary.json",
"bootstrap_summary":DIRS ["bootstrap"]/"summary.json",
"bootstrap_csv":DIRS ["bootstrap"]/"paired_bootstrap_intervals.csv",
"exposure":DIRS ["exposure"]/"summary.json",
"exposure_csv":DIRS ["exposure"]/"arm_exposure_summary.csv",
"crop":DIRS ["crop"]/"summary.json",
"scale_csv":DIRS ["crop"]/"scale_transition_matrix.csv",
"corruptions":DIRS ["corruptions"]/"summary.json",
"corruption_csv":DIRS ["corruptions"]/"model_corruption_family_summary.csv",
"efficiency":DIRS ["efficiency"]/"efficiency_summary.json",
"efficiency_csv":DIRS ["efficiency"]/"efficiency_table.csv",
"qualitative":DIRS ["qualitative"]/"metadata.json",
"qualitative_csv":DIRS ["qualitative"]/"case_selection.csv",
"qualitative_probe":DIRS ["qualitative"]/"corruption_probe_summary.json",
"trajectory_csv":DIRS ["trajectory"]/"trajectory_metrics.csv",
"trajectory_summary":DIRS ["trajectory"]/"trajectory_summary.json",
}

METRICS =(
"AP","AP50","AP75","AP_small",
"AP_medium","AP_large","AR100","AR_small",
)


def sha256 (path :Path )->str :
    digest =hashlib .sha256 ()
    with path .open ("rb")as handle :
        while True :
            block =handle .read (8 *1024 *1024 )
            if not block :
                break 
            digest .update (block )
    return digest .hexdigest ()


def load_json (path :Path ):
    return json .loads (path .read_text (encoding ="utf-8"))


def load_csv (path :Path )->list [dict ]:
    with path .open (newline ="",encoding ="utf-8")as handle :
        return list (csv .DictReader (handle ))


def write_json (path :Path ,payload )->None :
    path .parent .mkdir (parents =True ,exist_ok =True )
    path .write_text (
    json .dumps (
    payload ,
    indent =2 ,
    ensure_ascii =False ,
    sort_keys =True ,
    )+"\n",
    encoding ="utf-8",
    )


def write_csv (path :Path ,rows :list [dict ])->None :
    if not rows :
        raise ValueError (path )
    path .parent .mkdir (parents =True ,exist_ok =True )
    with path .open ("w",newline ="",encoding ="utf-8")as handle :
        writer =csv .DictWriter (
        handle ,
        fieldnames =list (rows [0 ]),
        extrasaction ="ignore",
        )
        writer .writeheader ()
        writer .writerows (rows )


def as_bool (value )->bool :
    if isinstance (value ,bool ):
        return value 
    return str (value ).lower ()=="true"


def fmt (value ,digits =4 )->str :
    return f"{float (value ):.{digits }f}"


def pct (value ,digits =1 )->str :
    return f"{100 *float (value ):.{digits }f}%"


def diff (models :dict ,positive :str ,reference :str ,metric :str )->float :
    return float (models [positive ][metric ])-float (models [reference ][metric ])


def table (rows :list [dict ],columns :list [tuple [str ,str ]])->str :
    lines =[
    "| "+" | ".join (label for _ ,label in columns )+" |",
    "| "+" | ".join ("---"for _ in columns )+" |",
    ]
    for row in rows :
        lines .append (
        "| "
        +" | ".join (
        str (row .get (key ,"")).replace ("|","\\|")
        for key ,_ in columns 
        )
        +" |"
        )
    return "\n".join (lines )


def parse_args ():
    parser =argparse .ArgumentParser ()
    group =parser .add_mutually_exclusive_group (required =True )
    group .add_argument ("--preflight-only",action ="store_true")
    group .add_argument ("--execute",action ="store_true")
    return parser .parse_args ()


def verify_complete (path :Path )->dict :
    data =load_json (path )
    if data .get ("status")!="complete":
        raise AssertionError ((path ,data .get ("status")))
    return data 


def preflight ()->dict :
    print ("=== FINAL PAPER BUNDLE PREFLIGHT ===")
    actual =sha256 (PROTOCOL )
    print ("protocol_expected:",PROTOCOL_SHA256 )
    print ("protocol_actual  :",actual )
    if actual !=PROTOCOL_SHA256 :
        raise AssertionError (PROTOCOL )

    for path ,expected in MANIFEST_HASHES .items ():
        if not path .is_file ():
            raise FileNotFoundError (path )
        actual =sha256 (path )
        print ("PASS"if actual ==expected else "FAIL",path )
        if actual !=expected :
            raise AssertionError ({
            "path":str (path ),
            "expected":expected ,
            "actual":actual ,
            })

    for path in FILES .values ():
        if not path .is_file ():
            raise FileNotFoundError (path )

    data ={
    "canonical":verify_complete (FILES ["canonical_summary"]),
    "threshold":verify_complete (FILES ["threshold_summary"]),
    "dynamics":verify_complete (FILES ["dynamics"]),
    "bootstrap":verify_complete (FILES ["bootstrap_summary"]),
    "exposure":verify_complete (FILES ["exposure"]),
    "crop":verify_complete (FILES ["crop"]),
    "corruptions":verify_complete (FILES ["corruptions"]),
    "efficiency":verify_complete (FILES ["efficiency"]),
    "qualitative":verify_complete (FILES ["qualitative"]),
    "trajectory":verify_complete (FILES ["trajectory_summary"]),
    }

    if len (data ["canonical"]["models"])!=13 :
        raise AssertionError (len (data ["canonical"]["models"]))
    if len (load_csv (FILES ["failure_csv"]))!=6 :
        raise AssertionError (FILES ["failure_csv"])

    qualitative_rows =load_csv (FILES ["qualitative_csv"])
    counts =Counter (row ["family"]for row in qualitative_rows )
    if len (qualitative_rows )!=32 or len (counts )!=8 or set (counts .values ())!={4 }:
        raise AssertionError (counts )

    if OUTPUT .exists ():
        raise FileExistsError (OUTPUT )
    if WORK .exists ():
        raise FileExistsError (WORK )

    print ("canonical_models:",len (data ["canonical"]["models"]))
    print ("failure_subsets:",6 )
    print ("qualitative_cases:",len (qualitative_rows ))
    print ("PASS: Stage 1-10 inputs are immutable and complete")
    print ("NOTHING CREATED, MODIFIED, TRAINED OR INFERRED")
    return data 


def canonical_rows (models :dict )->list [dict ]:
    rows =[]
    for key ,value in models .items ():
        row ={
        "model_key":key ,
        "display_name":value ["display_name"],
        }
        for metric in METRICS :
            row [metric ]=float (value [metric ])
        rows .append (row )
    return rows 


def architecture_rows (models :dict ,efficiency :dict )->list [dict ]:
    rows =[]
    for key in ("standard_e20","p2_e20"):
        clean =models [key ]
        runtime =efficiency ["models"][key ]
        row ={"model":key }
        for metric in METRICS :
            row [metric ]=float (clean [metric ])
        row .update ({
        "parameters":int (runtime ["parameters"]),
        "GFLOPs":float (runtime ["GFLOPs"]),
        "median_latency_ms":float (runtime ["median_forward_latency_ms"]),
        "p90_latency_ms":float (runtime ["p90_forward_latency_ms"]),
        "peak_inference_MiB":float (runtime ["peak_inference_memory_MiB"]),
        "peak_training_MiB":float (runtime ["peak_training_memory_MiB"]),
        })
        rows .append (row )
    delta ={"model":"p2_minus_standard"}
    for key in rows [0 ]:
        if key !="model":
            delta [key ]=rows [1 ][key ]-rows [0 ][key ]
    rows .append (delta )
    return rows 


def threshold_rows (summary :dict )->list [dict ]:
    rows =[]
    for comparison ,metrics in summary ["effect_direction_stability"].items ():
        for metric ,values in metrics .items ():
            rows .append ({
            "comparison":comparison ,
            "metric":metric ,
            **values ,
            })
    return rows 


def dynamics_rows (summary :dict )->list [dict ]:
    rows =[]
    for metric in ("AP","AP75","AP_small"):
        rows .append ({
        "metric":metric ,
        **summary ["terminal_epoch_100_attribution"][metric ],
        **summary ["long_horizon_persistence"][metric ],
        })
    return rows 


def exposure_rows (summary :dict )->list [dict ]:
    fields =(
    "training_views","original_views","object_centric_views",
    "unique_source_images","total_supervised_objects",
    "target_objects","non_target_objects","objects_per_view",
    "raw_input_pixels","network_input_pixels",
    "mean_source_exposures","maximum_source_exposures",
    )
    return [
    {
    "arm":arm ,
    **{field :values [field ]for field in fields },
    }
    for arm ,values in summary ["arms"].items ()
    ]


def corruption_rows (summary :dict )->list [dict ]:
    rows =[]
    for model ,values in summary ["models"].items ():
        rows .append ({
        "model":model ,
        "conditions":values ["conditions"],
        "mean_AP":values ["mean_corruption_AP"],
        "mean_AP50":values ["mean_corruption_AP50"],
        "mean_AP75":values ["mean_corruption_AP75"],
        "mean_AP_small":values ["mean_corruption_AP_small"],
        "mean_AP_medium":values ["mean_corruption_AP_medium"],
        "mean_AP_large":values ["mean_corruption_AP_large"],
        "mean_AP_retention":values ["mean_AP_retention"],
        "mean_AP_small_retention":values ["mean_AP_small_retention"],
        })
    return rows 


def bootstrap_find (rows :list [dict ],comparison :str ,metric :str ):
    for row in rows :
        if row ["comparison"]==comparison and row ["metric"]==metric :
            return row 
    return None 


def claim_rows (
models :dict ,
bootstrap :list [dict ],
dynamics :dict ,
crop :dict ,
corruptions :dict ,
efficiency :dict ,
)->list [dict ]:
    p2_ci =bootstrap_find (
    bootstrap ,"p2_e20_minus_standard_e20","AP_small"
    )
    risk_ci =bootstrap_find (
    bootstrap ,"rebu_risk_e56_minus_uniform_e56","AP_small"
    )
    p2_corrupt_delta =(
    corruptions ["models"]["p2_e20"]["mean_corruption_AP_small"]
    -corruptions ["models"]["standard_e20"]["mean_corruption_AP_small"]
    )
    return [
    {
    "claim_id":"C1",
    "claim":"P2 improves small-object detection in the evaluated setting.",
    "point_estimate_support":diff (models ,"p2_e20","standard_e20","AP_small")>0 ,
    "uncertainty_support":as_bool (p2_ci ["stable_positive"])if p2_ci else False ,
    "evidence":"Canonical AP small, AR small, and paired image bootstrap.",
    "permitted_language":"Improves under the evaluated single-seed setting.",
    "limitation":"Official validation participated in architecture adjudication.",
    },
    {
    "claim_id":"C2",
    "claim":"P2 retains a descriptive advantage under representative corruptions.",
    "point_estimate_support":p2_corrupt_delta >0 ,
    "uncertainty_support":"descriptive_only",
    "evidence":"Fifteen controlled corruption conditions per checkpoint.",
    "permitted_language":"Descriptive corruption advantage.",
    "limitation":"Synthetic corruptions; no strict deployment claim.",
    },
    {
    "claim_id":"C3",
    "claim":"Risk cropping does not establish a persistent AP-small advantage.",
    "point_estimate_support":not dynamics ["attribution_conclusion"]["persistent_small_AP_advantage_established"],
    "uncertainty_support":(not as_bool (risk_ci ["stable_positive"]))if risk_ci else "unavailable",
    "evidence":"Full saved-checkpoint trajectory, AUC, terminal metrics, bootstrap.",
    "permitted_language":"No persistent advantage was established.",
    "limitation":"Not proof that every risk-based intervention fails.",
    },
    {
    "claim_id":"C4",
    "claim":"Object-centric cropping changes target scale and surrounding context.",
    "point_estimate_support":(
    crop ["primary_scale_question"]["fraction"]>0 
    and crop ["context_shift"]["weighted_neighbor_retention_fraction"]<1 
    ),
    "uncertainty_support":"deterministic_dataset_audit",
    "evidence":"Nineteen-thousand-five-hundred target scale/context audit.",
    "permitted_language":"Quantified condition shift associated with non-transfer.",
    "limitation":"Association, not strict causal identification.",
    },
    {
    "claim_id":"C5",
    "claim":"Long-horizon decline has shared-schedule and risk-specific components.",
    "point_estimate_support":(
    dynamics ["attribution_conclusion"]["shared_schedule_degradation_present"]
    and dynamics ["attribution_conclusion"]["risk_specific_terminal_penalty_present"]
    ),
    "uncertainty_support":"descriptive_decomposition",
    "evidence":"Common40 anchor and matched E41-E100 trajectories.",
    "permitted_language":"Descriptive attribution only.",
    "limitation":"One fixed training seed.",
    },
    {
    "claim_id":"C6",
    "claim":"P2 has measurable compute and memory cost.",
    "point_estimate_support":efficiency ["p2_minus_standard"]["median_forward_latency_ms"]["increase_fraction"]>0 ,
    "uncertainty_support":"repeated_runtime_measurement",
    "evidence":"Two-order RTX4090 FP16 runtime benchmark.",
    "permitted_language":"RTX4090 PyTorch FP16 evidence.",
    "limitation":"Not edge-device end-to-end latency.",
    },
    ]


def build_markdown (
data :dict ,
failures :list [dict ],
bootstrap :list [dict ],
)->str :
    models =data ["canonical"]["models"]
    dynamics =data ["dynamics"]
    exposure =data ["exposure"]
    crop =data ["crop"]
    corruptions =data ["corruptions"]
    efficiency =data ["efficiency"]
    qualitative =data ["qualitative"]

    architecture =[]
    for key in ("standard_e20","p2_e20"):
        row =models [key ]
        architecture .append ({
        "model":row ["display_name"],
        "AP":fmt (row ["AP"]),
        "AP50":fmt (row ["AP50"]),
        "AP75":fmt (row ["AP75"]),
        "APs":fmt (row ["AP_small"]),
        "ARs":fmt (row ["AR_small"]),
        })
    architecture .append ({
    "model":"P2 minus Standard",
    "AP":fmt (diff (models ,"p2_e20","standard_e20","AP")),
    "AP50":fmt (diff (models ,"p2_e20","standard_e20","AP50")),
    "AP75":fmt (diff (models ,"p2_e20","standard_e20","AP75")),
    "APs":fmt (diff (models ,"p2_e20","standard_e20","AP_small")),
    "ARs":fmt (diff (models ,"p2_e20","standard_e20","AR_small")),
    })

    methods =[]
    for key in (
    "gate_uniform","gate_scalar","gate_typed","gate_typed_v2",
    "common40","uniform_e56","risk_e56","uniform_e100","risk_e100",
    ):
        row =models [key ]
        methods .append ({
        "model":row ["display_name"],
        "AP":fmt (row ["AP"]),
        "AP75":fmt (row ["AP75"]),
        "APs":fmt (row ["AP_small"]),
        "AR100":fmt (row ["AR100"]),
        })

    subsets =[
    {
    "subset":row ["subset"],
    "objects":f"{int (row ['num_objects']):,}",
    "images":f"{int (row ['num_images']):,}",
    "risk":fmt (row ["mean_risk"]),
    }
    for row in failures 
    ]

    bootstrap_md =[]
    chosen ={
    "p2_e20_minus_standard_e20",
    "gate_typed_minus_gate_uniform",
    "gate_typed_minus_gate_scalar",
    "rebu_risk_e56_minus_uniform_e56",
    "risk_e100_minus_uniform_e100",
    }
    for row in bootstrap :
        if not row ["evidence_role"].startswith ("primary"):
            continue 
        if row ["comparison"]not in chosen :
            continue 
        direction ="not detected"
        if as_bool (row ["stable_positive"]):
            direction ="positive"
        elif as_bool (row ["stable_negative"]):
            direction ="negative"
        bootstrap_md .append ({
        "comparison":row ["comparison"],
        "metric":row ["metric"],
        "delta":fmt (row ["point_estimate"]),
        "CI":"["+fmt (row ["ci95_lower"])+", "+fmt (row ["ci95_upper"])+"]",
        "direction":direction ,
        })

    corruption_md =[]
    for key in ("standard_e20","p2_e20","common40","uniform_e56","rebu_risk_e56"):
        row =corruptions ["models"][key ]
        corruption_md .append ({
        "model":key ,
        "AP":fmt (row ["mean_corruption_AP"]),
        "APs":fmt (row ["mean_corruption_AP_small"]),
        })

    runtime_md =[]
    for key in ("standard_e20","p2_e20"):
        row =efficiency ["models"][key ]
        runtime_md .append ({
        "model":key ,
        "params":f"{int (row ['parameters']):,}",
        "GFLOPs":fmt (row ["GFLOPs"],2 ),
        "median":fmt (row ["median_forward_latency_ms"],3 ),
        "p90":fmt (row ["p90_forward_latency_ms"],3 ),
        "infer":fmt (row ["peak_inference_memory_MiB"],1 ),
        "train":fmt (row ["peak_training_memory_MiB"],1 ),
        })

    lines =[
    "# Diagnose Before Repair",
    "",
    "## Matching Architectural and Data Interventions to Object-Level Failures in Autonomous-Driving Detection",
    "",
    "This paper-facing result record is generated only from the frozen Stage 1-10 evidence. It performs no new training, inference, threshold tuning, model selection, or qualitative replacement.",
    "",
    "## Experimental scope",
    "",
    "- BDD100K official validation: 10,000 images and 185,523 objects.",
    "- YOLO11m at 960 pixels is the final detector family; D-FINE-M is diagnosis-only.",
    "- Statistical uncertainty uses 2,000 paired official-validation image bootstrap replicates.",
    "- Training evidence uses one fixed seed.",
    "- Official validation participated in architecture adjudication and is not an untouched test set.",
    "",
    "## Architecture intervention",
    "",
    table (architecture ,[
    ("model","Model"),("AP","AP"),("AP50","AP50"),
    ("AP75","AP75"),("APs","AP small"),("ARs","AR small"),
    ]),
    "",
    (
    "P2 minus Standard changes canonical AP by "
    f"{diff (models ,'p2_e20','standard_e20','AP'):+.4f}, "
    "AP small by "
    f"{diff (models ,'p2_e20','standard_e20','AP_small'):+.4f}, "
    "and AR small by "
    f"{diff (models ,'p2_e20','standard_e20','AR_small'):+.4f}."
    ),
    "",
    "## Frozen failure subsets",
    "",
    table (subsets ,[
    ("subset","Subset"),("objects","Objects"),
    ("images","Images"),("risk","Mean risk"),
    ]),
    "",
    "Teacher operating points were independently calibrated. Threshold sensitivity tests directional stability but does not turn diagnostic association into causal labels.",
    "",
    "## Data-intervention results",
    "",
    table (methods ,[
    ("model","Model"),("AP","AP"),("AP75","AP75"),
    ("APs","AP small"),("AR100","AR100"),
    ]),
    "",
    (
    "Risk-E56 minus Uniform-E56 is "
    f"{diff (models ,'risk_e56','uniform_e56','AP'):+.4f} AP and "
    f"{diff (models ,'risk_e56','uniform_e56','AP_small'):+.4f} AP small. "
    "Risk-E100 minus Uniform-E100 is "
    f"{diff (models ,'risk_e100','uniform_e100','AP'):+.4f} AP and "
    f"{diff (models ,'risk_e100','uniform_e100','AP_small'):+.4f} AP small."
    ),
    "",
    (
    "Across saved E41-E100 checkpoints, the positive risk-minus-uniform "
    "AP-small checkpoint fraction is "
    f"{pct (dynamics ['long_horizon_persistence']['AP_small']['positive_delta_checkpoint_fraction'])}; "
    "normalized AP-small delta AUC is "
    f"{dynamics ['long_horizon_persistence']['AP_small']['delta_normalized_AUC']:+.4f}. "
    "A persistent small-object advantage was not established."
    ),
    "",
    "## Training dynamics",
    "",
    (
    "At E100, the AP shared-schedule effect is "
    f"{dynamics ['terminal_epoch_100_attribution']['AP']['shared_schedule_effect']:+.4f} "
    "and the risk-specific residual is "
    f"{dynamics ['terminal_epoch_100_attribution']['AP']['risk_specific_residual']:+.4f}. "
    "These are descriptive decompositions, not randomized causal effects."
    ),
    "",
    "## Exposure, scale, and context",
    "",
    (
    f"Both formal arms use {int (exposure ['arms']['uniform']['training_views']):,} views. "
    f"The risk arm contains {int (exposure ['arms']['rebu_risk']['object_centric_views']):,} "
    "object-centric views and "
    f"{int (exposure ['arms']['rebu_risk']['total_supervised_objects']):,} supervised objects, "
    "versus "
    f"{int (exposure ['arms']['uniform']['total_supervised_objects']):,} for uniform training. "
    "The arms are equal in views and optimization steps, not supervision content."
    ),
    "",
    (
    f"Among {int (crop ['targets']):,} crop targets, "
    f"{pct (crop ['primary_scale_question']['fraction'])} of originally small targets "
    "cease to be small in crop coordinates. Mean short-side amplification is "
    f"{crop ['scale_shift']['short_side_amplification']['mean']:.3f} times; "
    "weighted neighbor retention is "
    f"{crop ['context_shift']['weighted_neighbor_retention_fraction']:.3f}. "
    f"{int (crop ['context_shift']['fully_removed_objects']):,} neighbors are fully removed "
    "and "
    f"{int (crop ['context_shift']['boundary_clipped_objects']):,} are boundary-clipped."
    ),
    "",
    "These measurements establish condition shift and an empirical association with non-transfer, not strict causal identification.",
    "",
    "## Controlled corruptions",
    "",
    table (corruption_md ,[
    ("model","Model"),("AP","Mean corruption AP"),
    ("APs","Mean corruption AP small"),
    ]),
    "",
    (
    "P2 minus Standard changes mean corruption AP by "
    f"{corruptions ['models']['p2_e20']['mean_corruption_AP']-corruptions ['models']['standard_e20']['mean_corruption_AP']:+.4f} "
    "and mean corruption AP small by "
    f"{corruptions ['models']['p2_e20']['mean_corruption_AP_small']-corruptions ['models']['standard_e20']['mean_corruption_AP_small']:+.4f}. "
    "Risk-E56 minus Uniform-E56 changes the same metrics by "
    f"{corruptions ['models']['rebu_risk_e56']['mean_corruption_AP']-corruptions ['models']['uniform_e56']['mean_corruption_AP']:+.4f} and "
    f"{corruptions ['models']['rebu_risk_e56']['mean_corruption_AP_small']-corruptions ['models']['uniform_e56']['mean_corruption_AP_small']:+.4f}."
    ),
    "",
    "## Efficiency",
    "",
    table (runtime_md ,[
    ("model","Model"),("params","Parameters"),("GFLOPs","GFLOPs"),
    ("median","Median ms"),("p90","P90 ms"),
    ("infer","Peak inference MiB"),("train","Peak training MiB"),
    ]),
    "",
    (
    "P2 increases RTX4090 PyTorch FP16 median forward latency by "
    f"{pct (efficiency ['p2_minus_standard']['median_forward_latency_ms']['increase_fraction'])} "
    "and GFLOPs by "
    f"{pct (efficiency ['p2_minus_standard']['GFLOPs']['increase_fraction'])}. "
    "This is not an edge-device end-to-end latency claim."
    ),
    "",
    "## Paired-bootstrap primary evidence",
    "",
    table (bootstrap_md ,[
    ("comparison","Comparison"),("metric","Metric"),
    ("delta","Delta"),("CI","95% CI"),
    ("direction","Detected direction"),
    ]),
    "",
    "Intervals crossing zero are no stable detected difference, not proof of equivalence. Image bootstrap does not replace seed replication.",
    "",
    "## Deterministic qualitative evidence",
    "",
    (
    f"The frozen protocol produced {int (qualitative ['total_cases'])} cases "
    f"across {int (qualitative ['families'])} families, "
    f"{int (qualitative ['cases_per_family'])} each, using deterministic SHA ranking "
    "without manual replacement."
    ),
    "",
    "## Main conclusion",
    "",
    "Correctly diagnosing an object-level failure does not guarantee that increasing exposure to a transformed copy will repair the original failure condition. In this study, the scene-preserving P2 intervention improves small-object and representative-corruption metrics with measurable efficiency cost. Object-centric risk cropping changes scale, context, and supervision composition and does not establish a persistent small-object advantage. Failure-guided training should therefore verify that its intervention preserves the condition that produced the failure.",
    "",
    "## Required limitations",
    "",
    "- One fixed training seed.",
    "- Official validation participated in architecture adjudication.",
    "- Formal E41-E100 optimization used a restarted schedule.",
    "- Five-epoch checkpoint spacing cannot recover unsaved optima.",
    "- Risk crops do not preserve identical supervision content.",
    "- Mechanism evidence is associative, not strict causal identification.",
    "- Controlled corruptions are synthetic.",
    "- Efficiency evidence is RTX4090 PyTorch FP16 specific.",
    "",
    "## Reproducibility",
    "",
    "- Frozen final-testing protocol SHA256: "+PROTOCOL_SHA256 ,
    "- Paired bootstrap replicates: "+str (data ["bootstrap"]["replicates"]),
    "- Qualitative manifest SHA256: "+MANIFEST_HASHES [DIRS ["qualitative"]/"artifact_manifest.json"],
    "- This bundle performs no new training or inference.",
    "",
    ]
    return "\n".join (lines )


def build_manifest (directory :Path )->tuple [Path ,str ]:
    artifacts =[]
    for path in sorted (directory .rglob ("*")):
        if not path .is_file ()or path .name =="artifact_manifest.json":
            continue 
        artifacts .append ({
        "path":str (path .relative_to (directory )),
        "bytes":path .stat ().st_size ,
        "sha256":sha256 (path ),
        })
    payload ={
    "version":"final_paper_bundle_manifest_v1",
    "status":"complete",
    "created_utc":datetime .now (timezone .utc ).isoformat (),
    "protocol_sha256":PROTOCOL_SHA256 ,
    "source_manifests":{str (path ):value for path ,value in MANIFEST_HASHES .items ()},
    "artifacts":artifacts ,
    }
    path =directory /"artifact_manifest.json"
    write_json (path ,payload )
    return path ,sha256 (path )


def execute (data :dict )->None :
    WORK .mkdir (parents =True ,exist_ok =False )
    tables =WORK /"tables"
    tables .mkdir ()

    models =data ["canonical"]["models"]
    failures =load_csv (FILES ["failure_csv"])
    bootstrap =load_csv (FILES ["bootstrap_csv"])
    qualitative =load_csv (FILES ["qualitative_csv"])
    trajectory =load_csv (FILES ["trajectory_csv"])
    recovery =load_json (FILES ["recovery"])

    clean =canonical_rows (models )
    architecture =architecture_rows (models ,data ["efficiency"])
    method_keys ={
    "gate_uniform","gate_scalar","gate_typed","gate_typed_v2",
    "common40","uniform_e56","risk_e56","uniform_e100","risk_e100",
    }
    methods =[row for row in clean if row ["model_key"]in method_keys ]
    primary_bootstrap =[
    row for row in bootstrap 
    if row ["evidence_role"].startswith ("primary")
    ]

    central ="yolo_central__dfine_central"
    recovery_rows =[]
    for model ,combinations in recovery .items ():
        values =combinations [central ]
        recovery_rows .append ({
        "model":model ,
        "failure_macro_AR100":values ["failure_macro_AR100"],
        "small_jointly_missed_recall50":values ["small_jointly_missed_recall50"],
        })

    crop =data ["crop"]
    crop_rows =[{
    "targets":crop ["targets"],
    "original_small_targets":crop ["primary_scale_question"]["original_small_targets"],
    "original_small_ceased":crop ["primary_scale_question"]["original_small_ceased_to_be_small"],
    "original_small_ceased_fraction":crop ["primary_scale_question"]["fraction"],
    "mean_short_side_amplification":crop ["scale_shift"]["short_side_amplification"]["mean"],
    "mean_area_amplification":crop ["scale_shift"]["area_amplification"]["mean"],
    "weighted_neighbor_retention":crop ["context_shift"]["weighted_neighbor_retention_fraction"],
    "fully_removed_objects":crop ["context_shift"]["fully_removed_objects"],
    "boundary_clipped_objects":crop ["context_shift"]["boundary_clipped_objects"],
    }]

    outputs ={
    "table_01_canonical_clean.csv":clean ,
    "table_02_architecture_clean_efficiency.csv":architecture ,
    "table_03_method_clean.csv":methods ,
    "table_04_failure_subsets.csv":failures ,
    "table_05_failure_recovery.csv":recovery_rows ,
    "table_06_threshold_stability.csv":threshold_rows (data ["threshold"]),
    "table_07_bootstrap_primary.csv":primary_bootstrap ,
    "table_08_checkpoint_trajectory.csv":trajectory ,
    "table_09_training_dynamics.csv":dynamics_rows (data ["dynamics"]),
    "table_10_exposure_audit.csv":exposure_rows (data ["exposure"]),
    "table_11_crop_scale_context.csv":crop_rows ,
    "table_13_corruption_summary.csv":corruption_rows (data ["corruptions"]),
    "table_15_qualitative_cases.csv":qualitative ,
    }
    for name ,rows in outputs .items ():
        write_csv (tables /name ,rows )

    shutil .copyfile (FILES ["scale_csv"],tables /"table_12_scale_transition_matrix.csv")
    shutil .copyfile (FILES ["efficiency_csv"],tables /"table_14_efficiency.csv")

    claims =claim_rows (
    models =models ,
    bootstrap =bootstrap ,
    dynamics =data ["dynamics"],
    crop =data ["crop"],
    corruptions =data ["corruptions"],
    efficiency =data ["efficiency"],
    )
    write_csv (WORK /"claim_evidence_matrix.csv",claims )

    markdown =build_markdown (data ,failures ,bootstrap )
    (WORK /"FINAL_RESULTS.md").write_text (markdown ,encoding ="utf-8")

    figures =[]
    for path in sorted ((DIRS ["qualitative"]/"figures").glob ("*.png")):
        figures .append ({
        "figure":path .stem ,
        "path":str (path ),
        "bytes":path .stat ().st_size ,
        "sha256":sha256 (path ),
        })
    if len (figures )!=8 :
        raise AssertionError (len (figures ))
    write_csv (WORK /"qualitative_figure_registry.csv",figures )

    headline ={
    "p2_minus_standard":{
    metric :diff (models ,"p2_e20","standard_e20",metric )
    for metric in METRICS 
    },
    "risk_e56_minus_uniform_e56":{
    metric :diff (models ,"risk_e56","uniform_e56",metric )
    for metric in METRICS 
    },
    "risk_e100_minus_uniform_e100":{
    metric :diff (models ,"risk_e100","uniform_e100",metric )
    for metric in METRICS 
    },
    "crop_shift":crop_rows [0 ],
    "training_dynamics_conclusion":data ["dynamics"]["attribution_conclusion"],
    "efficiency":data ["efficiency"]["p2_minus_standard"],
    }
    bundle ={
    "version":"final_paper_evidence_bundle_v1",
    "status":"complete",
    "created_utc":datetime .now (timezone .utc ).isoformat (),
    "title":(
    "Diagnose Before Repair: Matching Architectural and Data "
    "Interventions to Object-Level Failures in Autonomous-Driving Detection"
    ),
    "protocol_sha256":PROTOCOL_SHA256 ,
    "new_training":False ,
    "new_inference":False ,
    "new_model_selection":False ,
    "headline_results":headline ,
    "claims":claims ,
    "required_limitations":[
    "single fixed training seed",
    "official validation participated in architecture adjudication",
    "formal E41-E100 optimization schedule restarted",
    "five-epoch checkpoint grid cannot recover unsaved epochs",
    "risk crops do not preserve identical supervision content",
    "mechanism evidence is associative, not strict causal identification",
    "synthetic controlled corruptions",
    "RTX4090 PyTorch FP16 efficiency scope",
    ],
    "source_manifests":{str (path ):value for path ,value in MANIFEST_HASHES .items ()},
    }
    write_json (WORK /"final_evidence_bundle.json",bundle )

    metadata ={
    "version":"final_paper_bundle_v1",
    "status":"complete",
    "created_utc":datetime .now (timezone .utc ).isoformat (),
    "script":str (Path (__file__ ).resolve ()),
    "script_sha256":sha256 (Path (__file__ ).resolve ()),
    "protocol_sha256":PROTOCOL_SHA256 ,
    "tables":15 ,
    "qualitative_figures":8 ,
    "new_training":False ,
    "new_inference":False ,
    "new_model_selection":False ,
    }
    write_json (WORK /"metadata.json",metadata )

    manifest_path ,manifest_sha =build_manifest (WORK )
    os .replace (WORK ,OUTPUT )

    print ("\n"+"="*100 )
    print ("FINAL PAPER EVIDENCE BUNDLE COMPLETE")
    print ("="*100 )
    print ("tables:",15 )
    print ("qualitative_figures:",8 )
    print ("results_markdown:",OUTPUT /"FINAL_RESULTS.md")
    print ("claim_matrix:",OUTPUT /"claim_evidence_matrix.csv")
    print ("evidence_bundle:",OUTPUT /"final_evidence_bundle.json")
    print ("manifest:",OUTPUT /manifest_path .name )
    print ("manifest_sha256:",manifest_sha )
    print ("NEW TRAINING: False")
    print ("NEW INFERENCE: False")
    print ("NEXT: manuscript drafting from the frozen evidence bundle")
    print ("="*100 )


def main ():
    args =parse_args ()
    data =preflight ()
    if args .preflight_only :
        return 
    execute (data )


if __name__ =="__main__":
    main ()
