from __future__ import annotations 

import csv 
import hashlib 
import json 
import math 
import os 
from collections import Counter 
from datetime import datetime ,timezone 
from pathlib import Path 

ROOT =Path (
"/root/rivermind-data/autodrive/results/evaluation"
)
CODE_ROOT =Path (
"/root/rivermind-data/autodrive/code/"
"robust-autonomous-driving-perception"
)

AUDIT_V3 =ROOT /"method_gate/sampling_audit_v3"
AUDIT_MANIFEST =AUDIT_V3 /"artifact_manifest.json"
AUDIT_SUMMARY =AUDIT_V3 /"sampling_summary.json"
TYPED_CSV =AUDIT_V3 /"typed_cafr_selection.csv"
SCALAR_CSV =AUDIT_V3 /"scalar_risk_selection.csv"
COMMON_BASE =AUDIT_V3 /"common_base_image_ids.json"
UNIFORM_SUPPLEMENT =(
AUDIT_V3 /"uniform_supplement_image_ids.json"
)

CORE_IDS =(
ROOT 
/"train_risk_labels/calibration_split/"
"train_core_image_ids.json"
)
V3_SCRIPT =(
CODE_ROOT 
/"scripts/prepare_cafr_sampling_audit_v3.py"
)

FREEZE_PATH =(
ROOT 
/"method_gate/frozen_manifests/"
"cafr_sampling_protocol_v1.json"
)

EXTERNAL_DEPENDENCIES ={
ROOT /(
"method_gate/sampling_audit_v1/"
"artifact_manifest.json"
):(
"77a84ab2b02198c687b3eaf449d0aae78902a1bad9df7a8ee38b6e495a533f10"
),
ROOT /(
"method_gate/sampling_audit_v2/"
"artifact_manifest.json"
):(
"7d7e7b8e02d7104eb92c05a6bb10fdcc0639963ecb53626f9db7ba34c742e5a7"
),
ROOT /(
"method_gate/object_view_smoke_v1/"
"artifact_manifest.json"
):(
"937835f1665575fde45da0ad3f7f85ca1eb5b7b00f8451f4256262f89d82434a"
),
ROOT /(
"architecture_gate/frozen_manifests/"
"final_architecture_selection_v1.json"
):(
"b3a20fc148a508334268aa4d43bc251219244dffba134173dee3ac1648de29f0"
),
ROOT /(
"architecture_gate/formal_gate_v4/"
"p2_projected/weights/resume_epoch20.pt"
):(
"642a6b4009f085649b6da3f1647821d3e760e6724b019e08d022080c8050bc29"
),
ROOT /(
"architecture_gate/prepared_inputs_v2/"
"yolo11m_p2_projected_coco_init.pt"
):(
"eef43a83adb99ee9b3d9e75c25d3454488b28ad5f86ac67fcf850351c3e848dd"
),
}

EXPECTED_V3_MANIFEST_SHA256 =(
"7c96b15970798025e2482613c6fb0304d972e480a7166c5f2e8817b1e7ec93a9"
)

EXPECTED_TYPED_QUOTAS ={
"small_joint_not_detected":7800 ,
"dfine_supported_yolo_failure":3900 ,
"yolo_geometric_failure":1950 ,
"yolo_low_confidence":1950 ,
"yolo_classification_failure":1365 ,
"other_joint_not_detected":975 ,
"small_high_or_critical":975 ,
"other_high_or_critical":585 ,
}

EXPECTED_VIEWS =19_500 
EXPECTED_COMMON_BASE =45_500 
EXPECTED_CORE =65_000 
RISK_BIN_COUNT =20 
MAX_MEAN_RISK_GAP =0.03 


def sha256_file (path :Path )->str :
    digest =hashlib .sha256 ()
    with path .open ("rb")as handle :
        while True :
            block =handle .read (8 *1024 *1024 )
            if not block :
                break 
            digest .update (block )
    return digest .hexdigest ()


def load_ids (path :Path )->list [int ]:
    data =json .loads (path .read_text ())

    if isinstance (data ,list ):
        values =data 
    elif isinstance (data ,dict ):
        values =None 
        for key in (
        "image_ids",
        "train_core_image_ids",
        "ids",
        ):
            if key in data and isinstance (data [key ],list ):
                values =data [key ]
                break 
        if values is None :
            raise KeyError (path )
    else :
        raise TypeError (type (data ).__name__ )

    return [int (value )for value in values ]


def risk_bin (value :float )->int :
    if not 0.0 <=value <=1.0 :
        raise AssertionError (value )
    return min (
    RISK_BIN_COUNT -1 ,
    int (value *RISK_BIN_COUNT ),
    )


def audit_csv (path :Path )->dict :
    annotations =set ()
    images =set ()
    classes =Counter ()
    scales =Counter ()
    bins =Counter ()
    groups =Counter ()
    risks =[]

    with path .open (
    "r",
    encoding ="utf-8",
    newline ="",
    )as handle :
        reader =csv .DictReader (handle )

        for row in reader :
            annotation_id =int (row ["annotation_id"])
            image_id =int (row ["image_id"])
            category_id =int (
            row ["target_category_id"]
            )
            scale =row ["scale"]
            score =float (row ["risk_score"])
            group =row ["diagnostic_group"]

            if annotation_id ==1005741 :
                raise AssertionError (
                "Excluded duplicate was selected"
                )
            if category_id ==6 :
                raise AssertionError (
                "Unsupported train class was targeted"
                )

            if annotation_id in annotations :
                raise AssertionError (
                f"Duplicate annotation: {annotation_id }"
                )
            if image_id in images :
                raise AssertionError (
                f"Duplicate source image: {image_id }"
                )

            annotations .add (annotation_id )
            images .add (image_id )
            classes [category_id ]+=1 
            scales [scale ]+=1 
            bins [risk_bin (score )]+=1 
            groups [group ]+=1 
            risks .append (score )

    return {
    "count":len (annotations ),
    "unique_images":len (images ),
    "classes":dict (sorted (classes .items ())),
    "scales":dict (sorted (scales .items ())),
    "risk_bins":dict (sorted (bins .items ())),
    "groups":dict (sorted (groups .items ())),
    "mean_risk":sum (risks )/len (risks ),
    "annotation_ids":annotations ,
    "image_ids":images ,
    }


print ("=== VERIFYING V3 ARTIFACT MANIFEST ===")

actual_manifest_sha256 =sha256_file (
AUDIT_MANIFEST 
)
print ("expected:",EXPECTED_V3_MANIFEST_SHA256 )
print ("actual  :",actual_manifest_sha256 )

if actual_manifest_sha256 !=(
EXPECTED_V3_MANIFEST_SHA256 
):
    raise AssertionError ("V3 manifest mismatch")

manifest =json .loads (
AUDIT_MANIFEST .read_text ()
)

for name ,record in manifest ["files"].items ():
    path =AUDIT_V3 /name 
    if not path .is_file ():
        raise FileNotFoundError (path )

    actual =sha256_file (path )
    expected =record ["sha256"]

    print (f"{name }:")
    print (" expected:",expected )
    print (" actual  :",actual )

    if actual !=expected :
        raise AssertionError (path )

print ()
print ("=== VERIFYING EXTERNAL DEPENDENCIES ===")

for path ,expected in EXTERNAL_DEPENDENCIES .items ():
    if not path .is_file ():
        raise FileNotFoundError (path )

    actual =sha256_file (path )
    print (path )
    print (" expected:",expected )
    print (" actual  :",actual )

    if actual !=expected :
        raise AssertionError (path )

summary =json .loads (
AUDIT_SUMMARY .read_text ()
)

actual_script_sha256 =sha256_file (V3_SCRIPT )
expected_script_sha256 =summary ["script_sha256"]

print ()
print ("=== VERIFYING V3 SCRIPT ===")
print ("expected:",expected_script_sha256 )
print ("actual  :",actual_script_sha256 )

if actual_script_sha256 !=expected_script_sha256 :
    raise AssertionError ("V3 script mismatch")

print ()
print ("=== AUDITING SELECTED ROWS ===")

typed =audit_csv (TYPED_CSV )
scalar =audit_csv (SCALAR_CSV )

for name ,result in (
("typed",typed ),
("scalar",scalar ),
):
    print (
    name ,
    "count=",result ["count"],
    "unique_images=",result ["unique_images"],
    "mean_risk=",result ["mean_risk"],
    )

    if result ["count"]!=EXPECTED_VIEWS :
        raise AssertionError ((name ,result ["count"]))
    if result ["unique_images"]!=EXPECTED_VIEWS :
        raise AssertionError (
        (name ,result ["unique_images"])
        )

if typed ["classes"]!=scalar ["classes"]:
    raise AssertionError ("Class mismatch")

if typed ["scales"]!=scalar ["scales"]:
    raise AssertionError ("Scale mismatch")

if typed ["risk_bins"]!=scalar ["risk_bins"]:
    raise AssertionError ("Risk-bin mismatch")

if typed ["groups"]!=dict (sorted (
EXPECTED_TYPED_QUOTAS .items ()
)):
    raise AssertionError ({
    "actual":typed ["groups"],
    "expected":EXPECTED_TYPED_QUOTAS ,
    })

mean_risk_gap =abs (
typed ["mean_risk"]
-scalar ["mean_risk"]
)
if mean_risk_gap >MAX_MEAN_RISK_GAP :
    raise AssertionError (mean_risk_gap )

print ("class_counts:",typed ["classes"])
print ("scale_counts:",typed ["scales"])
print ("risk_bin_counts:",typed ["risk_bins"])
print ("typed_groups:",typed ["groups"])
print ("mean_risk_gap:",mean_risk_gap )

print ()
print ("=== VERIFYING VIEW ACCOUNTING ===")

core_ids =load_ids (CORE_IDS )
common_ids =load_ids (COMMON_BASE )
supplement_ids =load_ids (UNIFORM_SUPPLEMENT )

if len (core_ids )!=EXPECTED_CORE :
    raise AssertionError (len (core_ids ))
if len (common_ids )!=EXPECTED_COMMON_BASE :
    raise AssertionError (len (common_ids ))
if len (supplement_ids )!=EXPECTED_VIEWS :
    raise AssertionError (len (supplement_ids ))

core_set =set (core_ids )
common_set =set (common_ids )
supplement_set =set (supplement_ids )

if len (core_set )!=EXPECTED_CORE :
    raise AssertionError ("Duplicate core IDs")
if common_set &supplement_set :
    raise AssertionError ("Base/supplement overlap")
if common_set |supplement_set !=core_set :
    raise AssertionError (
    "Base/supplement do not reconstruct core"
    )

print ("train_core:",len (core_set ))
print ("common_base:",len (common_set ))
print ("uniform_supplement:",len (supplement_set ))
print ("PASS: view accounting")

if FREEZE_PATH .exists ():
    raise FileExistsError (
    f"Refusing to overwrite: {FREEZE_PATH }"
    )

FREEZE_PATH .parent .mkdir (
parents =True ,
exist_ok =True ,
)

record ={
"version":"cafr_sampling_protocol_v1",
"status":"frozen_before_method_gate_training",
"created_utc":datetime .now (
timezone .utc 
).isoformat (),
"selected_protocol":"sampling_audit_v3",
"selected_protocol_manifest":str (
AUDIT_MANIFEST 
),
"selected_protocol_manifest_sha256":(
actual_manifest_sha256 
),
"selection_script":str (V3_SCRIPT ),
"selection_script_sha256":(
actual_script_sha256 
),
"frozen_files":manifest ["files"],
"design":summary ["design"],
"verified_results":{
"views_per_object_centric_arm":(
EXPECTED_VIEWS 
),
"common_original_views":(
EXPECTED_COMMON_BASE 
),
"total_views_per_arm":EXPECTED_CORE ,
"target_class_counts":typed ["classes"],
"target_scale_counts":typed ["scales"],
"target_risk_bin_counts":(
typed ["risk_bins"]
),
"typed_failure_quotas":typed ["groups"],
"typed_mean_risk":typed ["mean_risk"],
"scalar_mean_risk":scalar ["mean_risk"],
"mean_risk_gap":mean_risk_gap ,
"annotation_overlap":len (
typed ["annotation_ids"]
&scalar ["annotation_ids"]
),
"source_image_overlap":len (
typed ["image_ids"]
&scalar ["image_ids"]
),
},
"discarded_protocol_audits":{
"sampling_audit_v1":{
"manifest_sha256":(
"77a84ab2b02198c687b3eaf449d0aae78902a1bad9df7a8ee38b6e495a533f10"
),
"reason":(
"Scalar control was materially easier: "
"mean risk 0.711 versus Typed 0.856."
),
"training_performed":False ,
},
"sampling_audit_v2":{
"manifest_sha256":(
"7d7e7b8e02d7104eb92c05a6bb10fdcc0639963ecb53626f9db7ba34c742e5a7"
),
"reason":(
"Hard top-risk truncation was pathological: "
"mean risk 0.985 and excessive concentration "
"on small jointly missed objects."
),
"training_performed":False ,
},
},
"external_dependencies":{
str (path ):expected 
for path ,expected 
in EXTERNAL_DEPENDENCIES .items ()
},
"images_materialized":False ,
"training_started":False ,
"method_training_hyperparameters_frozen":False ,
}

temporary =Path (
str (FREEZE_PATH )
+f".incomplete-{os .getpid ()}"
)

temporary .write_text (
json .dumps (
record ,
indent =2 ,
ensure_ascii =False ,
)+"\n",
encoding ="utf-8",
)
os .replace (temporary ,FREEZE_PATH )

print ()
print ("="*92 )
print ("CAFR SAMPLING PROTOCOL FROZEN")
print ("="*92 )
print ("protocol:",FREEZE_PATH )
print (
"protocol_sha256:",
sha256_file (FREEZE_PATH ),
)
print ("selected_protocol: sampling_audit_v3")
print ("views_per_arm:",EXPECTED_CORE )
print (
"object_centric_views_per_arm:",
EXPECTED_VIEWS ,
)
print ("mean_risk_gap:",mean_risk_gap )
print ("PASS: matched Scalar and Typed sampling is immutable")
print ("NOTHING TRAINED")
