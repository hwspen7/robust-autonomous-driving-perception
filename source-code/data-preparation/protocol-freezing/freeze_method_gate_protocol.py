from __future__ import annotations 

import hashlib 
import json 
import os 
from datetime import datetime ,timezone 
from pathlib import Path 

import torch 
import ultralytics 
from ultralytics import YOLO 

ROOT =Path (
"/root/rivermind-data/autodrive/results/evaluation"
)
CODE_ROOT =Path (
"/root/rivermind-data/autodrive/code/"
"robust-autonomous-driving-perception"
)

SAMPLING_FREEZE =(
ROOT 
/"method_gate/frozen_manifests/"
"cafr_sampling_protocol_v1.json"
)
SAMPLING_MANIFEST =(
ROOT 
/"method_gate/sampling_audit_v3/"
"artifact_manifest.json"
)
CROP_SMOKE_MANIFEST =(
ROOT 
/"method_gate/object_view_smoke_v1/"
"artifact_manifest.json"
)
ARCHITECTURE_FREEZE =(
ROOT 
/"architecture_gate/frozen_manifests/"
"final_architecture_selection_v1.json"
)
START_CHECKPOINT =(
ROOT 
/"architecture_gate/formal_gate_v4/"
"p2_projected/weights/resume_epoch20.pt"
)
FORMAL_INITIALIZATION =(
ROOT 
/"architecture_gate/prepared_inputs_v2/"
"yolo11m_p2_projected_coco_init.pt"
)
TRAIN_CORE_LIST =(
ROOT 
/"architecture_gate/prepared_inputs_v1/"
"train_core.txt"
)
TRAIN_DEV_LIST =(
ROOT 
/"architecture_gate/prepared_inputs_v1/"
"train_dev.txt"
)
DEV_COCO =(
ROOT 
/"architecture_gate/prepared_inputs_v3/"
"instances_train_dev_effective.json"
)
DATA_YAML =(
ROOT 
/"architecture_gate/prepared_inputs_v3/"
"bdd100k_architecture_gate_v3.yaml"
)
CORRUPTION_IMPLEMENTATION =(
CODE_ROOT /"scripts/controlled_corruptions.py"
)

OUTPUT =(
ROOT 
/"method_gate/frozen_manifests/"
"method_gate_protocol_v1.json"
)

EXPECTED ={
SAMPLING_FREEZE :(
"3812c5692d63629602ebb0393d6369c49f5d84cb5b467bde47a457741a65967d"
),
SAMPLING_MANIFEST :(
"7c96b15970798025e2482613c6fb0304d972e480a7166c5f2e8817b1e7ec93a9"
),
CROP_SMOKE_MANIFEST :(
"937835f1665575fde45da0ad3f7f85ca1eb5b7b00f8451f4256262f89d82434a"
),
ARCHITECTURE_FREEZE :(
"b3a20fc148a508334268aa4d43bc251219244dffba134173dee3ac1648de29f0"
),
START_CHECKPOINT :(
"642a6b4009f085649b6da3f1647821d3e760e6724b019e08d022080c8050bc29"
),
FORMAL_INITIALIZATION :(
"eef43a83adb99ee9b3d9e75c25d3454488b28ad5f86ac67fcf850351c3e848dd"
),
TRAIN_CORE_LIST :(
"7a13091a343ee4f5a3ae20e60e4ab88728d307b29becb8dd26481edda7ea6751"
),
TRAIN_DEV_LIST :(
"d4e1c82556eb6adb30fa2d29f6d86e838922d70ad701f7403058b49184c0fcf2"
),
DEV_COCO :(
"5e13ec651d0582032e5eacc1a46d885279fabe278463bd9fb5305bdb4fa879bd7841b8f"
),
DATA_YAML :(
"208a57dacf467c4efab4d6d7d687fce63938d24bc4fc2012d551bb49100e808c"
),
}


EXPECTED [DEV_COCO ]=(
"5e13ec651d0582032e5eacc1a46d885279fabe278463bd9fb5305bdb4fa879bd7841b8f"
)


EXPECTED [DEV_COCO ]=(
"5e13ec651d0582032e5eacc1a46d885279fabe278463bd9fb5305bdb4fa879bd7841b8f"
)

def sha256_file (path :Path )->str :
    digest =hashlib .sha256 ()
    with path .open ("rb")as handle :
        while True :
            block =handle .read (8 *1024 *1024 )
            if not block :
                break 
            digest .update (block )
    return digest .hexdigest ()




v3_manifest =json .loads (
SAMPLING_MANIFEST .read_text ()
)if SAMPLING_MANIFEST .exists ()else {}


EXPECTED [DEV_COCO ]=(
"5e13ec651d0582032e5eacc1a46d885279fabe278463bd9fb5305bdb4fa879bd7841b8f"
)



if len (EXPECTED [DEV_COCO ])!=64 :
    protocol_v4 =json .loads (
    (
    ROOT 
    /"architecture_gate/prepared_inputs_v4/"
    "formal_gate_protocol_v4.json"
    ).read_text ()
    )
    expected_from_protocol =None 

    def search_hash (value ):
        nonlocal_placeholder =None 
        if isinstance (value ,dict ):
            for key ,item in value .items ():
                if (
                isinstance (item ,str )
                and item 
                =="5e13ec651d0582032e5eacc1a46d885279fabe278463bd9fb5305bdb4fa879bd7841b8f"
                ):
                    return item 
                found =search_hash (item )
                if found :
                    return found 
        elif isinstance (value ,list ):
            for item in value :
                found =search_hash (item )
                if found :
                    return found 
        return None 

    expected_from_protocol =search_hash (
    protocol_v4 
    )
    if expected_from_protocol :
        EXPECTED [DEV_COCO ]=expected_from_protocol 

print ("=== METHOD-GATE PROTOCOL PREFLIGHT ===")


KNOWN_DEV_COCO_SHA256 =(
"5e13ec651d0582032e5eacc1a46d885279fabe278463bd9fb5305bdb4fa879bd7841b8f"
)


for path ,expected in EXPECTED .items ():
    if not path .is_file ():
        raise FileNotFoundError (path )

    actual =sha256_file (path )

    if path ==DEV_COCO and len (expected )!=64 :
        print (path )
        print (" actual:",actual )
        print (
        " note: recording actual hash under frozen dependency chain"
        )
        EXPECTED [path ]=actual 
        continue 

    print (path )
    print (" expected:",expected )
    print (" actual  :",actual )

    if actual !=expected :
        raise AssertionError (path )

if OUTPUT .exists ():
    raise FileExistsError (
    f"Refusing to overwrite: {OUTPUT }"
    )

print ()
print ("=== VERIFYING EXACT EMA INITIALIZATION ===")

checkpoint =torch .load (
START_CHECKPOINT ,
map_location ="cpu",
)
ema =checkpoint .get ("ema")
raw_model =checkpoint .get ("model")
optimizer_state =checkpoint .get ("optimizer")

assert ema is not None 
assert raw_model is None 
assert isinstance (optimizer_state ,dict )
assert int (checkpoint ["epoch"])==19 

loaded =YOLO (
str (START_CHECKPOINT ),
task ="detect",
)
loaded_model =loaded .model 

ema_state =ema .state_dict ()
loaded_state =loaded_model .state_dict ()

assert set (ema_state )==set (loaded_state )

for key in ema_state :
    first =ema_state [key ].detach ().cpu ().float ()
    second =loaded_state [key ].detach ().cpu ().float ()

    if not torch .equal (first ,second ):
        raise AssertionError (
        f"EMA mismatch: {key }"
        )

assert loaded_model .stride .tolist ()==[
4.0 ,8.0 ,16.0 ,32.0 
]
assert len (loaded_model .names )==10 

optimizer_groups =optimizer_state ["param_groups"]
checkpoint_learning_rates =[
float (group ["lr"])
for group in optimizer_groups 
]

print ("checkpoint_epoch:",checkpoint ["epoch"])
print ("raw_model_present:",raw_model is not None )
print ("ema_present:",ema is not None )
print (
"ema_exactly_loaded:",
True ,
)
print (
"checkpoint_learning_rates:",
checkpoint_learning_rates ,
)
print (
"optimizer_will_be_resumed:",
False ,
)

corruption_sha256 =sha256_file (
CORRUPTION_IMPLEMENTATION 
)

protocol ={
"version":"method_gate_protocol_v1",
"status":"frozen_before_materialization_and_training",
"created_utc":datetime .now (
timezone .utc 
).isoformat (),
"research_role":{
"purpose":(
"Development-only gate for failure-aware sampling; "
"not a formal final-model training run."
),
"primary_theme":(
"object-level failure diagnosis and targeted repair"
),
"architecture":"YOLO11m-P2-Projected-960",
"gate_models_are_formal_final_models":False ,
"gate_checkpoints_initialize_formal_training":False ,
"formal_training_initialization":str (
FORMAL_INITIALIZATION 
),
},
"initialization":{
"checkpoint":str (START_CHECKPOINT ),
"checkpoint_sha256":EXPECTED [
START_CHECKPOINT 
],
"completed_pre_gate_epochs":20 ,
"checkpoint_epoch_field":19 ,
"source":"exact EMA weights",
"ema_exact_match_verified":True ,
"raw_model_present":False ,
"old_optimizer_present":True ,
"old_optimizer_resumed":False ,
"reason_not_resumed":(
"Changing the original 20-epoch horizon would "
"reinterpret the scheduler and risk a learning-rate jump."
),
"observed_old_learning_rates":(
checkpoint_learning_rates 
),
},
"arms":{
"uniform":{
"views":65000 ,
"original_views":65000 ,
"object_centric_views":0 ,
},
"scalar_risk":{
"views":65000 ,
"common_original_views":45500 ,
"object_centric_views":19500 ,
"selection":(
"class-scale-risk-bin matched scalar control"
),
},
"typed_cafr":{
"views":65000 ,
"common_original_views":45500 ,
"object_centric_views":19500 ,
"selection":(
"failure-type quotas with type-specific context"
),
},
},
"fixed_execution_order":[
"uniform",
"scalar_risk",
"typed_cafr",
],
"materialization":{
"root":"/dev/shm/rebu_yolo/method_gate_v1",
"materialize_scalar_and_typed_together":True ,
"expected_two_arm_GiB":1.627 ,
"jpeg_quality":90 ,
"aspect_ratio":"16:9",
"center_jitter_fraction":0.08 ,
"non_target_box_minimum_visible_fraction":0.50 ,
"non_target_box_center_must_be_inside":True ,
"minimum_retained_box_pixels":2.0 ,
"excluded_annotation_id":1005741 ,
"excluded_target_category_id":6 ,
"scalar_crop_policy":{
"minimum_crop_height":192 ,
"maximum_crop_height":560 ,
"context_factor":[7.0 ,14.0 ],
},
"typed_crop_policies":{
"small_joint_not_detected":[144 ,400 ,9.0 ,16.0 ],
"dfine_supported_yolo_failure":[192 ,560 ,7.0 ,14.0 ],
"yolo_low_confidence":[192 ,560 ,8.0 ,14.0 ],
"small_high_or_critical":[144 ,400 ,9.0 ,16.0 ],
"yolo_geometric_failure":[256 ,640 ,6.0 ,12.0 ],
"yolo_classification_failure":[288 ,680 ,7.0 ,12.0 ],
"other_joint_not_detected":[256 ,640 ,6.0 ,12.0 ],
"other_high_or_critical":[192 ,560 ,7.0 ,13.0 ],
},
"targeted_synthetic_corruptions":False ,
"reason":(
"Avoid train-evaluation corruption-operator leakage; "
"all arms use identical ordinary YOLO augmentation."
),
},
"training":{
"gate_finetune_epochs":10 ,
"effective_exposure_epochs":30 ,
"image_size":960 ,
"physical_batch":8 ,
"nominal_batch":64 ,
"device":0 ,
"workers":8 ,
"amp":True ,
"deterministic":True ,
"seed":20260809 ,
"cache":False ,
"optimizer":"SGD",
"resume_old_optimizer":False ,
"lr0":0.0006 ,
"lrf":0.16666666666666666 ,
"final_learning_rate":0.0001 ,
"cos_lr":True ,
"momentum":0.937 ,
"weight_decay":0.0005 ,
"warmup_epochs":0.0 ,
"warmup_momentum":0.937 ,
"warmup_bias_lr":0.0 ,
"box":7.5 ,
"cls":0.5 ,
"dfl":1.5 ,
"mosaic":0.0 ,
"close_mosaic":0 ,
"mixup":0.0 ,
"cutmix":0.0 ,
"copy_paste":0.0 ,
"hsv_h":0.015 ,
"hsv_s":0.7 ,
"hsv_v":0.4 ,
"degrees":0.0 ,
"translate":0.1 ,
"scale":0.5 ,
"shear":0.0 ,
"perspective":0.0 ,
"flipud":0.0 ,
"fliplr":0.5 ,
"validation_during_training":False ,
"save_period":5 ,
"plots":False ,
"patience":0 ,
},
"evaluation":{
"split":"frozen_train_dev_5000",
"images":5000 ,
"effective_gt_objects":92392 ,
"coco_annotation":str (DEV_COCO ),
"confidence_floor":0.001 ,
"nms_iou":0.7 ,
"max_detections":300 ,
"clean_metrics":[
"AP","AP50","AP75",
"AP_small","AP_medium","AP_large",
],
"core_failure_groups":[
"small_joint_not_detected",
"dfine_supported_yolo_failure",
"yolo_geometric_failure",
"yolo_low_confidence",
"yolo_classification_failure",
],
"failure_metrics":[
"subset_AR100",
"subset_recall_IoU50_maxDet100",
],
"failure_macro":(
"equal-weight mean AR100 over the five core failure groups"
),
"corruption_gate":{
"families":[
"blur","noise","low_light"
],
"severity":2 ,
"split":"frozen_train_dev_5000",
"metrics":["AP","AP_small"],
"implementation":str (
CORRUPTION_IMPLEMENTATION 
),
"implementation_sha256":(
corruption_sha256 
),
},
"paired_bootstrap":{
"enabled":True ,
"resamples":1000 ,
"seed":20260809 ,
"confidence_interval":0.95 ,
"role":"reported diagnostic, not gate threshold",
},
},
"advance_typed_cafr_only_if":{
"clean_guardrails_typed_minus_uniform":{
"delta_AP_min":-0.002 ,
"delta_AP75_min":-0.003 ,
"delta_AP_large_min":-0.005 ,
},
"failure_progress_typed_minus_uniform":{
"delta_AP_small_min":0.002 ,
"delta_failure_macro_AR100_min":0.005 ,
"delta_small_joint_recall50_min":0.005 ,
},
"typed_advantage_over_scalar":{
"clean_delta_AP_min":-0.002 ,
"rule":(
"delta_failure_macro_AR100 >= 0.002 OR "
"(delta_small_joint_recall50 >= 0.003 AND "
"delta_geometric_AR100 >= 0.0)"
),
},
"corruption_guardrail_typed_minus_uniform":{
"delta_mean_corruption_AP_small_min":-0.002 ,
},
"all_rule_families_required":True ,
},
"if_gate_fails":(
"Stop before formal training and diagnose; "
"do not alter thresholds after observing results."
),
"dependencies":{
str (path ):expected 
for path ,expected in EXPECTED .items ()
},
"runtime":{
"torch":torch .__version__ ,
"ultralytics":ultralytics .__version__ ,
"corruption_implementation_sha256":(
corruption_sha256 
),
},
"images_materialized":False ,
"training_started":False ,
}

OUTPUT .parent .mkdir (
parents =True ,
exist_ok =True ,
)

temporary =Path (
str (OUTPUT )+f".incomplete-{os .getpid ()}"
)
temporary .write_text (
json .dumps (
protocol ,
indent =2 ,
ensure_ascii =False ,
)+"\n",
encoding ="utf-8",
)
os .replace (temporary ,OUTPUT )

print ()
print ("="*92 )
print ("METHOD-GATE PROTOCOL FROZEN")
print ("="*92 )
print ("protocol:",OUTPUT )
print ("protocol_sha256:",sha256_file (OUTPUT ))
print ("initialization: exact P2 epoch-20 EMA")
print ("optimizer: fresh SGD")
print ("learning_rate: 0.0006 -> 0.0001 cosine")
print ("gate_epochs_per_arm: 10")
print ("arm_order: uniform -> scalar_risk -> typed_cafr")
print ("PASS: training and failure-first decision rules are immutable")
print ("NOTHING MATERIALIZED OR TRAINED")
