from __future__ import annotations 

import hashlib 
import math 
from typing import Any 

import numpy as np 
from PIL import Image ,ImageDraw ,ImageEnhance ,ImageFilter 


GLOBAL_SEED =20260809 
CORRUPTION_CONFIG :dict [str ,dict [int ,dict [str ,Any ]]]={
"low_light":{
1 :{
"gamma":1.25 ,
"gain":0.90 ,
},
2 :{
"gamma":1.60 ,
"gain":0.78 ,
},
3 :{
"gamma":2.00 ,
"gain":0.65 ,
},
},

"fog":{
1 :{
"transmission":0.82 ,
"variation":0.04 ,
"atmospheric_light":245.0 ,
},
2 :{
"transmission":0.68 ,
"variation":0.06 ,
"atmospheric_light":245.0 ,
},
3 :{
"transmission":0.52 ,
"variation":0.08 ,
"atmospheric_light":245.0 ,
},
},

"rain":{
1 :{
"density":0.00018 ,
"length":12 ,
"width":1 ,
"alpha":55 ,
"blur_radius":0.4 ,
"brightness":0.98 ,
},
2 :{
"density":0.00035 ,
"length":18 ,
"width":1 ,
"alpha":80 ,
"blur_radius":0.7 ,
"brightness":0.95 ,
},
3 :{
"density":0.00055 ,
"length":24 ,
"width":2 ,
"alpha":110 ,
"blur_radius":1.0 ,
"brightness":0.92 ,
},
},

"blur":{
1 :{
"radius":1.0 ,
},
2 :{
"radius":2.0 ,
},
3 :{
"radius":3.0 ,
},
},

"noise":{
1 :{
"std":5.0 ,
},
2 :{
"std":10.0 ,
},
3 :{
"std":18.0 ,
},
},
}


CORRUPTION_NAMES =tuple (
CORRUPTION_CONFIG .keys ()
)

SEVERITIES =(
1 ,
2 ,
3 ,
)


def variant_name (
corruption :str ,
severity :int ,
)->str :
    validate_corruption (
    corruption ,
    severity ,
    )

    return (
    f"{corruption }_s{severity }"
    )


def validate_corruption (
corruption :str ,
severity :int ,
)->None :
    if corruption not in CORRUPTION_CONFIG :
        raise ValueError (
        f"Unknown corruption: {corruption }"
        )

    if severity not in CORRUPTION_CONFIG [
    corruption 
    ]:
        raise ValueError (
        f"Invalid severity={severity } "
        f"for corruption={corruption }"
        )


def deterministic_seed (
*,
image_key :str ,
corruption :str ,
severity :int ,
)->int :

    validate_corruption (
    corruption ,
    severity ,
    )

    text =(
    f"{GLOBAL_SEED }|"
    f"{image_key }|"
    f"{corruption }|"
    f"{severity }"
    )

    digest =hashlib .sha256 (
    text .encode (
    "utf-8"
    )
    ).digest ()

    return int .from_bytes (
    digest [:8 ],
    byteorder ="little",
    signed =False ,
    )


def build_rng (
*,
image_key :str ,
corruption :str ,
severity :int ,
)->np .random .Generator :
    seed =deterministic_seed (
    image_key =image_key ,
    corruption =corruption ,
    severity =severity ,
    )

    return np .random .default_rng (
    seed 
    )






def apply_low_light (
image :Image .Image ,
config :dict [str ,Any ],
)->Image .Image :

    array =(
    np .asarray (
    image .convert ("RGB"),
    dtype =np .float32 ,
    )
    /255.0 
    )

    gamma =float (
    config ["gamma"]
    )

    gain =float (
    config ["gain"]
    )

    output =(
    np .power (
    array ,
    gamma ,
    )
    *gain 
    )

    output =np .clip (
    output *255.0 ,
    0.0 ,
    255.0 ,
    ).astype (
    np .uint8 
    )

    return Image .fromarray (
    output ,
    mode ="RGB",
    )






def build_smooth_noise_map (
*,
width :int ,
height :int ,
rng :np .random .Generator ,
)->np .ndarray :

    small_width =max (
    8 ,
    width //64 ,
    )

    small_height =max (
    8 ,
    height //64 ,
    )

    raw =rng .random (
    (
    small_height ,
    small_width ,
    )
    )

    raw_image =Image .fromarray (
    np .uint8 (
    raw *255.0 
    ),
    mode ="L",
    )

    smooth_image =raw_image .resize (
    (
    width ,
    height ,
    ),
    resample =(
    Image .Resampling .BICUBIC 
    ),
    )

    smooth =(
    np .asarray (
    smooth_image ,
    dtype =np .float32 ,
    )
    /255.0 
    )


    return (
    smooth -0.5 
    )*2.0 


def apply_fog (
image :Image .Image ,
config :dict [str ,Any ],
rng :np .random .Generator ,
)->Image .Image :
    array =np .asarray (
    image .convert ("RGB"),
    dtype =np .float32 ,
    )

    height ,width =(
    array .shape [:2 ]
    )

    transmission_base =float (
    config ["transmission"]
    )

    variation =float (
    config ["variation"]
    )

    atmospheric_light =float (
    config [
    "atmospheric_light"
    ]
    )

    variation_map =(
    build_smooth_noise_map (
    width =width ,
    height =height ,
    rng =rng ,
    )
    )

    transmission =(
    transmission_base 
    +variation 
    *variation_map 
    )

    transmission =np .clip (
    transmission ,
    0.20 ,
    1.0 ,
    )

    transmission =(
    transmission [
    ...,
    None 
    ]
    )

    output =(
    array 
    *transmission 
    +atmospheric_light 
    *(
    1.0 
    -transmission 
    )
    )

    output =np .clip (
    output ,
    0.0 ,
    255.0 ,
    ).astype (
    np .uint8 
    )

    return Image .fromarray (
    output ,
    mode ="RGB",
    )






def apply_rain (
image :Image .Image ,
config :dict [str ,Any ],
rng :np .random .Generator ,
)->Image .Image :

    base =image .convert (
    "RGB"
    )

    width ,height =(
    base .size 
    )

    density =float (
    config ["density"]
    )

    length =int (
    config ["length"]
    )

    line_width =int (
    config ["width"]
    )

    alpha =int (
    config ["alpha"]
    )

    blur_radius =float (
    config ["blur_radius"]
    )

    brightness =float (
    config ["brightness"]
    )

    num_streaks =max (
    1 ,
    int (
    width 
    *height 
    *density 
    ),
    )

    overlay =Image .new (
    "RGBA",
    (
    width ,
    height ,
    ),
    (
    0 ,
    0 ,
    0 ,
    0 ,
    ),
    )

    draw =ImageDraw .Draw (
    overlay 
    )

    for _ in range (
    num_streaks 
    ):
        x =int (
        rng .integers (
        -length ,
        width +length ,
        )
        )

        y =int (
        rng .integers (
        -length ,
        height +length ,
        )
        )

        dx_max =max (
        3 ,
        length //3 ,
        )

        dx =int (
        rng .integers (
        2 ,
        dx_max ,
        )
        )

        draw .line (
        (
        x ,
        y ,
        x +dx ,
        y +length ,
        ),
        fill =(
        210 ,
        220 ,
        230 ,
        alpha ,
        ),
        width =line_width ,
        )

    if blur_radius >0 :
        overlay =overlay .filter (
        ImageFilter .GaussianBlur (
        radius =blur_radius 
        )
        )

    darkened =(
    ImageEnhance .Brightness (
    base 
    ).enhance (
    brightness 
    )
    )

    output =Image .alpha_composite (
    darkened .convert (
    "RGBA"
    ),
    overlay ,
    )

    return output .convert (
    "RGB"
    )






def apply_blur (
image :Image .Image ,
config :dict [str ,Any ],
)->Image .Image :
    radius =float (
    config ["radius"]
    )

    return (
    image 
    .convert ("RGB")
    .filter (
    ImageFilter .GaussianBlur (
    radius =radius 
    )
    )
    )






def apply_noise (
image :Image .Image ,
config :dict [str ,Any ],
rng :np .random .Generator ,
)->Image .Image :
    array =np .asarray (
    image .convert ("RGB"),
    dtype =np .float32 ,
    )

    std =float (
    config ["std"]
    )

    noise =rng .normal (
    loc =0.0 ,
    scale =std ,
    size =array .shape ,
    ).astype (
    np .float32 
    )

    output =np .clip (
    array +noise ,
    0.0 ,
    255.0 ,
    ).astype (
    np .uint8 
    )

    return Image .fromarray (
    output ,
    mode ="RGB",
    )






def apply_corruption (
image :Image .Image ,
*,
image_key :str ,
corruption :str ,
severity :int ,
)->Image .Image :

    validate_corruption (
    corruption ,
    severity ,
    )

    source =image .convert (
    "RGB"
    )

    source_size =source .size 

    config =(
    CORRUPTION_CONFIG [
    corruption 
    ][
    severity 
    ]
    )

    rng =build_rng (
    image_key =image_key ,
    corruption =corruption ,
    severity =severity ,
    )

    if corruption =="low_light":
        output =apply_low_light (
        source ,
        config ,
        )

    elif corruption =="fog":
        output =apply_fog (
        source ,
        config ,
        rng ,
        )

    elif corruption =="rain":
        output =apply_rain (
        source ,
        config ,
        rng ,
        )

    elif corruption =="blur":
        output =apply_blur (
        source ,
        config ,
        )

    elif corruption =="noise":
        output =apply_noise (
        source ,
        config ,
        rng ,
        )

    else :
        raise RuntimeError (
        "Unreachable corruption branch."
        )

    if output .size !=source_size :
        raise RuntimeError (
        "Controlled corruption changed image geometry."
        )

    return output .convert (
    "RGB"
    )


def image_sha256 (
image :Image .Image ,
)->str :

    rgb =image .convert (
    "RGB"
    )

    array =np .asarray (
    rgb ,
    dtype =np .uint8 ,
    )

    digest =hashlib .sha256 ()

    digest .update (
    str (
    array .shape 
    ).encode (
    "utf-8"
    )
    )

    digest .update (
    array .tobytes ()
    )

    return digest .hexdigest ()


def all_variants ()->list [
tuple [str ,int ]
]:
    return [
    (
    corruption ,
    severity ,
    )
    for corruption 
    in CORRUPTION_NAMES 
    for severity 
    in SEVERITIES 
    ]
