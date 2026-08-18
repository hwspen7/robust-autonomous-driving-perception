
import torch 
import torchvision 

from ...core import register 
from .utils import IntermediateLayerGetter 

__all__ =["TorchVisionModel"]


@register ()
class TorchVisionModel (torch .nn .Module ):
    def __init__ (self ,name ,return_layers ,weights =None ,**kwargs )->None :
        super ().__init__ ()

        if weights is not None :
            weights =getattr (torchvision .models .get_model_weights (name ),weights )

        model =torchvision .models .get_model (name ,weights =weights ,**kwargs )


        if hasattr (model ,"features"):
            model =IntermediateLayerGetter (model .features ,return_layers )
        else :
            model =IntermediateLayerGetter (model ,return_layers )

        self .model =model 

    def forward (self ,x ):
        return self .model (x )
















