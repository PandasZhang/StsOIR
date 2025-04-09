from .CosineFaceLoss import CosineFaceLoss
from .SepcializedMarginLoss import SepcializedMarginLoss
from .SupConLoss import SupConLoss
from torch import nn
loss_map = {
                'CrossEntropyLoss': nn.CrossEntropyLoss(), 
                'CosineFaceLoss': CosineFaceLoss(),
                'TripletMarginWithDistanceLoss': nn.TripletMarginWithDistanceLoss(),
                'SepcializedMarginLoss': SepcializedMarginLoss(),
                'SupConLoss': SupConLoss(),
            }
