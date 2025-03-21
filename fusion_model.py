import torch
import torch.nn as nn
import torch.nn.functional as F
from metamodel import MetadataMLP
from vit import ViT
from torchsummary import summary

class CombinedModel(nn.Module):
    def __init__(self, model_vis, model_meta, num_classes=2):
        super(CombinedModel, self).__init__()
        self.model_vis = model_vis     
        self.model_meta = model_meta    
        self.w_vis = nn.Parameter(torch.tensor(1.0))
        self.w_meta = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, image, metadata):
        if image.dim() == 4 and image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)  
        logits_vis = self.model_vis(image)   
        logits_meta = self.model_meta(metadata)   
        combined_logits = self.w_vis * logits_vis + self.w_meta * logits_meta
        output = torch.sigmoid(combined_logits)
        return output


# model_vis = ViT(
#     image_size = 256,      
#     patch_size = 16,      
#     num_classes = 2,      
#     dim = 128,            
#     depth = 6,            
#     heads = 8,            
#     mlp_dim = 256,        
#     pool = 'cls',         
#     channels = 3,         
#     dim_head = 64,        
#     dropout = 0.1,        
#     emb_dropout = 0.1     
# )
# model_meta = MetadataMLP(input_dim=27, hidden_dim=64, output_dim=2, dropout_p=0.1)
# combined_model = CombinedModel(model_vis, model_meta, num_classes=2)


