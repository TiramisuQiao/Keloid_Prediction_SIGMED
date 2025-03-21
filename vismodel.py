import torch
from rich import print
from vit import ViT
from torchsummary import summary
model = ViT(
    image_size = 64,      
    patch_size = 16,      
    num_classes = 2,      
    dim = 128,            
    depth = 6,            
    heads = 8,            
    mlp_dim = 256,        
    pool = 'cls',         
    channels = 3,         
    dim_head = 64,        
    dropout = 0.1,        
    emb_dropout = 0.1     
)

model.to('cuda')
summary(model, input_size=(3, 64, 64))
# output = model(sample_input)
# print("Output:", output.shape)  

