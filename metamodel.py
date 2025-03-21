import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class MetadataMLP(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=2, dropout_p=0.1):

        super(MetadataMLP, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            Swish(),
            nn.Dropout(p=dropout_p),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            Swish(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            Swish()
        )
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: [batch_size, input_dim]
        """
        x = self.feature_extractor(x)
        out = self.output_layer(x)
        
        return out

if __name__ == "__main__":
    batch_size = 4
    input_dim = 32
    model = MetadataMLP(input_dim=input_dim, hidden_dim=64, output_dim=2, dropout_p=0.1)
    sample_input = torch.randn(batch_size, input_dim)
    output = model(sample_input)
    print("Test Metadata output:", output.shape)  
