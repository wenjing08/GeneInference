import torch
import torch.nn as nn



class df_relu_2(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, dropout_rate):
        super(df_relu_2, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Dropout(dropout_rate),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Linear(in_size + hidden_size, hidden_size),
            nn.Dropout(dropout_rate),
            nn.ReLU())
        self.output_layer = nn.Sequential(
            nn.Linear(in_size + hidden_size + hidden_size, out_size))
    def forward(self, x):
        layer1_out = self.layer1(x)

        layer2_in = torch.cat((x, layer1_out), 1)
        layer2_out = self.layer2(layer2_in)

        output_layer_in = torch.cat((layer2_in, layer2_out), 1)
        output_layer_out = self.output_layer(output_layer_in)

        return output_layer_out