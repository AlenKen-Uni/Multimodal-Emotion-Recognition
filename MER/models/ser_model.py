import torch
import torch.nn as nn

class SpatialAttention(nn.Module):
    def __init__(self, time_step=128, hidden_dim=512):
        super(SpatialAttention, self).__init__()
        self.pool = nn.AdaptiveMaxPool1d(time_step)
        self.attn_mlp = nn.Sequential(
            nn.Linear(time_step, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1) # (N, C, L)
        x = self.pool(x) # (N, C, 256)
        w = self.attn_mlp(x) # (N, C, 1)
        w = nn.functional.softmax(w, dim=1).permute(0, 2, 1) # (N, 1, C)

        return w


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim=256):
        super(TemporalAttention, self).__init__()
        self.attn_mlp = nn.Sequential(
            nn.Linear(1064, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        w = self.attn_mlp(x)  # (N, L, 1)
        w = nn.functional.softmax(w, dim=1)
        
        return w
        

class SizeReducer(nn.Module):
    def __init__(self, feature_dim):
        super(SizeReducer, self).__init__()
        self.conv1 = nn.Conv1d(feature_dim, feature_dim, 5, 3)
        self.conv2 = nn.Conv1d(feature_dim, feature_dim, 5, 3)
        self.norm1 = nn.InstanceNorm1d(feature_dim, affine=True)
        self.norm2 = nn.InstanceNorm1d(feature_dim, affine=True)
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # x: (N, L, C)
        x = self.conv1(x.permute(0, 2, 1))
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.pool(x)

        return x

class Classifier(nn.Module):
    def __init__(self, feature_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 256)
        self.fc2 = nn.Linear(256, output_dim)
        self.norm = nn.LayerNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # x: (N, C)
        x = self.fc1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x) # (N, output_dim)

        return x


class SER(nn.Module):
    def __init__(self, wav2vec2_model, output_dim=7):
        super(SER, self).__init__()
        self.output_dim = output_dim
        self.wav2vec2 = wav2vec2_model
        self.feature_dim = 1024+40

        self.gru = nn.GRU(
            input_size=1024,
            hidden_size=1024,
            num_layers=1, 
            batch_first=True, 
            bidirectional=False
        )
        self.spatial_attention = SpatialAttention()
        self.temporal_attention = TemporalAttention()
        self.size_reduction = SizeReducer(self.feature_dim)
        self.classifier1 = Classifier(self.feature_dim - 40, self.output_dim)
        self.classifier2 = Classifier(self.feature_dim, self.output_dim)
    
    def forward(self, wave, hfeat):
        # extract deep feature
        attn_wave_mask = (torch.abs(wave) > 1e-8).int()
        x = self.wav2vec2(
            input_values=wave, 
            attention_mask=attn_wave_mask,
        ).last_hidden_state # (N, L, 1024)

        # branch 1
        _, hn = self.gru(x)
        x1 = hn.squeeze(0) # (N, 1024)
        out1 = self.classifier1(x1)
        
        # branch 2
        x_det = x.detach() # detach x so no grads flow back into branch 1
        x2 = torch.cat((x_det, hfeat), dim=2) # (N, L, 1068)
        x2 = x2 * self.temporal_attention(x2)
        x2 = x2 * self.spatial_attention(x2)
        x2 = self.size_reduction(x2)
        x2 = torch.flatten(x2, 1) # (N, 1068)
        out2 = self.classifier2(x2)
        
        return out1, out2