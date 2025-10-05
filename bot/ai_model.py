import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
ORIGINAL_MAP_WIDTH=23

MAP_CHANNELS=11
class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.alpha * x + self.beta

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, num_patches):
        super().__init__()
        self.norm1 = Affine(dim)
        self.attn = nn.Linear(num_patches, num_patches)
        self.norm2 = Affine(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.mlp(self.norm2(x))
        return x

class ResMLP(nn.Module):
    def __init__(self,embed_dim, depth):
        super().__init__()
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels=MAP_CHANNELS,out_channels=embed_dim,kernel_size=5,padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(embed_dim)
        )
        self.blocks = nn.ModuleList([ResidualMLPBlock(embed_dim, ORIGINAL_MAP_WIDTH*ORIGINAL_MAP_WIDTH) for _ in range(depth)])
        self.norm = Affine(embed_dim)
        self.tile_position=nn.Sequential(
            nn.Conv2d(in_channels=embed_dim,out_channels=1,kernel_size=9,padding="same"),
            nn.ReLU(),
            nn.Flatten()
        )
        self.move_direction=nn.Sequential(
            nn.Conv2d(in_channels=embed_dim,out_channels=4,kernel_size=9,padding="same"),
            nn.ReLU(),
            nn.Softmax(dim=1),
            nn.Flatten()
        )

    def forward(self, x):
        B,C,H,W=x.shape
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x).transpose(1,2).reshape(B,-1,H,W)
        tile_pos=self.tile_position(x)
        move_dir=self.move_direction(x)
        return tile_pos,move_dir
    

class Model:
    def __init__(self,device_name="cpu",ckpt_dir="./ai-bot/models/epoch5.pt"):
        self.device = torch.device(device_name)
        self.model=ResMLP(96,5)
        self.model.to(self.device)
        checkpoint = torch.load(ckpt_dir,map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    def infer(self,game_state):
        input_x=np.copy(game_state)
        input_x=input_x.transpose(2,0,1)
        input_x=torch.tensor(input_x).reshape(1,MAP_CHANNELS,ORIGINAL_MAP_WIDTH,ORIGINAL_MAP_WIDTH)
        with torch.no_grad():
            input_x=input_x.to(self.device)
            tile_pos,move_dir=self.model(input_x)
            tile_pos=F.softmax(tile_pos,dim=1).reshape(1,ORIGINAL_MAP_WIDTH,ORIGINAL_MAP_WIDTH).cpu()
            move_dir=move_dir.reshape(1,4,ORIGINAL_MAP_WIDTH,ORIGINAL_MAP_WIDTH).cpu()
            return tile_pos.numpy(),move_dir.numpy()