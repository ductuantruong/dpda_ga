import torch.nn as nn
import fairseq
from dataclasses import dataclass, field
import torch.nn.functional as F
from nets.mamba_blocks import MixerModel

@dataclass
class MambaConfig:
    d_model: int = 64
    n_layer: int = 6
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    
class SSLModel(nn.Module): #W2V
    def __init__(self,device):
        super(SSLModel, self).__init__()
        cp_path = 'pretrained_models/xlsr2_300m.pt'   # Change the pre-trained XLSR model path. 
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device=device
        self.out_dim = 1024

    def extract_feat(self, input_data):
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()      

        # input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
                
        # [batch, length, dim]
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        layerresult = self.model(input_tmp, mask=False, features_only=True)['layer_results']
        return emb, layerresult


class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device=device
        ####
        # create network wav2vec 2.0
        ####
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(1024, args.emb_size)
        print('W2V + mamba')
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.config = MambaConfig(d_model=args.emb_size, n_layer=args.num_mamba_encoders // 2)
        print(self.config)
        self.conformer=MixerModel(d_model=self.config.d_model, 
                                  n_layer=self.config.n_layer, 
                                  ssm_cfg=self.config.ssm_cfg, 
                                  rms_norm=self.config.rms_norm, 
                                  residual_in_fp32=self.config.residual_in_fp32, 
                                  fused_add_norm=self.config.fused_add_norm
                                  )
    def forward(self, x):
        #-------pre-trained Wav2vec model fine tunning ------------------------##
        x_ssl_feat,_ = self.ssl_model.extract_feat(x.squeeze(-1))
        x=self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim) (bs, 208, 256)
        x = x.unsqueeze(dim=1) # add channel #(bs, 1, frame_number, 256)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)
        out =self.conformer(x) 
        return out


