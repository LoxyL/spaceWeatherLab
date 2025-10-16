import torch
from model import ARModel

transformer_config=dict(
    inp_dim = 16,
    dim = 32,
    out_dim = 16,
    num_layers = 4,
    num_heads = 8,
    ff_hidden_dim = 64,
    max_seq_len = 512,
    dropout = 0.0
)


mlp_config=dict(
    in_channels=transformer_config['inp_dim'],
    model_channels=32,
    out_channels=transformer_config['inp_dim'],
    z_channels=transformer_config['out_dim'],
    num_res_blocks=2,
    grad_checkpointing=False
)


diffusion_config=dict(
    num_steps = 64
)

model = ARModel(transformer_config=transformer_config,
                mlp_config=mlp_config,
                diffusion_config=diffusion_config,
                device=torch.device('cpu')
                )