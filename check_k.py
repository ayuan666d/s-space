import torch
p = torch.load('data/coord_nav_params_K100.pt', map_location='cpu', weights_only=False)
L0 = list(p['principal_dirs'].keys())[0]
print(f'principal_dirs[L0] shape: {p["principal_dirs"][L0].shape}')
print(f'K key in params: {p.get("K", "NOT FOUND")}')
print(f'metric_weights[L0] shape: {p["metric_weights"][L0].shape}')
print(f'inject_layers: {p.get("inject_layers", "NOT FOUND")}')
