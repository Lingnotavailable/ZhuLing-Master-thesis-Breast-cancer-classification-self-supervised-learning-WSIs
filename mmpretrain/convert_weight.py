import torch
weights = torch.load('work_dirs/mae_vit-base-p16_8xb512-amp-coslr-20e_in1k_wsi/epoch_28.pth')

new_pretrained_weights = {}
for key in weights['state_dict'].keys():
    new_key = key
    # 假设我们想要将名为'old_param_name'的权重改为'new_param_name'
    if 'backbone.' in new_key:
        new_key = new_key.replace('backbone.', '')
        new_pretrained_weights[new_key] = weights['state_dict'][key]

torch.save(new_pretrained_weights, 'weight/in1k_wsi_mae_pretrained.pth')
