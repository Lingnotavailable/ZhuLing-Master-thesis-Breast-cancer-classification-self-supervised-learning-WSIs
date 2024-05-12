auto_scale_lr = dict(base_batch_size=64)
bgr_mean = [
    103.53,
    116.28,
    123.675,
]
bgr_std = [
    57.375,
    57.12,
    58.395,
]
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=4,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
data_root = 'data/Dataset-Finetuning-new/Dataset100'
dataset_type = 'ImageNet'
default_hooks = dict(
    checkpoint=dict(interval=2, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
init_cfg = dict(
    checkpoint='weight/in1k_wsi_mae_pretrained.pth', type='Pretrained')
launcher = 'none'
load_from = 'work_dirs/vit-base-p16_64xb64_in1k_wsi_Dataset100/epoch_160.pth'
log_level = 'INFO'
model = dict(
    backbone=dict(
        arch='b',
        drop_rate=0.1,
        img_size=224,
        init_cfg=dict(
            checkpoint='weight/in1k_wsi_mae_pretrained.pth',
            type='Pretrained'),
        patch_size=16,
        type='VisionTransformer'),
    head=dict(
        hidden_dim=3072,
        in_channels=768,
        loss=dict(
            label_smooth_val=0.1, mode='classy_vision',
            type='LabelSmoothLoss'),
        num_classes=4,
        type='VisionTransformerClsHead'),
    neck=None,
    train_cfg=dict(augments=dict(alpha=0.2, type='Mixup')),
    type='ImageClassifier')
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0),
    optimizer=dict(lr=0.003, type='AdamW', weight_decay=0.3),
    paramwise_cfg=dict(
        custom_keys=dict({
            '.cls_token': dict(decay_mult=0.0),
            '.pos_embed': dict(decay_mult=0.0)
        })))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=30,
        start_factor=0.0001,
        type='LinearLR'),
    dict(
        T_max=170, begin=30, by_epoch=True, end=300, type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=None)
resume = 'work_dirs/vit-base-p16_64xb64_in1k_wsi_Dataset100/epoch_160.pth'
test_cfg = dict()
test_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='data/Dataset-Finetuning-new/Dataset100',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                edge='short',
                interpolation='bicubic',
                scale=256,
                type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        split='val',
        type='ImageNet'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(average=None, topk=1, type='MultiLabelMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        backend='pillow',
        edge='short',
        interpolation='bicubic',
        scale=256,
        type='ResizeEdge'),
    dict(crop_size=224, type='CenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=2)
train_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='data/Dataset-Finetuning-new/Dataset100',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                interpolation='bicubic',
                scale=224,
                type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(
                hparams=dict(
                    interpolation='bicubic', pad_val=[
                        104,
                        116,
                        124,
                    ]),
                policies='imagenet',
                type='AutoAugment'),
            dict(type='PackInputs'),
        ],
        split='train',
        type='ImageNet'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        backend='pillow',
        interpolation='bicubic',
        scale=224,
        type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(
        hparams=dict(interpolation='bicubic', pad_val=[
            104,
            116,
            124,
        ]),
        policies='imagenet',
        type='AutoAugment'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='data/Dataset-Finetuning-new/Dataset100',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                backend='pillow',
                edge='short',
                interpolation='bicubic',
                scale=256,
                type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        split='val',
        type='ImageNet'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(average=None, topk=1, type='MultiLabelMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/vit-base-p16_64xb64_in1k_wsi_Dataset100'