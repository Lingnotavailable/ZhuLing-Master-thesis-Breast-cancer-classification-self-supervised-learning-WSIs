data_root = 'data/WSI/'
model = dict(
    backbone=dict(
        depth=50,
        norm_cfg=dict(type='SyncBN'),
        type='ResNet',
        zero_init_residual=True),
    head=dict(
        loss=dict(type='CrossEntropyLoss'),
        temperature=0.1,
        type='ContrastiveHead'),
    neck=dict(
        hid_channels=2048,
        in_channels=2048,
        num_layers=2,
        out_channels=128,
        type='NonLinearNeck',
        with_avg_pool=True),
    type='SimCLR')

optim_wrapper = dict(
    optimizer=dict(lr=1e-2, momentum=0.9, type='LARS', weight_decay=1e-06),
    paramwise_cfg=dict(
        custom_keys=dict({
            'bias': dict(decay_mult=0, lars_exclude=True),
            'bn': dict(decay_mult=0, lars_exclude=True),
            'downsample.1': dict(decay_mult=0, lars_exclude=True)
        })),
    type='OptimWrapper')

param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=10,
        start_factor=0.001,
        type='LinearLR'),
    dict(
        T_max=10, begin=1, by_epoch=True, end=20, type='CosineAnnealingLR'),
]

randomness = dict(deterministic=False, seed=None)
resume = None
train_cfg = dict(max_epochs=20, type='EpochBasedTrainLoop')
default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=5, type='CheckpointHook'),
    logger=dict(interval=30, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))


train_dataloader = dict(
    batch_size=64,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root=data_root,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                num_views=2,
                transforms=[
                    [
                        dict(
                            backend='pillow',
                            scale=224,
                            type='RandomResizedCrop'),
                        dict(prob=0.5, type='RandomFlip'),
                        dict(
                            prob=0.8,
                            transforms=[
                                dict(
                                    brightness=0.8,
                                    contrast=0.8,
                                    hue=0.2,
                                    saturation=0.8,
                                    type='ColorJitter'),
                            ],
                            type='RandomApply'),
                        dict(
                            channel_weights=(
                                0.114,
                                0.587,
                                0.2989,
                            ),
                            keep_channels=True,
                            prob=0.2,
                            type='RandomGrayscale'),
                        dict(
                            magnitude_range=(
                                0.1,
                                2.0,
                            ),
                            magnitude_std='inf',
                            prob=0.5,
                            type='GaussianBlur'),
                    ],
                ],
                type='MultiView'),
            dict(type='PackInputs'),
        ],
        split='train',
        type='ImageNet'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        num_views=2,
        transforms=[
            [
                dict(backend='pillow', scale=224, type='RandomResizedCrop'),
                dict(prob=0.5, type='RandomFlip'),
                dict(
                    prob=0.8,
                    transforms=[
                        dict(
                            brightness=0.8,
                            contrast=0.8,
                            hue=0.2,
                            saturation=0.8,
                            type='ColorJitter'),
                    ],
                    type='RandomApply'),
                dict(
                    channel_weights=(
                        0.114,
                        0.587,
                        0.2989,
                    ),
                    keep_channels=True,
                    prob=0.2,
                    type='RandomGrayscale'),
                dict(
                    magnitude_range=(
                        0.1,
                        2.0,
                    ),
                    magnitude_std='inf',
                    prob=0.5,
                    type='GaussianBlur'),
            ],
        ],
        type='MultiView'),
    dict(type='PackInputs'),
]

view_pipeline = [
    dict(backend='pillow', scale=224, type='RandomResizedCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        prob=0.8,
        transforms=[
            dict(
                brightness=0.8,
                contrast=0.8,
                hue=0.2,
                saturation=0.8,
                type='ColorJitter'),
        ],
        type='RandomApply'),
    dict(
        channel_weights=(
            0.114,
            0.587,
            0.2989,
        ),
        keep_channels=True,
        prob=0.2,
        type='RandomGrayscale'),
    dict(
        magnitude_range=(
            0.1,
            2.0,
        ),
        magnitude_std='inf',
        prob=0.5,
        type='GaussianBlur'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])


dataset_type = 'ImageNet'
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True,
    type='SelfSupDataPreprocessor')
