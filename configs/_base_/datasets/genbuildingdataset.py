# dataset settings
dataset_type = 'GenBuildingDataset'
data_root = '../data/LargeMapBuildingSegmentationData'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type='LoadGenImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='RandomFlip', prob=0.5, direction="horizontal"),
    dict(type='RandomFlip', prob=0.5, direction="vertical"),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    # Insert custom transform for combining the images
    dict(type="CombineImageGen"),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadGenImageFromFile'),
    dict(type='Normalize', **img_norm_cfg),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(512, 512), ratio_range=(1.0, 1.0)),
            dict(type='DefaultFormatBundle'),
            dict(type='CombineImageGen'),
            dict(type='Collect', keys=['img'])
            ]
    )
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/images',
        gen_dir='train/images',
        ann_dir='train/masks',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/images',
        gen_dir='val/images',
        ann_dir='val/masks',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test/images',
        gen_dir='test/images',
        ann_dir='test/masks',
        pipeline=test_pipeline))

"""

"""