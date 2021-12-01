#tools/dist_train.sh cascade_mask_rcnn_cbv2_swin_small_patch4_3x_rats.py 6

_base_ = [
    'configs/cbnet/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.py'
]
num_classes=2
dataset_type = 'CocoDatasetRat'
model = dict(
    backbone=dict(
        type='CBSwinTransformer',
    ),
    neck=dict(
        type='CBFPN',
    ),
    roi_head=dict(
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ],
        mask_head=dict(num_classes=num_classes)
    ),
    test_cfg = dict(
        rcnn=dict(
            score_thr=0.001,
            nms=dict(type='soft_nms'),
        )
    )
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from HTC
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Resize',
        img_scale=[(1600, 400), (1600, 1400)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 1400),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
samples_per_gpu=1
data = dict(samples_per_gpu=samples_per_gpu,
            train=dict(type = 'ConcatDataset', 
                datasets = [dict(pipeline=train_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/coco_BWsiderat_train.json',
                        img_prefix='data/rats/bwsiderat/'),
                    dict(pipeline=train_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/coco_bwsiderat800x600_train.json',
                        img_prefix='data/rats/bwsiderat800x600/'), 
                    dict(pipeline=train_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/bw_rat_800x600_1130_train.json',
                        img_prefix='data/rats/bw_rat_800x600_1130/'),
                    dict(pipeline=train_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/bw_rat_800x600_1130_train.json',
                        img_prefix='data/rats/bw_rat_800x600_1130/'),
                    dict(pipeline=train_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/bw_rat_800x600_1130_train.json',
                        img_prefix='data/rats/bw_rat_800x600_1130/')]), 
            val=dict(type = 'ConcatDataset', 
                datasets = [dict(pipeline=test_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/coco_BWsiderat_val.json',
                        img_prefix='data/rats/bwsiderat/'),
                    dict(pipeline=test_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/coco_bwsiderat800x600_val.json',
                        img_prefix='data/rats/bwsiderat800x600/'),
                    dict(pipeline=test_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/bw_rat_800x600_1130_val.json',
                        img_prefix='data/rats/bw_rat_800x600_1130/')]), 
            test=dict(type = 'ConcatDataset', 
                datasets = [dict(pipeline=test_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/coco_BWsiderat_val.json',
                        img_prefix='data/rats/bwsiderat/'),
                    dict(pipeline=test_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/coco_bwsiderat800x600_val.json',
                        img_prefix='data/rats/bwsiderat800x600/'),
                    dict(pipeline=test_pipeline,
                        type=dataset_type,
                        ann_file='data/rats/bw_rat_800x600_1130_val.json',
                        img_prefix='data/rats/bw_rat_800x600_1130/')]))
optimizer = dict(lr=0.0001*(samples_per_gpu/2))
load_from = 'work_dirs/cascade_mask_rcnn_cbv2_swin_small_patch4_3x_rats/latest.pth'
# load_from = None
checkpoint_config = dict(interval=20)
runner = dict(type='EpochBasedRunnerAmp', max_epochs=400)
