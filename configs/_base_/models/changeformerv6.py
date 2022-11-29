# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    #pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='EncoderTransformerV3',
        img_size=512,
        patch_size=7,
        in_chans=3,
        num_classes=2,
        embed_dims=[64, 128, 320, 512],
        num_heads=[2, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.1,
        depths=[3, 3, 4, 3],
        sr_ratios=[8, 4, 2, 1]),
    decode_head=dict(
        type='DecoderTransformerV3',
        in_index=[0, 1, 2, 3],
        in_channel_list=[32, 64, 128, 256],
        channels=512,
        embedding_dim=64,
        output_nc=2,
        decoder_softmax=False,
        feature_strides=[2, 4, 8, 16],
        align_corners=True,
        in_channels=3,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
