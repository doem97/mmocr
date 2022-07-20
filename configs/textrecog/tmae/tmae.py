exp_name_usr = "ori_tmae"
exp_notes_usr = "the original tmae backbone"

_base_ = [
    "../../_base_/default_runtime.py",
    "../../_base_/recog_pipelines/vit_pipeline.py",
    "../../_base_/recog_datasets/ST_train.py",
    "../../_base_/recog_datasets/academic_test.py",
]

log_config = dict(
    interval=10,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(project="tmae", name=exp_name_usr, notes=exp_notes_usr),
        ),
    ],
)

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}

label_convertor = dict(type="AttnConvertor", dict_type="DICT90", with_unknown=True)

model = dict(
    type="SATRN",
    backbone=dict(type="ShallowCNN", input_channels=3, hidden_dim=512),
    encoder=dict(
        type="SatrnEncoder",
        n_layers=12,
        n_head=8,
        d_k=512 // 8,
        d_v=512 // 8,
        d_model=512,
        n_position=100,
        d_inner=512 * 4,
        dropout=0.1,
    ),
    decoder=dict(
        type="NRTRDecoder",
        n_layers=6,
        d_embedding=512,
        n_head=8,
        d_model=512,
        d_inner=512 * 4,
        d_k=512 // 8,
        d_v=512 // 8,
    ),
    loss=dict(type="TFLoss"),
    label_convertor=label_convertor,
    max_seq_len=25,
)

# optimizer
optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.05, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy="CosineAnnealing",
    min_lr=0,
    warmup="linear",
    warmup_by_epoch=True,
    warmup_iters=0.5,
)
total_epochs = 6

data = dict(
    samples_per_gpu=256,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type="UniformConcatDataset", datasets=train_list, pipeline=train_pipeline
    ),
    val=dict(type="UniformConcatDataset", datasets=test_list, pipeline=test_pipeline),
    test=dict(type="UniformConcatDataset", datasets=test_list, pipeline=test_pipeline),
)

evaluation = dict(interval=1, metric="acc")
