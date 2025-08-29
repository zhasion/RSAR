# -*- coding: utf-8 -*-
_base_ = 'csl_detr_r50_rsar_fix.py'
model = dict(
    bbox_head=dict(
        angle_coder=dict(
            window='aspect_ratio',
            normalize=True,
        )
    )
)
work_dir = 'work_dirs/arcsl_detr/'
