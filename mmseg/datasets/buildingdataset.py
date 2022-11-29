# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class BuildingDataset(CustomDataset):
    """Building Dataset"""
    CLASSES = ('background', 'building')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(BuildingDataset, self).__init__(
            img_suffix='.tif',
            seg_map_suffix='.tif',
            reduce_zero_label=False,
            **kwargs)

@DATASETS.register_module()
class GenBuildingDataset(CustomDataset):
    """Building Dataset"""
    CLASSES = ('background', 'building')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, **kwargs):
        super(GenBuildingDataset, self).__init__(
            img_suffix='.tif',
            gen_suffix='.png',
            seg_map_suffix='.tif',
            reduce_zero_label=False,
            **kwargs)
