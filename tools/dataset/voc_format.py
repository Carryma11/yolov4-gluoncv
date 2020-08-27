"""Pascal VOC object detection dataset."""

from gluoncv import data as gdata


class VOCLike(gdata.VOCDetection):
    CLASSES = ['person', 'dog']
    def __init__(self, root, splits, transform=None, index_map=None, preload_label=True):
        super(VOCLike, self).__init__(root, splits, transform, index_map, preload_label)
