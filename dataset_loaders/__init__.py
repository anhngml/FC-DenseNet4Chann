import os
from subprocess import check_output, CalledProcessError

from dataset_loaders.images.Vaihingen import VaihingenDataset
from dataset_loaders.images.camvid import CamvidDataset
# from dataset_loaders.images.cifar10 import Cifar10Dataset
# from dataset_loaders.images.cityscapes import CityscapesDataset
# from dataset_loaders.images.isbi_em_stacks import IsbiEmStacksDataset
# from dataset_loaders.images.kitti import KITTIdataset
# from dataset_loaders.images.mscoco import MSCocoDataset
# from dataset_loaders.images.pascalvoc import PascalVOCdataset
# from dataset_loaders.images.polyps912 import Polyps912Dataset
# from dataset_loaders.images.scene_parsing_MIT import SceneParsingMITDataset

from dataset_loaders.videos.change_detection import ChangeDetectionDataset
from dataset_loaders.videos.davis import DavisDataset
from dataset_loaders.videos.gatech import GatechDataset

try:
    cwd = os.path.join(__path__[0], os.path.pardir)
    __version__ = check_output('git rev-parse HEAD', cwd=cwd,
                               shell=True).strip().decode('ascii')
except CalledProcessError:
    __version__ = -1

__all__ = [
    "VaihingenDataset",
    "CamvidDataset",
    "Cifar10Dataset",
    "CityscapesDataset",
    "IsbiEmStacksDataset",
    "KITTIdataset",
    "MSCocoDataset",
    "PascalVOCdataset",
    "Polyps912Dataset",
    "SceneParsingMITDataset",
    "ChangeDetectionDataset",
    "DavisDataset",
    "GatechDataset",
]
