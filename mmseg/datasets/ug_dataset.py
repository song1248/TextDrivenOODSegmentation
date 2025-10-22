import json
import os.path as osp

import mmcv
import numpy as np
import torch

from . import CityscapesDataset
from .builder import DATASETS


def get_rcs_class_probs(data_root, temperature):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()


@DATASETS.register_module()
class UGDataset(object):
    def __init__(self, source, cfg):
        self.source = source
        self.ignore_index = source.ignore_index
        self.CLASSES = source.CLASSES
        self.PALETTE = source.PALETTE

        rcs_cfg = cfg.get('rare_class_sampling')
        self.rcs_enabled = rcs_cfg is not None
        if self.rcs_enabled:
            self.rcs_class_temp = rcs_cfg['class_temp']
            self.rcs_min_crop_ratio = rcs_cfg['min_crop_ratio']
            self.rcs_min_pixels = rcs_cfg['min_pixels']

            self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(
                cfg['source']['data_root'], self.rcs_class_temp)
            
            mmcv.print_log(f'RCS Classes: {self.rcs_classes}', 'mmseg')
            mmcv.print_log(f'RCS ClassProb: {self.rcs_classprob}', 'mmseg')

            with open(
                    osp.join(cfg['source']['data_root'],
                             'samples_with_class.json'), 'r') as of:
                samples_with_class_and_n = json.load(of)
            samples_with_class_and_n = {
                int(k): v
                for k, v in samples_with_class_and_n.items()
                if int(k) in self.rcs_classes
            }
            self.samples_with_class = {}
            for c in self.rcs_classes:
                self.samples_with_class[c] = []
                for file, pixels in samples_with_class_and_n[c]:
                    if pixels > self.rcs_min_pixels:
                        self.samples_with_class[c].append(file.split('/')[-1])
                assert len(self.samples_with_class[c]) > 0
            self.file_to_idx = {}
            for i, dic in enumerate(self.source.img_infos):
                file = dic['ann']['seg_map']
                if isinstance(self.source, CityscapesDataset):
                    file = file.split('/')[-1]
                self.file_to_idx[file] = i

        # #$#$#$#$#$
        # print("self.source:", self.source)
        
        # #$#$#$#$#$
        # print("cfg['source']",cfg['source'])
        # print("Source Dataset:", type(self.source))
        # print("Source Dataset Length:", len(self.source))
        # print("Source img_infos:", len(self.source.img_infos))
        # print("First Source Info:", self.source.img_infos[0] if self.source.img_infos else "Empty")
    
        # #$#$#$#$#$
        # import os
        # data_root = cfg['source']['data_root']
        # img_dir = os.path.join(data_root, cfg['source']['img_dir'])
        # ann_dir = os.path.join(data_root, cfg['source']['ann_dir'])

        # print("img_dir : ", img_dir)
        # print("ann_dir : ", ann_dir)
        # print(f"Image Directory Exists: {os.path.exists(img_dir)}")
        # print(f"Annotation Directory Exists: {os.path.exists(ann_dir)}")
        
        # if os.path.exists(img_dir):
        #     print(f"Number of Images: {len(os.listdir(img_dir))}")
        # if os.path.exists(ann_dir):
        #     print(f"Number of Annotations: {len(os.listdir(ann_dir))}")
        # assert False
        
    
    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        f1 = np.random.choice(self.samples_with_class[c])
        i1 = self.file_to_idx[f1]
        s1 = self.source[i1]
        if self.rcs_min_crop_ratio > 0:
            for j in range(10):
                n_class = torch.sum(s1['gt_semantic_seg'].data == c)
                # mmcv.print_log(f'{j}: {n_class}', 'mmseg')
                if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
                    break
                # Sample a new random crop from source image i1.
                # Please note, that self.source.__getitem__(idx) applies the
                # preprocessing pipeline to the loaded image, which includes
                # RandomCrop, and results in a new crop of the image.
                s1 = self.source[i1]

        return {
            **s1
        }

    def __getitem__(self, idx):
        if self.rcs_enabled:
            return self.get_rare_class_sample()
        else:
            s1 = self.source[idx % len(self.source)]
            return {
                **s1
            }

    def __len__(self):
        return len(self.source)