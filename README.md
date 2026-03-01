# Single Slice-to-3D Reconstruction in Medical Imaging and Natural Objects: A Comparative Benchmark with SAM 3D

A benchmark of five image-to-3D foundation models on medical and natural datasets showed that all methods suffer from low volumetric overlap due to severe depth ambiguity in single-slice inputs, though SAM3D best preserved topological similarity to ground-truth medical shapes. These findings demonstrate that reliable medical 3D reconstruction cannot rely on zero-shot transfer from natural-image priors and instead requires domain-specific adaptation and anatomical constraints. This project is built upon [sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects).

## Installation

Follow the [setup](doc/setup.md) steps before running the following.

## Datasets

You can download the datasets from their official websites, i.e., [AeroPath](https://github.com/raidionics/AeroPath), [BTCV](https://www.kaggle.com/datasets/ipythonx/abdomen), [Duke C-Spine](https://data.midrc.org/discovery/H6K0-A61V), [Medical Segmentation Decathlon](http://medicaldecathlon.com/dataaws/), [Google Scanned Objects](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/), and [Animal3D](https://xujiacong.github.io/Animal3D/).

## Preprocessing

For simplicity, we preprocess the data by isolating each object as an individual sample, e.g., 

```
cd notebook
python decompose_nii_to_obj_AeroPath.py
```

## Benchmark 3D Reconstruction

Run the following commands

```
cd notebook
bash infer_and_eval.sh
```


## Citing Our Work

If you find our work helpful for your research, please cite it using the following BibTeX entry.

```
@article{luo2026single,
  title={Single-Slice-to-3D Reconstruction in Medical Imaging and Natural Objects: A Comparative Benchmark with SAM 3D},
  author={Luo, Yan and Ravishankar, Advaith and Liu, Serena and Yang, Yutong and Wang, Mengyu},
  journal={arXiv preprint arXiv:2602.09407},
  year={2026}
}
```
