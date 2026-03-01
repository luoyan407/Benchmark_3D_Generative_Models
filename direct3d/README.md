
# Direct3D: Scalable Image-to-3D Generation via 3D Latent Diffusion Transformer (NeurIPS 2024)

<div align="center">
  <a href=https://nju-3dv.github.io/projects/Direct3D/ target="_blank"><img src=https://img.shields.io/badge/Project%20Page-333399.svg?logo=googlehome height=22px></a>
  <a href= target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Demo-276cb4.svg height=22px></a>
  <a href=https://huggingface.co/DreamTechAI/Direct3D/tree/main target="_blank"><img src=https://img.shields.io/badge/%F0%9F%A4%97%20Models-d96902.svg height=22px></a>
  <a href=https://openreview.net/pdf?id=vCOgjBIZuL target="_blank"><img src=https://img.shields.io/badge/Paper-b5212f.svg?logo=paperswithcode height=22px></a>
  <a href=https://arxiv.org/abs/2405.14832 target="_blank"><img src=https://img.shields.io/badge/Arxiv-b5212f.svg?logo=arxiv height=22px></a>
</div>

<p align="center">
  <img src="assets/demo/video2.gif", width="48%">
  <img src="assets/demo/video1.gif", width="48%">
  <br>
</p>

---

## ✨ News

- Feb 11, 2025: 🔨 We are working on the Gradio demo and will release it soon!
- Feb 11, 2025: 🎁 Enjoy our improved version of Direct3D with high quality geometry and texture at [https://www.neural4d.com](https://www.neural4d.com/).
- Feb 11, 2025: 🚀 Release inference code of Direct3D and the pretrained models are available at 🤗 [Hugging Face](https://huggingface.co/DreamTechAI/Direct3D/tree/main).

## 📝 Abstract

We introduce **Direct3D**, a native 3D generative model scalable to in-the-wild input images, without requiring a multiview diffusion model or SDS optimization. Our approach comprises two primary components: a Direct 3D Variational Auto-Encoder **(D3D-VAE)** and a Direct 3D Diffusion Transformer **(D3D-DiT)**. D3D-VAE efficiently encodes high-resolution 3D shapes into a compact and continuous latent triplane space. Notably, our method directly supervises the decoded geometry using a semi-continuous surface sampling strategy, diverging from previous methods relying on rendered images as supervision signals. D3D-DiT models the distribution of encoded 3D latents and is specifically designed to fuse positional information from the three feature maps of the triplane latent, enabling a native 3D generative model scalable to large-scale 3D datasets. Additionally, we introduce an innovative image-to-3D generation pipeline incorporating semantic and pixel-level image conditions, allowing the model to produce 3D shapes consistent with the provided conditional image input. Extensive experiments demonstrate the superiority of our large-scale pre-trained Direct3D over previous image-to-3D approaches, achieving significantly better generation quality and generalization ability, thus establishing a new state-of-the-art for 3D content creation.

<p align="center">
  <img src="assets/figure/teaser.gif", width="99%">
  <br>
</p>

## 🚀 Getting Started

### Installation

```sh
git clone https://github.com/DreamTechAI/Direct3D.git

cd Direct3D

pip install -r requirements.txt

pip install -e .
```

### Usage

```python
from direct3d.pipeline import Direct3dPipeline
pipeline = Direct3dPipeline.from_pretrained("DreamTechAI/Direct3D")
pipeline.to("cuda")
mesh = pipeline(
    "assets/devil.png",
    remove_background=False, # set to True if the background of the image needs to be removed
    mc_threshold=-1.0,
    guidance_scale=4.0,
    num_inference_steps=50,
)["meshes"][0]
mesh.export("output.obj")
```

## 🤗 Acknowledgements

Thanks to the following repos for their great work, which helps us a lot in the development of Direct3D:

- [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet/tree/master)
- [Michelangelo](https://github.com/NeuralCarver/Michelangelo)
- [Objaverse](https://objaverse.allenai.org/)
- [diffusers](https://github.com/huggingface/diffusers)

## 📖 Citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{direct3d,
  title={Direct3D: Scalable Image-to-3D Generation via 3D Latent Diffusion Transformer},
  author={Wu, Shuang and Lin, Youtian and Zhang, Feihu and Zeng, Yifei and Xu, Jingxi and Torr, Philip and Cao, Xun and Yao, Yao},
  journal={arXiv preprint arXiv:2405.14832},
  year={2024}
}
```

---
