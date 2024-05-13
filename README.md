# ZhuLing-Master-thesis-Breast-cancer-classification-self-supervised-learning-WSIs
# Introduction

该项目使用MMpretrain工具箱进行模型预训练，利用其中的Simclar (ICML'2020) 和 MAE(CVPR'2022)的自监督学习策略。所有预训练/Finetune/下游任务的**参数脚本**位于[config_medical](ling/zuling-wsi/config_medical)

1. *Pretrain*：使用2种不同的预训练方式进行比较:**simclr_resnet50_xb64-coslr-20e_wsi**和**simclr_resnet50_xb128-coslr-200e_wsi**

2. *Finetuning*：初始权重分别来自'_used imagenet without supervised learning_', '_used imagenet with 1k classes_','_kaiming random weights initialize_','wsi from start'

预训练/Finetune/下游4分类任务的**训练脚本**位于[train.py](ling/zuling-wsi/tools/train.py)，具体使用教程位于下面的How to use，主干模型为'_ResNet50_'和'_Vit_'

可视化的结果图和表格位于[visualization](ling/zuling-wsi/visualization)



## MMpretrain

MMPreTrain 是一款基于 PyTorch 的开源深度学习预训练工具箱，是 [OpenMMLab](https://openmmlab.com/) 项目的成员之一

### Main features

- 支持多样的主干网络与预训练模型
- 支持多种训练策略（有监督学习，无监督学习，多模态学习等）
- 提供多种训练技巧
- 大量的训练配置文件
- 高效率和高可扩展性
- 功能强大的工具箱，有助于模型分析和实验


## Installation

以下是安装的简要步骤：

```shell
conda create -n open-mmlab python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate open-mmlab
pip3 install openmim
git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain
mim install -e .
```
## How to use
- [学习配置文件](https://mmpretrain.readthedocs.io/zh_CN/latest/user_guides/config.html)
- [准备数据集](https://mmpretrain.readthedocs.io/zh_CN/latest/user_guides/dataset_prepare.html)
- [使用现有模型推理](https://mmpretrain.readthedocs.io/zh_CN/latest/user_guides/inference.html)
- [训练](https://mmpretrain.readthedocs.io/zh_CN/latest/user_guides/train.html)
- [测试](https://mmpretrain.readthedocs.io/zh_CN/latest/user_guides/test.html)
- [下游任务](https://mmpretrain.readthedocs.io/zh_CN/latest/user_guides/downstream.html)

## Models

<div align="center">
  <b>概览</b>
</div>
<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
        <b>支持的主干网络</b>
      </td>
      <td>
        <b>自监督学习</b>
      </td>
    </tr>
    <tr valign="top">
      <td>
        <ul>
        <li><a href="configs/vgg">VGG</a></li>
        <li><a href="configs/resnet">ResNet</a></li>
        <li><a href="configs/resnext">ResNeXt</a></li>
        <li><a href="configs/seresnet">SE-ResNet</a></li>
        <li><a href="configs/seresnet">SE-ResNeXt</a></li>
        <li><a href="configs/regnet">RegNet</a></li>
        <li><a href="configs/shufflenet_v1">ShuffleNet V1</a></li>
        <li><a href="configs/shufflenet_v2">ShuffleNet V2</a></li>
        <li><a href="configs/mobilenet_v2">MobileNet V2</a></li>
        <li><a href="configs/mobilenet_v3">MobileNet V3</a></li>
        <li><a href="configs/swin_transformer">Swin-Transformer</a></li>
        <li><a href="configs/swin_transformer_v2">Swin-Transformer V2</a></li>
        <li><a href="configs/repvgg">RepVGG</a></li>
        <li><a href="configs/vision_transformer">Vision-Transformer</a></li>
        <li><a href="configs/tnt">Transformer-in-Transformer</a></li>
        <li><a href="configs/res2net">Res2Net</a></li>
        <li><a href="configs/mlp_mixer">MLP-Mixer</a></li>
        <li><a href="configs/deit">DeiT</a></li>
        <li><a href="configs/deit3">DeiT-3</a></li>
        <li><a href="configs/conformer">Conformer</a></li>
        <li><a href="configs/t2t_vit">T2T-ViT</a></li>
        <li><a href="configs/twins">Twins</a></li>
        <li><a href="configs/efficientnet">EfficientNet</a></li>
        <li><a href="configs/edgenext">EdgeNeXt</a></li>
        <li><a href="configs/convnext">ConvNeXt</a></li>
        <li><a href="configs/hrnet">HRNet</a></li>
        <li><a href="configs/van">VAN</a></li>
        <li><a href="configs/convmixer">ConvMixer</a></li>
        <li><a href="configs/cspnet">CSPNet</a></li>
        <li><a href="configs/poolformer">PoolFormer</a></li>
        <li><a href="configs/inception_v3">Inception V3</a></li>
        <li><a href="configs/mobileone">MobileOne</a></li>
        <li><a href="configs/efficientformer">EfficientFormer</a></li>
        <li><a href="configs/mvit">MViT</a></li>
        <li><a href="configs/hornet">HorNet</a></li>
        <li><a href="configs/mobilevit">MobileViT</a></li>
        <li><a href="configs/davit">DaViT</a></li>
        <li><a href="configs/replknet">RepLKNet</a></li>
        <li><a href="configs/beit">BEiT</a></li>
        <li><a href="configs/mixmim">MixMIM</a></li>
        <li><a href="configs/revvit">RevViT</a></li>
        <li><a href="configs/convnext_v2">ConvNeXt V2</a></li>
        <li><a href="configs/vig">ViG</a></li>
        <li><a href="configs/xcit">XCiT</a></li>
        <li><a href="configs/levit">LeViT</a></li>
        <li><a href="configs/riformer">RIFormer</a></li>
        <li><a href="configs/glip">GLIP</a></li>
        <li><a href="configs/sam">ViT SAM</a></li>
        <li><a href="configs/eva02">EVA02</a></li>
        <li><a href="configs/dinov2">DINO V2</a></li>
        <li><a href="configs/hivit">HiViT</a></li>
        </ul>
      </td>
      <td>
        <ul>
        <li><a href="configs/mocov2">MoCo V1 (CVPR'2020)</a></li>
        <li><a href="configs/simclr">SimCLR (ICML'2020)</a></li>
        <li><a href="configs/mocov2">MoCo V2 (arXiv'2020)</a></li>
        <li><a href="configs/byol">BYOL (NeurIPS'2020)</a></li>
        <li><a href="configs/swav">SwAV (NeurIPS'2020)</a></li>
        <li><a href="configs/densecl">DenseCL (CVPR'2021)</a></li>
        <li><a href="configs/simsiam">SimSiam (CVPR'2021)</a></li>
        <li><a href="configs/barlowtwins">Barlow Twins (ICML'2021)</a></li>
        <li><a href="configs/mocov3">MoCo V3 (ICCV'2021)</a></li>
        <li><a href="configs/beit">BEiT (ICLR'2022)</a></li>
        <li><a href="configs/mae">MAE (CVPR'2022)</a></li>
        <li><a href="configs/simmim">SimMIM (CVPR'2022)</a></li>
        <li><a href="configs/maskfeat">MaskFeat (CVPR'2022)</a></li>
        <li><a href="configs/cae">CAE (arXiv'2022)</a></li>
        <li><a href="configs/milan">MILAN (arXiv'2022)</a></li>
        <li><a href="configs/beitv2">BEiT V2 (arXiv'2022)</a></li>
        <li><a href="configs/eva">EVA (CVPR'2023)</a></li>
        <li><a href="configs/mixmim">MixMIM (arXiv'2022)</a></li>
        <li><a href="configs/itpn">iTPN (CVPR'2023)</a></li>
        <li><a href="configs/spark">SparK (ICLR'2023)</a></li>
        <li><a href="configs/mff">MFF (ICCV'2023)</a></li>
        </ul>
      </td>
      <td>
  </tbody>
</table>

