# mmdetection 学习笔记

## 一、mmcv

### 1.1介绍 

​    mmcv是MMLABS实验室用来实现计算机视觉相关项目的基础python库，如mmdetection

### 1.2 __API__ 接口

> mmcv.list_from_file()

for example `a.txt` is a text file with 5 lines

```python 
a
b
c
d
```

then use ` list_from_file` to load the list from `a.txt`

```python
>>> mmcv.list_from_file('a.txt')
['a', 'b', 'c', 'd', 'e']
>>> mmcv.list_from_file('a.txt', offset=2)
['c', 'd', 'e']
>>> mmcv.list_from_file('a.txt', max_num=2)
['a', 'b']
>>> mmcv.list_from_file('a.txt', prefix='/mnt/')
['/mnt/a', '/mnt/b', '/mnt/c', '/mnt/d', '/mnt/e']
```

##　二. mmdet 

### 2.1 介绍

mmdet是mmdetection中用来存放搭建网络和dataset代码的文件，下面先来看看其主要结构:

>mmdet
>├── apis
>├── core
>├── datasets
>├──__ __init____.py
>├── models
>├── ops
>└── version.py

### 2.2 datasets

其中 `datasets` 中存放了coco 与 voc 数据集的加载文件（目前还没完全看懂），

### 2.3 models

`models` 中存放了搭建模型需要的 5 个部分

* backbone
* neck
* rpn_head
* bbox_head
* mask_head

我们先来看看其文件结构:

> mmdet/models
> ├── anchor_heads
> ├── backbones
> ├── bbox_heads
> ├── builder.py
> ├── detectors
> ├── __init__.py
> ├── mask_heads
> ├── necks
> ├── registry.py
> ├── roi_extractors
> ├── shared_heads
> └── utils

#### 2.3.1 __Registry__

除了模型搭建的５个部分外，其中 `registry.py` 是我们需要重点了解的，它实现了一个注册器的功能，主要的作用是将 `mmdet/models` 中搭建的模型保存在 Registry 类的 model_dict 中，就像将模型注册在案一样，其代码如下：

```python
import torch.nn as nn


class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def _register_module(self, module_class):
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not issubclass(module_class, nn.Module):
            raise TypeError('module must be a child of nn.Module, but got {}'.format(module_class))
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls):
        self._register_module(cls)
        return cls


BACKBONES = Registry('backbone')
NECKS = Registry('neck')
ROI_EXTRACTORS = Registry('roi_extractor')
SHARED_HEADS = Registry('shared_head')
HEADS = Registry('head')
DETECTORS = Registry('detector')
```

可以看到在 `Registry` 类中，定义两个属性用于表示所注册的模型，`name` 表示注册模型的类别，`model_dict` 是一个字典，用于存放模型的组件。在`registry.py` 文件中已经实例化了６个注册器，`BACKBONES`, `NECKS` ,`ROI_EXTRACTORS`, `SHARED_HEADS`  `HEADS`, `DETECTORS` 其中保存了搭建一个完整的检测网络所需要的全部组件。

#### 2.3.2 __builder__ 

当我们需要实例化一个网络时，通过 `mmdet/builder.py` 文件中的`builde_detector` 函数来将上述６个注册器中所储存的网络组件按照所搭建网络的配置文件拼接成一个完整的检测网络，以ＳＳＤ为例，`builde_detector` 接收创建ＳＳＤ网络的配置文件，并交由`build` 函数去搭建网络，下面直接看看代码：

首先是`builde_detector` 的调用：

```python
model = build_detector(
    cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
```

其中`cfg.model` 为ＳＳＤ网络的配置文件，`train_cfg` , `test_cfg` 分别为训练与测试阶段所需要的配置文件。

```python 
def build(cfg, registry, default_args=None):
    '''build()主要通过_build_module()从registry.module_dict中实例化注册过的模型'''
    if isinstance(cfg, list):
        modules = [_build_module(cfg_, registry, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return _build_module(cfg, registry, default_args)
```

我们可以看到 `build` 函数在接收了`builde_detector` 传进来的`cfg` , `registry` ,`default_args` 后又交由`_build_module` 来从`registry` 中实例化模型，我们再看看`_build_module` 的代码：

```python 
def _build_module(cfg, registry, default_args):
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None
    args = cfg.copy()
    obj_type = args.pop('type')
    if mmcv.is_str(obj_type):
        if obj_type not in registry.module_dict:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
        obj_type = registry.module_dict[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_type(**args) # 接收cfg.model中的参数，实例化DETECTOR
```

在这段代码中，先利用 `args` 继承`cfg` 中定义好的参数，经过`args.pop('type')` 后只保留搭建检测器所需要的必要参数  , `obj_type` 接收网络的类型，由于ＳＳＤ是单阶段检测器，在其配置文件中类型为`SingleStageDetector` , 然后在已注册的`DETECTOR` 中去寻找是否存在`SingleStageDetector` ， 则由`obj_type` 这个变量接收`SingleStageDetector` 这个网络框架，最后由`obj_type` 接收`args` 从而实例化`SingleStageDetector`。

#### 2.3.3  detector 

接下来我们以单阶段检测器（SingleStageDetector）为例，来看看网络是如何搭建的，`SingleStageDetector` 类定义在`models/detector/single_stage.py` 中，我们首先来看看`SingleStageDetector` 类的`forward` :

```python
def forward_train(self,
                  img,
                  img_metas,
                  gt_bboxes,
                  gt_labels,
                  gt_bboxes_ignore=None):
    x = self.extract_feat(img)
    outs = self.bbox_head(x)
    loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)
    losses = self.bbox_head.loss(
        *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
    return losses
```

在`forward_train` 中，`self.extract_feat()` 封装了`backbone` 和`neck` (_SSD是直接通过backbone得到feature_maps, 没有FPN网络所以没有`neck`层_) , 用于提取`feature_maps` ,  SSD的backbone网络在`models/backbones/ssd_vgg.py` 中定义，我们直接看其前向过程：

```python 
def forward(self, x):
    outs = []
    for i, layer in enumerate(self.features):
        x = layer(x)
        if i in self.out_feature_indices:
            outs.append(x)
    for i, layer in enumerate(self.extra):
        x = F.relu(layer(x), inplace=True)
        if i % 2 == 1:
            outs.append(x)
    outs[0] = self.l2_norm(outs[0])
    if len(outs) == 1:
        return outs[0]
    else:
        return tuple(outs)
```

在`forward`中 outs 取的是多stage的输出，先拼成一个list再转成tuple，去哪些stage是根据SSD的config中的`outs_indices` 和`outs_feature_indices` 决定的，二者表达的意思是一样的只是表示方式不同，由于在SSD中`extra` 层的每一层卷积的输出我们都需要所以`outs_indices` 中只定义了原 `vgg` 的`feature_maps` 位置。

下一步将抽取的特征送入`bbox_head` 中，在SSD的config中`bbox_head` 为`SSDHead` 这主要涉及两个文件`models/anchor_heads/anchor_head.py` 和 `models/anchor_heads/ssd_head.py` 后者是前者的子类，接下来我们来看看`SSDHead` 我们先看config文件：

```  python
	bbox_head=dict(
        type='SSDHead',
        input_size=input_size,
        in_channels=(512, 1024, 512, 256, 256, 256),
        num_classes=21, # for VOC dadtaset
        anchor_strides=(8, 16, 32, 64, 100, 300),
        basesize_ratio_range=(0.2, 0.9),
        anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
        target_means=(.0, .0, .0, .0),
        target_stds=(0.1, 0.1, 0.2, 0.2))
```

`in_channels` 为`extract_feat` 抽取得到的`feature_maps` 的channels, `anchor_strides` 为每层`feature_map` 对应原图的下采样的倍数。

然后回到`SSDHead` 中，从`SingleStageDetector` 的前向过程可以看出`SSDHead` 中主要实现两个功能：

* 1、对各个`feature_maps` 上面预设的`anchor`进行回归和分类。

* 2、求出回归和分类这两项任务的`loss`，分别为：`loss_cls` 、`loss_reg` 。 

  1、首先来看看在`SSDHead` 中是如何设定`anchor` 的，下面给出代码：

  ```python
  def __init__(self,
               input_size=300,
               num_classes=81,
               in_channels=(512, 1024, 512, 256, 256, 256),
               anchor_strides=(8, 16, 32, 64, 100, 300),
               basesize_ratio_range=(0.1, 0.9),
               anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2], [2]),
               target_means=(.0, .0, .0, .0),
               target_stds=(1.0, 1.0, 1.0, 1.0)):
      super(AnchorHead, self).__init__()
      self.input_size = input_size
      self.num_classes = num_classes
      self.in_channels = in_channels
      self.cls_out_channels = num_classes
      num_anchors = [len(ratios) * 2 + 2 for ratios in anchor_ratios]
      # 这一步得到对于不同feature_map，每个cell上设定的anchor数量为 [4, 6, 6, 6, 4, 4]
      # 设定回归和分类所需要的卷积
      reg_convs = []
      cls_convs = []
      for i in range(len(in_channels)):
          reg_convs.append(
              nn.Conv2d(
                  in_channels[i],
                  num_anchors[i] * 4,
                  kernel_size=3,
                  padding=1))
          cls_convs.append(
              nn.Conv2d(
                  in_channels[i],
                  num_anchors[i] * num_classes,
                  kernel_size=3,
                  padding=1))
      # 将reg_convs、和cls_convs注册到模型中，这样才能进行backward
      self.reg_convs = nn.ModuleList(reg_convs)
      self.cls_convs = nn.ModuleList(cls_convs)
  
      min_ratio, max_ratio = basesize_ratio_range # [0.2, 0.9] --> [2, 6] 第一层特征层的ratio为0.1
      min_ratio = int(min_ratio * 100)
      max_ratio = int(max_ratio * 100)
      step = int(np.floor(max_ratio - min_ratio) / (len(in_channels) - 2)) # step = 17
      min_sizes = []
      max_sizes = []
      # 对于每一张feature_map, 其上anchor的尺度是不一样的，所以要得到6个不同的尺度，每个尺度又有最小和最大两种size，并且这一层feature_map对应的max_size是下一层的min_size。
      for r in range(int(min_ratio), int(max_ratio) + 1, step):
          min_sizes.append(int(input_size * r / 100))
          max_sizes.append(int(input_size * (r + step) / 100))
      # 由于第一层feature_map的raitio是另外设定的，所以在这一步额外加上，
      if input_size == 300:
          if basesize_ratio_range[0] == 0.15:  # SSD300 COCO
              min_sizes.insert(0, int(input_size * 7 / 100))
              max_sizes.insert(0, int(input_size * 15 / 100))
          elif basesize_ratio_range[0] == 0.2:  # SSD300 VOC
              min_sizes.insert(0, int(input_size * 10 / 100))
              max_sizes.insert(0, int(input_size * 20 / 100))
      elif input_size == 512:
          if basesize_ratio_range[0] == 0.1:  # SSD512 COCO
              min_sizes.insert(0, int(input_size * 4 / 100))
              max_sizes.insert(0, int(input_size * 10 / 100))
          elif basesize_ratio_range[0] == 0.15:  # SSD512 VOC
              min_sizes.insert(0, int(input_size * 7 / 100))
              max_sizes.insert(0, int(input_size * 15 / 100))
      self.anchor_generators = []
      self.anchor_strides = anchor_strides
      # 下面这一个循环是对于每层feature_map, 求它的base_anchors，对应的数量定义在了num_anchors中，即[4, 6, 6, 6, 4, 4], 这些base_anchors其实就是映射到原始图像上的对应尺度的第一个anchor，以其作为基础再产生所有的anchors。 
      for k in range(len(anchor_strides)):
          # 每个base_anchors的基准size
          base_size = min_sizes[k]
          stride = anchor_strides[k]
          # 直接给定base_anchors的中心坐标（理解为什么用stride求得），对于同一层feature_map所有的base_anchor是共用一个center的
          ctr = ((stride - 1) / 2., (stride - 1) / 2.)
          # 这个scale很关键，我将在下面仔细分析其作用
          scales = [1., np.sqrt(max_sizes[k] / min_sizes[k])]
          ratios = [1.]
          # 对于不同的feature_map得到4个或6个长宽比(其实这里只有3或5，还有一个定义在了scales中)
          for r in anchor_ratios[k]:
              ratios += [1 / r, r]  # 4 or 6 ratio
          # 回顾一下上面所做的准备，对于每一层featur_map，通过其对应的stirde求出在他上面的base_anchors的中心坐标(这里所说的所有的anchor的坐标都是映射到原图上的)，根据这个中心坐标和这些base_anchors的基准大小、几组不同的长宽比，来得到每一个base_anchor的角点表示（左上角坐标(x1, y1)，右下角坐标(x2, y2)）,这就是AnchorGenerator()实现的内容。
          anchor_generator = AnchorGenerator(
              base_size, scales, ratios, scale_major=False, ctr=ctr)
          indices = list(range(len(ratios)))
          indices.insert(1, len(indices))
          anchor_generator.base_anchors = torch.index_select(
              anchor_generator.base_anchors, 0, torch.LongTensor(indices))
          self.anchor_generators.append(anchor_generator)
  
      self.target_means = target_means
      self.target_stds = target_stds
      self.use_sigmoid_cls = False
      self.use_focal_loss = False
  ```

对于以上代码中的 `scales` 做一个详细的解释：

`SSD` 论文中给出的长宽比设定是`1`  、`1/2` 、`2` 、`1/3`  、`3` 、和改变尺度的 `1` ，这里改变尺度的`1` 就是由`scales` 来体现了，在论文中改变的尺度为 $$ s_k'=\sqrt[]{s_ks_{k+1}}$$  其中$$s_k$$=`min_size[k]`, $$s_{k+1}$$=`max_size[k]` , 所以代码中的`scales=[1., np.sqrt(max_sizes[k] / min_sizes[k])]` 如果我们对其左乘一个$s_k$ 的话，就变成了$$scales=[s_k,  s_k\sqrt[]{\frac{s_{k+1}}{s_k}}]$$, 即$$ scales=[s_k, s_k']$$ , 由此我们就得到了原尺度和变化之后的尺度。

​	2、回到`SingleStageDetector` 中，继续分析`losses` 是怎么计算的， 在`SingleStageDetector` 中通过调用 `SSDHead` 类中的 `loss` 函数来计算回归和分类的$$ loss$$ ,下面是`SSDHead` 中`loss` 方法的代码：

```python
def loss(self,
         cls_scores,
         bbox_preds,
         gt_bboxes,
         gt_labels,
         img_metas,
         cfg,
         gt_bboxes_ignore=None):
    # 传入的参数分别是：类别得分，预测框与default anchor之间的offsets，ground truth框，ground truth labels，img_meta和train_cfg
    featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    assert len(featmap_sizes) == len(self.anchor_generators)
	# anchor_list中存放的是一个batch中每一张图片的每个特征图的default anchor的角点坐标， valid_flag_list是个真值表，其上1出的位置表示每个特征图中有效(中心坐标在原图片上)的default anchor的idx值。
    anchor_list, valid_flag_list = self.get_anchors(
        featmap_sizes, img_metas)
    # 每个default anchor都找一个与其iou最大的gt_bboxes与其对应, 根据设定的阈值pos_iou_thr=0.5，如果其IOU小于该阈值，设为bg，如果大于则将其lables设为gt_bboxes对应的label，为保证每个gt_bbox都能有一个default_anchor去回归，还需要对每个gt_bbox都找一个与其有最大iou的default anchor，并将找到的这个default anchor的lables设为gt_bboxes对应的label。
    cls_reg_targets = anchor_target(
        anchor_list,
        valid_flag_list,
        gt_bboxes,
        img_metas,
        self.target_means,
        self.target_stds,
        cfg,
        gt_bboxes_ignore_list=gt_bboxes_ignore,
        gt_labels_list=gt_labels,
        label_channels=1,
        sampling=False,
        unmap_outputs=False)
    if cls_reg_targets is None:
        return None
    (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets
    # batch_size的大小
    num_images = len(img_metas)
	# all_cls_scores： 
    # 每个anchor的类别得分size为： (batch_size, num_all_anchors, num_classes)
    # all_labels\all_label_weights：
    # 为每个anchor找到的target_labels\权重size：(batch_size, num_all_anchors)
    # all_bbox_preds 
    # 网络为每个default anchor预测出的与offsets，size：(batch_size, num_all_anchors, 4)
    # all_bbox_target (batch_size, num_all_anchors, 4)
    # 每个default anchor与gt_bboxes的offsets， size:(batch_size, num_all_anchors, 4)
    # all_bbox_weights (batch_size, num_all_anchors, 4)
    all_cls_scores = torch.cat([
        s.permute(0, 2, 3, 1).reshape(
            num_images, -1, self.cls_out_channels) for s in cls_scores
    ], 1)
    all_labels = torch.cat(labels_list, -1).view(num_images, -1)
    all_label_weights = torch.cat(label_weights_list, -1).view(
        num_images, -1)
    all_bbox_preds = torch.cat([
        b.permute(0, 2, 3, 1).reshape(num_images, -1, 4)
        for b in bbox_preds
    ], -2)
    all_bbox_targets = torch.cat(bbox_targets_list, -2).view(
        num_images, -1, 4)
    all_bbox_weights = torch.cat(bbox_weights_list, -2).view(
        num_images, -1, 4)

    losses_cls, losses_reg = multi_apply(
        self.loss_single,
        all_cls_scores,
        all_bbox_preds,
        all_labels,
        all_label_weights,
        all_bbox_targets,
        all_bbox_weights,
        num_total_samples=num_total_pos,
        cfg=cfg)
    return dict(loss_cls=losses_cls, loss_reg=losses_reg)
```



## 三、Guided anchor源码解析

### 3.1 论文解读

首先回顾一下 `Guided anchor` 的网络架构，如下图所示：

![framework structure](/home/vansikey/Pictures/guidedanchor.png)

首先由 `backbone`  提取各层 `feature map` ，然后每层 `feature map` 都会通过 `Guided anchoring` 层来得到 `anchors` 和 `adaption feature` ，在 `Guided anchoring` 层中，定义了 $$ N_L$$ 与 $$N_S$$ 两个模块，分别用于预测 `anchor` 的位置 (location) 与长宽 (shape)，都是通过使用一个 $$1\times1$$ 的卷积来实现的，在得到了 `anchor` 的位置图之后，对其再使用一个 $$1\times 1$$ 的conv来提取用于`deformable conv`的`offset field`，结合提取的`offset field`与原feature map使用deformable conv来得到`adaption feature map` 。

在训练 `Guided anchoring` 时，需要利用两个 `loss` 来监督location prediction 和 shape prediciton的过程，这样就需要对feature map上的每个存在`anchor` 的cell分配一个 `location target`和 `shape target`：

1、__anchor location target__

![anchor location target](/home/vansikey/Pictures/loctar.png)

如上图所示，首先我们将原图的ground-truth的bounding box  $$(x_g, y_g, w_g, h_g)$$ 映射到**对应层级**的 `feature map` 上去得到 $$(x_g', y_g', w_g', h_g')$$（*这里的对应层级是指：对于大物体框，交由高层feature去预测，小物体框，交由底层feature去预测*）， 以 $$R(x, y, w, h)$$ 作为矩形框区域的符号标记，其中心点为 $$(x, y)$$，长宽为 $$w\times h$$ ，我们希望预测得到的`anchor` 的中心点距离groudn-truth很近以得到更大的初始IoU，因此对每个ground-truth bounding box的矩形区域定义三种不同类型的区域并分别配以不同的target：

1）中心区域（center region）：$$CR=\mathcal{R}(x_g', y_g', \sigma_1w', \sigma_1h')$$ ，如上图的绿色部分。$$CR$$ 中的每个Pixels都作为正样本（正样本loss）。

2）无视区域（ignore region）：$$ IR=\mathcal{R}(x_g', y_g', \sigma_2w', \sigma_2h') \setminus{CR}$$ ，如上图的黄色部分，在$$IR$$ 中，不分配target，即在训练过程中，我们将落在此区域的 `Prediction` 无视掉 （不计算loss）。

3）外围区域（outside region）：$$OR$$ ，$$OR$$ 即整张 `feature map` 除了$$CR$$ 和 $$IR$$ ，将其标记为负样本（负样本loss）。

> 这里有一个需要注意的地方：如上图所示，对于大物体框(白羊)是交由最高层的feature去预测，并将其相邻层的对应物体的区域设为ignore region，在图中表示为第二层的黄色部分，而对于较小物体(黑羊)交由较低层的feature去预测，并通样将其相邻的两层feature对应的位置设为ignore region，我的理解是这样做的目的是为了控制anchor的数量并得到最好的anchor，以缓解正负样本不均衡的问题，由于相邻两层feature的尺度比较接近，这样做可以保证相邻层不产生对应物体的anchor，每个gt_bbox只交由尺度最符合的那一层feature去管。

2、__anchor shape target__

以前传统的方法是：先对每一个 `gt-bbx` 都找一个与其有最大 $$IoU$$  的 `anchor` ，并将该 `anchor` 的 `target` 标为该 `gt-bbx` ，然后再对每一个 `anchor` 都找一个与其有最大 $$IoU$$ 的 `gt-bbx` ，并将这些最大 $$IoU$$ 大于正样本阈值的 `anchor` 的 `target` 标记为对应的 `gt-bbx` 。但是在 `Guided anchor` 中，$$w$$ 和 $$ h$$ 不是预先设定好的，所以无法直接得到 `anchor` 与 `gt-bbx` 的 $$IoU$$ ，为了解决这个问题论文中提出来一种计算 $$vIoU$$ 的方法来定义target，该方法定义如下：

1）首先我们将 `feature map` 上的每个cell对应的anchor的w，h视为变量即 $$a_{wh}=(x_0, y_0, w, h)$$ ，对于ground-truth bounding box 定义为 $$gt=(x_g, y_g, w_g, h_g)$$ ，则 $$vIoU=\max \limits_{w>0, h>0}{IoU_{normal}(a_{wh}, gt)}$$ ，其中$$IoU_{normal}$$ 是$$IoU$$ 的一般定义，$$w$$ 和 $$h$$ 是变量 。

2）由于直接利用数学推导计算 $$vIoU$$  不便实现网络的端到端训练，所以得找到一种数值方法去逼近它，即，对于给定的 `anchor` 位置 $$(x_0, y_0)$$ ，选取 $$9$$ 对不同的 $$(w, h)$$ （3种尺度，3种长宽比）分别计算 $$IoU_{normal}$$ ，并选取其中最大的作为 $$vIoU$$ 。

以此 $$vIoU$$ 替换传统 `anchor target` 方法中的 $$IoU$$ 从而通过传统的 `anchor target` 方法得到每个正样本 `ahchor` 的 `target` 即对应 `gt-bbx` 的 $$(w_g, h_g)$$ ，然后采用 `bounded IoU loss` 去优化 `anchor` 与对应 `gt_bbx`   之间的 $$IoU$$ ，由于`anchor` 的位置 $$(x_0, y_0)$$ 是固定的，所以去掉 `bounded IoU loss` 公式中的中心坐标项，只保留长宽项，得到一下的定义方式：

​								$$\mathcal{L}_{shape} = \mathcal{L}_1(1-\min{(\frac{w_g}{w}, \frac{w}{w_g})})+\mathcal{L}_1(1-\min(\frac{h_g}{h}, \frac{h}{h_g}))$$

相比于传统的 $$\mathcal{L_1}$$ 和 $$\mathcal{L_2}$$ loss，这种计算方式对 `anchor` 的长宽尺度不敏感，不会出现大 `anchor` 与小 `anchor` 的loss的数值差距过大的情况 。

### 3.2 mmdetection中的实现

`Guided anchor` 代码实现在 `mmdet/models/guided_anchor_head.py` 中，该文件中定义了 `GuidedAnchorHead` 类与 `FeatureAdaption` 类，先来看看 `GuidedAnchorHead` 类的 构造函数：

### __GuidedAnchorHead（）__

#### __\__init__\__()

```python
    def __init__(self,
                 num_classes,  		  # 用于Guided anchor的cls_pred
                 in_channels,  		  # 输入feature的channels个人感觉多余了
                 feat_channels=256,   # feature的channels就是in_channels
                 octave_base_scale=8, # 在feature_map上设定anchor的基础尺寸
                 scales_per_octave=3, # 在feature_map上为anchor设定3中不同的尺寸
                 octave_ratios=[0.5, 1.0, 2.0], # anchor的三种不同长宽比
                 anchor_strides=[4, 8, 16, 32, 64], 
                 anchor_base_sizes=None,
                 anchoring_means=(.0, .0, .0, .0),
                 anchoring_stds=(1.0, 1.0, 1.0, 1.0),
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0),
                 deformable_groups=4,
                 loc_filter_thr=0.01,
                 loss_loc=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_shape=dict(
                     type='BoundedIoULoss',
                     beta=0.2,
                     loss_weight=1.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0)):
        super(AnchorHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        # 用于产生square的scale
        self.octave_base_scale = octave_base_scale
        # anchor的尺度的个数为3
        self.scales_per_octave = scales_per_octave
        # 得到anchor的3个尺度
        self.octave_scales = octave_base_scale * np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        # 为每个cell计算vIoU所用的anchor，数量------9 该anchor在代码中是以approxs来表示的
        self.approxs_per_octave = len(self.octave_scales) * len(octave_ratios) # =9
        # approxs的三种长宽比
        self.octave_ratios = octave_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.anchoring_means = anchoring_means
        self.anchoring_stds = anchoring_stds
        self.target_means = target_means
        self.target_stds = target_stds
        self.deformable_groups = deformable_groups
        # 用于过滤掉loc_pred值低的cell的阈值
        self.loc_filter_thr = loc_filter_thr
        self.approx_generators = []
        self.square_generators = []
        # 为每一层feature maps产生一个base anchors
        for anchor_base in self.anchor_base_sizes: # 每层feature map 对应的stride
            # Generators for approxs （用于得到9对不同的（w,h）来计算vIoU）
            self.approx_generators.append(
                AnchorGenerator(anchor_base, self.octave_scales,
                                self.octave_ratios))
            # Generators for squares
            self.square_generators.append(
                AnchorGenerator(anchor_base, [self.octave_base_scale], [1.0]))
        # one anchor per location
        # featuer map上的每个cell只有一个anchor
        self.num_anchors = 1
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.cls_focal_loss = loss_cls['type'] in ['FocalLoss']
        self.loc_focal_loss = loss_loc['type'] in ['FocalLoss']
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes - 1
        else:
            self.cls_out_channels = self.num_classes

        # build losses
        self.loss_loc = build_loss(loss_loc)
        self.loss_shape = build_loss(loss_shape)
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        self._init_layers()
```

在 `__init__` 中定义了一些需要的参数和 `square`、`approxs`  还有计算 `loss` 所用的函数，下面解释一下 `square` 和 `approx` ：

> * __squares__
>
> 在论文中我们得到的shape_pred是 $$dw$$ 和 $$dh$$ ，并不是真正的 $$w$$ 和 $$h$$ ，需要通过公式：
>
> ​									$$w = \sigma s e^{dw}$$ , $$h = \sigma s e^{dh}$$ 
>
> 来得到，所以在feature_map的每个cell上都需要得到一个scale为 $$\sigma s$$ 的square anchor。
>
> * __approxs__
>
> 论文中为了给shape_pred分配target使用的是 $$vIoU$$ ，需要用9组不同的 (w, h) 去计算，这里的approxs就是在feature_map的每个cell上产生9个anchor（通过使用三种不同的scale和三种不同的长宽比来得到） 

下面是为网络前向过程准备的卷积层定义在 `_init_layers` 中：

#### __\_init_layers()__

```python
    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        # 定义用于得到anchor位置(location)和长宽(shape)的卷积
        # 由于得到的位置的预测图是一张heat map所以out channel为1
        self.conv_loc = nn.Conv2d(self.feat_channels, 1, 1) 
        # anchor的shape为(w, h)两种属性, 故out channel=num_anchors*2
        self.conv_shape = nn.Conv2d(self.feat_channels, self.num_anchors * 2, 1)
        # 根据得到的shape使用deformable conv来调整feature_map
        self.feature_adaption = FeatureAdaption(
            self.feat_channels,
            self.feat_channels,
            kernel_size=3,
            deformable_groups=self.deformable_groups)
        self.conv_cls = MaskedConv2d(self.feat_channels,
                                     self.num_anchors * self.cls_out_channels,
                                     1)
        self.conv_reg = MaskedConv2d(self.feat_channels, self.num_anchors * 4,
                                     1)
```

#### __forward()__

```python
    def forward_single(self, x):
        loc_pred = self.conv_loc(x)
        shape_pred = self.conv_shape(x)
        x = self.feature_adaption(x, shape_pred)
        # masked conv is only used during inference for speed-up
        if not self.training:
            mask = loc_pred.sigmoid()[0] >= self.loc_filter_thr
        else:
            mask = None
        cls_score = self.conv_cls(x, mask)
        bbox_pred = self.conv_reg(x, mask)
        return cls_score, bbox_pred, shape_pred, loc_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)
```

#### get_sample_approxs()

将base approxs平铺到feature_maps上去:

```python
    def get_sampled_approxs(self, featmap_sizes, img_metas, cfg):
        """Get sampled approxs and inside flags according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: approxes of each image, inside flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # approxes for one time
        multi_level_approxs = []
        # 遍历每一张feature
        for i in range(num_levels):
            # 每一张feature的每个cell上都产生3个不同scale和3个不同ratio的anchor来求vIoU的
            approxs = self.approx_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            # approx.shape = [num_anchors_for_this_feat, 4]
            multi_level_approxs.append(approxs)
        # 一个batch中每张img的approx都相同，所以直接copy就行
        approxs_list = [multi_level_approxs for _ in range(num_imgs)]

        # for each image, we compute inside flags of multi level approxes
        inside_flag_list = []
        # 遍历每张img
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            multi_level_approxs = approxs_list[img_id]
            # 对每张img，遍历其每张feat
            for i in range(num_levels):
                # 获取对应feat的approx
                approxs = multi_level_approxs[i]
                # 获取对应feat的stride
                anchor_stride = self.anchor_strides[i]
                # 获取feat的size
                feat_h, feat_w = featmap_sizes[i]
                # 获取原图的size
                h, w, _ = img_meta['pad_shape']
                # 将feature map上的anchor映射到原图上并过滤中心点在原图外的anchor
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                # for every anchors in feature maps, set a flag
                flags = self.approx_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                # len(flags) = num_anchors_for_this_level
                # 再次过滤到边缘超出原图 allowed_border大小的anchor
                inside_flags_list = []
                for i in range(self.approxs_per_octave):
                    split_valid_flags = flags[i::self.approxs_per_octave]
                    split_approxs = approxs[i::self.approxs_per_octave, :]
                    inside_flags = anchor_inside_flags(
                        split_approxs, split_valid_flags,
                        img_meta['img_shape'][:2], cfg.allowed_border)
                    inside_flags_list.append(inside_flags)
                # inside_flag for a position is true if any anchor in this
                # position is true
                # 用torch.stack, 实现list->tensor
                inside_flags = (
                    torch.stack(inside_flags_list, 0).sum(dim=0) > 0)
                # inside_flags表示对于该feature_map上的每个cell是否存在至少一个valid anchor
                multi_level_flags.append(inside_flags) # 将一张img的所有feat的inside_flags打														包成一个列表
            inside_flag_list.append(multi_level_flags) # 将每张img的multi_level_flags打包成														列表
        return approxs_list, inside_flag_list
```

#### __get_anchors()__

得到Guidied anchor和square :

```python
 def get_anchors(self,
                 featmap_sizes,
                 shape_preds,
                 loc_preds,
                 img_metas,
                 use_loc_filter=False):
        """Get squares according to feature map sizes and guided
        anchors.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            shape_preds (list[tensor]): Multi-level shape predictions.
            loc_preds (list[tensor]): Multi-level location predictions.
            img_metas (list[dict]): Image meta info.
            use_loc_filter (bool): Use loc filter or not.

        Returns:
            tuple: square approxs of each image, guided anchors of each image,
                loc masks of each image
        """
        # 即batch size
        num_imgs = len(img_metas)
        # 一个img中的feature map层数
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # squares for one time
        # squares中每个anchor保留了位置信息和的 (w,h) = (sigma*s, sigma*s)
        multi_level_squares = []
        for i in range(num_levels):
            squares = self.square_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_squares.append(squares)
        squares_list = [multi_level_squares for _ in range(num_imgs)]

        # for each image, we compute multi level guided anchors
        guided_anchors_list = []
        loc_mask_list = []
        # 遍历每张图片
        for img_id, img_meta in enumerate(img_metas):
            multi_level_guided_anchors = []
            multi_level_loc_mask = []
            # 遍历每张feature
            for i in range(num_levels):
                squares = squares_list[img_id][i]
                shape_pred = shape_preds[i][img_id]
                loc_pred = loc_preds[i][img_id]
                guided_anchors, loc_mask = self.get_guided_anchors_single(
                    squares,
                    shape_pred,
                    loc_pred,
                    use_loc_filter=use_loc_filter)
                multi_level_guided_anchors.append(guided_anchors)
                multi_level_loc_mask.append(loc_mask)
            guided_anchors_list.append(multi_level_guided_anchors)
            loc_mask_list.append(multi_level_loc_mask)
        return squares_list, guided_anchors_list, loc_mask_list
```

#### __get_guided_anchors_single()__

```python
    def get_guided_anchors_single(self,
                                  squares,
                                  shape_pred,
                                  loc_pred,
                                  use_loc_filter=False):
        """Get guided anchors and loc masks for a single level (对每一张feature求guided anchor).

        Args:
            square (tensor): Squares of a single level.
            shape_pred (tensor): Shape predections of a single level.
            loc_pred (tensor): Loc predections of a single level.
            use_loc_filter (list[tensor]): Use loc filter or not.

        Returns:
            tuple: guided anchors, location masks
        """
        # calculate location filtering mask
        # 求sigmiod(), 得到conf
        # .detach()从计算图中抽取出来，不计算梯度
        loc_pred = loc_pred.sigmoid().detach()
        if use_loc_filter:
            loc_mask = loc_pred >= self.loc_filter_thr
        else:
            loc_mask = loc_pred >= 0.0
        mask = loc_mask.permute(1, 2, 0).expand(-1, -1, self.num_anchors)
        mask = mask.contiguous().view(-1)
        # calculate guided anchors
        # 将conf小于阈值的位置去掉
        squares = squares[mask]
        anchor_deltas = shape_pred.permute(1, 2, 0).contiguous().view(
            -1, 2).detach()[mask]
        bbox_deltas = anchor_deltas.new_full(squares.size(), 0)
        bbox_deltas[:, 2:] = anchor_deltas
        # 将通过mask过滤后的bbox_deltas与square，map成guided_anchors
        guided_anchors = delta2bbox(
            squares,
            bbox_deltas,
            self.anchoring_means,
            self.anchoring_stds,
            wh_ratio_clip=1e-6)
        return guided_anchors, mask
```

#### __loss_shape_single()__

计算每层feature map的loss_shape, 在loss中调用：

```python
    def loss_shape_single(self, shape_pred, bbox_anchors, bbox_gts,
                          anchor_weights, anchor_total_num):

        shape_pred = shape_pred.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        bbox_anchors = bbox_anchors.contiguous().view(-1, 4)
        bbox_gts = bbox_gts.contiguous().view(-1, 4)
        anchor_weights = anchor_weights.contiguous().view(-1, 4)
        bbox_deltas = bbox_anchors.new_full(bbox_anchors.size(), 0)
        bbox_deltas[:, 2:] += shape_pred
        # filter out negative samples to speed-up weighted_bounded_iou_loss
        inds = torch.nonzero(anchor_weights[:, 0] > 0).squeeze(1)
        bbox_deltas_ = bbox_deltas[inds]
        bbox_anchors_ = bbox_anchors[inds]
        bbox_gts_ = bbox_gts[inds]
        anchor_weights_ = anchor_weights[inds]
        pred_anchors_ = delta2bbox(
            bbox_anchors_,
            bbox_deltas_,
            self.anchoring_means,
            self.anchoring_stds,
            wh_ratio_clip=1e-6)
        loss_shape = self.loss_shape(
            pred_anchors_,
            bbox_gts_,
            anchor_weights_,
            avg_factor=anchor_total_num)
        return loss_shape
```

#### __loss_loc_single()__

计算每层feature_map的loss_loc, 在loss中调用：

```python
    def loss_loc_single(self, loc_pred, loc_target, loc_weight, loc_avg_factor,
                        cfg):
        loss_loc = self.loss_loc(
            loc_pred.reshape(-1, 1),
            loc_target.reshape(-1, 1).long(),
            loc_weight.reshape(-1, 1),
            avg_factor=loc_avg_factor)
        return loss_loc
```

#### __loss()__

计算loss，在loss中计算了四种loss：

> For anchor :
>
> * loss_loc
> * loss_shape
>
> For predicted bounding box:
>
> * loss_cls
> * loss_bbox

```python
    def loss(self,
             cls_scores,    # 每个anchor的类别得分 
             				#shape=[(batch_size, num_classes, h, w), ...]
             bbox_preds,    # 每个anchor粗调结果        
             				#shape=[(batch_size, num_anchors*4, h, w), ...]
             shape_preds,   # guided anchor的shape    
             				#shape=(batch_size, num_anchors*2, h, w)
             loc_preds,     # guided anchor的位置      
             				#shape=(batch_size, num_anchors, h, w)
             gt_bboxes,     # ground-truth bounding box
             gt_labels,     # ground-truth labels
             img_metas,     # 图片的各种信息
             cfg,           # 配置文件
             gt_bboxes_ignore=None):
        # 首先得到每张feature的size，组合成一个列表, cls_scores是一个列表每个元素表示一层特征
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.approx_generators)

        # get loc targets
        loc_targets, loc_weights, loc_avg_factor = ga_loc_target(
            gt_bboxes,
            featmap_sizes,
            self.octave_base_scale,
            self.anchor_strides,
            center_ratio=cfg.center_ratio,
            ignore_ratio=cfg.ignore_ratio)

        # get sampled approxes
        approxs_list, inside_flag_list = self.get_sampled_approxs(
            featmap_sizes, img_metas, cfg)
        # get squares and guided anchors
        squares_list, guided_anchors_list, _ = self.get_anchors(
            featmap_sizes, shape_preds, loc_preds, img_metas)

        # get shape targets
        sampling = False if not hasattr(cfg, 'ga_sampler') else True
        shape_targets = ga_shape_target(
            approxs_list,
            inside_flag_list,
            squares_list,
            gt_bboxes,
            img_metas,
            self.approxs_per_octave,
            cfg,
            sampling=sampling)
        if shape_targets is None:
            return None
        (bbox_anchors_list, bbox_gts_list, anchor_weights_list, anchor_fg_num,
         anchor_bg_num) = shape_targets
        anchor_total_num = (
            anchor_fg_num if not sampling else anchor_fg_num + anchor_bg_num)

        # get anchor targets
        sampling = False if self.cls_focal_loss else True
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(
            guided_anchors_list,
            inside_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos if self.cls_focal_loss else num_total_pos +
            num_total_neg)

        # get classification and bbox regression losses
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg)

        # get anchor location loss
        losses_loc = []
        for i in range(len(loc_preds)):
            loss_loc = self.loss_loc_single(
                loc_preds[i],
                loc_targets[i],
                loc_weights[i],
                loc_avg_factor=loc_avg_factor,
                cfg=cfg)
            losses_loc.append(loss_loc)

        # get anchor shape loss
        losses_shape = []
        for i in range(len(shape_preds)):
            loss_shape = self.loss_shape_single(
                shape_preds[i],
                bbox_anchors_list[i],
                bbox_gts_list[i],
                anchor_weights_list[i],
                anchor_total_num=anchor_total_num)
            losses_shape.append(loss_shape)

        return dict(
            loss_cls=losses_cls,
            loss_bbox=losses_bbox,
            loss_shape=losses_shape,
            loss_loc=losses_loc)
```

在计算 `loss` 的时候需要对shape_pred和loc_pred分配target，方法分别定义在 `mmdet/core/anchor/guided_anchor_target.py` 中：

#### ga_loc_target()

```python
def ga_loc_target(gt_bboxes_list,
                  featmap_sizes,
                  anchor_scale,
                  anchor_strides,
                  center_ratio=0.2,
                  ignore_ratio=0.5):
    """Compute location targets for guided anchoring.

    Each feature map is divided into positive, negative and ignore regions.
    - positive regions: target 1, weight 1
    - ignore regions: target 0, weight 0
    - negative regions: target 0, weight 0.1

    Args:
        gt_bboxes_list (list[Tensor]): Gt bboxes of each image.
        featmap_sizes (list[tuple]): Multi level sizes of each feature maps.
        anchor_scale (int): Anchor scale.
        anchor_strides ([list[int]]): Multi level anchor strides.
        center_ratio (float): Ratio of center region.
        ignore_ratio (float): Ratio of ignore region.

    Returns:
        tuple
    """
    # batch_size
    img_per_gpu = len(gt_bboxes_list)
    # feature 的层数
    num_lvls = len(featmap_sizes)
    r1 = (1 - center_ratio) / 2
    r2 = (1 - ignore_ratio) / 2

    all_loc_targets = []
    all_loc_weights = []
    all_ignore_map = []
    # 遍历每层feature
    for lvl_id in range(num_lvls):
        h, w = featmap_sizes[lvl_id]
        # 初始化loc_target为0，shape与loc_preds相同 (batch_size, 1, h, w)
        loc_targets = torch.zeros(img_per_gpu,
                                  1,
                                  h,
                                  w,
                                  device=gt_bboxes_list[0].device,
                                  dtype=torch.float32)
        # 初始化loc_weights为-1， shape与loc_targets相同
        loc_weights = torch.full_like(loc_targets, -1)
        # 初始化ignore_map为0， shape与loc_targets相同
        ignore_map = torch.zeros_like(loc_targets)
        all_loc_targets.append(loc_targets)
        all_loc_weights.append(loc_weights)
        all_ignore_map.append(ignore_map)
    for img_id in range(img_per_gpu):
        gt_bboxes = gt_bboxes_list[img_id]
        # 得到所有gt_bboxes的尺度scale(行向量), 即对面积开根号
        scale = torch.sqrt((gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) *
                           (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1))
        min_anchor_size = scale.new_full(
            (1, ), float(anchor_scale * anchor_strides[0]))
        # assign gt bboxes to different feature levels w.r.t. their scales
        # 将每一个gt bboxes根据其scale分别分配到不同层的feature上
        # 小物体交由低层feature去预测, 大物体交由高层feature去预测
        target_lvls = torch.floor(
            torch.log2(scale) - torch.log2(min_anchor_size) + 0.5)
        target_lvls = target_lvls.clamp(min=0, max=num_lvls - 1).long()
        for gt_id in range(gt_bboxes.size(0)):
            lvl = target_lvls[gt_id].item()
            # rescaled to corresponding feature map
            # 将gt映射到对应的feature层上去
            gt_ = gt_bboxes[gt_id, :4] / anchor_strides[lvl]
            # 调用calc_region去得到ignore和positive区域
            # calculate ignore regions
            ignore_x1, ignore_y1, ignore_x2, ignore_y2 = calc_region(
                gt_, r2, featmap_sizes[lvl])
            # calculate positive (center) regions
            ctr_x1, ctr_y1, ctr_x2, ctr_y2 = calc_region(
                gt_, r1, featmap_sizes[lvl])
            # 将对应loc_target的positive region标记为1
            all_loc_targets[lvl][img_id, 0, ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 +
                                 1] = 1
            # 将ignore region的权重标记为0
            all_loc_weights[lvl][img_id, 0, ignore_y1:ignore_y2 +
                                 1, ignore_x1:ignore_x2 + 1] = 0
            # 将positive region的权重标记为1, 其他区域的权重为-1
            all_loc_weights[lvl][img_id, 0, ctr_y1:ctr_y2 + 1, ctr_x1:ctr_x2 +
                                 1] = 1
            # 下面这两个判断语句执行的的意思是将这个gt_bbox只交由对应的feature map去管，
            # 与该feature相邻两层的
            # feature对应gt_bbox的区域设为ignore(即在ignore_map上设为1)
            # calculate ignore map on nearby low level feature
            if lvl > 0:
                d_lvl = lvl - 1
                # rescaled to corresponding feature map
                gt_ = gt_bboxes[gt_id, :4] / anchor_strides[d_lvl]
                ignore_x1, ignore_y1, ignore_x2, ignore_y2 = calc_region(
                    gt_, r2, featmap_sizes[d_lvl])
                all_ignore_map[d_lvl][img_id, 0, ignore_y1:ignore_y2 +
                                      1, ignore_x1:ignore_x2 + 1] = 1
            # calculate ignore map on nearby high level feature
            if lvl < num_lvls - 1:
                u_lvl = lvl + 1
                # rescaled to corresponding feature map
                gt_ = gt_bboxes[gt_id, :4] / anchor_strides[u_lvl]
                ignore_x1, ignore_y1, ignore_x2, ignore_y2 = calc_region(
                    gt_, r2, featmap_sizes[u_lvl])
                all_ignore_map[u_lvl][img_id, 0, ignore_y1:ignore_y2 +
                                      1, ignore_x1:ignore_x2 + 1] = 1
    for lvl_id in range(num_lvls):
        # ignore negative regions w.r.t. ignore map
        all_loc_weights[lvl_id][(all_loc_weights[lvl_id] < 0)
                                & (all_ignore_map[lvl_id] > 0)] = 0
        # set negative regions with weight 0.1
        all_loc_weights[lvl_id][all_loc_weights[lvl_id] < 0] = 0.1
    # loc average factor to balance loss
    loc_avg_factor = sum(
        [t.size(0) * t.size(-1) * t.size(-2) for t in all_loc_targets]) / 200
    return all_loc_targets, all_loc_weights, loc_avg_factor
```

#### ga_shape_target()

```python
def ga_shape_target(approx_list,
                    inside_flag_list,
                    square_list,
                    gt_bboxes_list,
                    img_metas,
                    approxs_per_octave,
                    cfg,
                    gt_bboxes_ignore_list=None,
                    sampling=True,
                    unmap_outputs=True):
    """Compute guided anchoring targets.

    Args:
        approx_list (list[list]): Multi level approxs of each image.
        inside_flag_list (list[list]): Multi level inside flags of each image.
        square_list (list[list]): Multi level squares of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        approxs_per_octave (int): number of approxs per octave
        cfg (dict): RPN train configs.
        gt_bboxes_ignore_list (list[Tensor]): ignore list of gt bboxes.
        sampling (bool): sampling or not.
        unmap_outputs (bool): unmap outputs or not.

    Returns:
        tuple
    """
    num_imgs = len(img_metas)
    assert len(approx_list) == len(inside_flag_list) == len(
        square_list) == num_imgs
    # anchor number of multi levels
    num_level_squares = [squares.size(0) for squares in square_list[0]]
    # concat all level anchors and flags to a single tensor
    inside_flag_flat_list = []
    approx_flat_list = []
    square_flat_list = []
    for i in range(num_imgs):
        assert len(square_list[i]) == len(inside_flag_list[i])
        inside_flag_flat_list.append(torch.cat(inside_flag_list[i]))
        approx_flat_list.append(torch.cat(approx_list[i]))
        square_flat_list.append(torch.cat(square_list[i]))
    # inside_flag_flat_list = [(num_lvls, num_anchors)]
    # compute targets for each image
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    (all_bbox_anchors, all_bbox_gts, all_bbox_weights, pos_inds_list,
     neg_inds_list) = multi_apply(ga_shape_target_single,
                                  approx_flat_list,
                                  inside_flag_flat_list,
                                  square_flat_list,
                                  gt_bboxes_list,
                                  gt_bboxes_ignore_list,
                                  img_metas,
                                  approxs_per_octave=approxs_per_octave,
                                  cfg=cfg,
                                  sampling=sampling,
                                  unmap_outputs=unmap_outputs)
    # no valid anchors
    if any([bbox_anchors is None for bbox_anchors in all_bbox_anchors]):
        return None
    # sampled anchors of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    # split targets to a list w.r.t. multiple levels
    bbox_anchors_list = images_to_levels(all_bbox_anchors, num_level_squares)
    bbox_gts_list = images_to_levels(all_bbox_gts, num_level_squares)
    bbox_weights_list = images_to_levels(all_bbox_weights, num_level_squares)
    return (bbox_anchors_list, bbox_gts_list, bbox_weights_list, num_total_pos,
            num_total_neg)

```

#### ga_shape_target_single()

```python
def ga_shape_target_single(flat_approxs,
                           inside_flags,
                           flat_squares,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           img_meta,
                           approxs_per_octave,
                           cfg,
                           sampling=True,
                           unmap_outputs=True):
    """Compute guided anchoring targets.

    This function returns sampled anchors and gt bboxes directly
    rather than calculates regression targets.

    Args:
        flat_approxs (Tensor): flat approxs of a single image,
            shape (n, 4)
        inside_flags (Tensor): inside flags of a single image,
            shape (n, ).
        flat_squares (Tensor): flat squares of a single image,
            shape (approxs_per_octave * n, 4)
        gt_bboxes (Tensor): Ground truth bboxes of a single image.
        img_meta (dict): Meta info of a single image.
        approxs_per_octave (int): number of approxs per octave
        cfg (dict): RPN train configs.
        sampling (bool): sampling or not.
        unmap_outputs (bool): unmap outputs or not.

    Returns:
        tuple
    """
    if not inside_flags.any():
        return (None, ) * 6
    # assign gt and sample anchors
    expand_inside_flags = inside_flags[:, None].expand(
        -1, approxs_per_octave).reshape(-1)
    # pick inside approxs
    approxs = flat_approxs[expand_inside_flags, :]
    # pick inside squares
    squares = flat_squares[inside_flags, :]
	# 构建assigner（分配器）为squares分配target
    bbox_assigner = build_assigner(cfg.ga_assigner)
    assign_result = bbox_assigner.assign(approxs, squares, approxs_per_octave,
                                         gt_bboxes, gt_bboxes_ignore)
    if sampling:
        bbox_sampler = build_sampler(cfg.ga_sampler)
    else:
        bbox_sampler = PseudoSampler()
    sampling_result = bbox_sampler.sample(assign_result, squares, gt_bboxes)

    bbox_anchors = torch.zeros_like(squares)
    bbox_gts = torch.zeros_like(squares)
    bbox_weights = torch.zeros_like(squares)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        bbox_anchors[pos_inds, :] = sampling_result.pos_bboxes
        bbox_gts[pos_inds, :] = sampling_result.pos_gt_bboxes
        bbox_weights[pos_inds, :] = 1.0

    # map up to original set of anchors
    if unmap_outputs:
        num_total_anchors = flat_squares.size(0)
        bbox_anchors = unmap(bbox_anchors, num_total_anchors, inside_flags)
        bbox_gts = unmap(bbox_gts, num_total_anchors, inside_flags)
        bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

    return (bbox_anchors, bbox_gts, bbox_weights, pos_inds, neg_inds)
```

在 `approx_max_iou_assigner.py` 中定义了 `ApproxMaxIoUAssigner` ，利用其 `assign` 方法来为得到的 `squares` 分配 `target` ：

```python
class ApproxMaxIoUAssigner(MaxIoUAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates

    def assign(self,
               approxs,
               squares,
               approxs_per_octave,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        """Assign gt to approxs.

        This method assign a gt bbox to each group of approxs (bboxes),
        each group of approxs is represent by a base approx (bbox) and
        will be assigned with -1, 0, or a positive number.
        -1 means don't care, 0 means negative sample,
        positive number is the index (1-based) of assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to -1
        2. use the max IoU of each group of approxs to assign
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            approxs (Tensor): Bounding boxes to be assigned,
        shape(approxs_per_octave*n, 4).
            squares (Tensor): Base Bounding boxes to be assigned,
        shape(n, 4).
            approxs_per_octave (int): number of approxs per octave
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """

        if squares.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or approxs')
        num_squares = squares.size(0)
        num_gts = gt_bboxes.size(0)
        # re-organize anchors by approxs_per_octave x num_squares
        approxs = torch.transpose(
            approxs.view(num_squares, approxs_per_octave, 4), 0,
            1).contiguous().view(-1, 4)
        all_overlaps = bbox_overlaps(approxs, gt_bboxes)
        # 在approxs_per_octave方向上求max, 得到9个预定框里iou最大的那个
        overlaps, _ = all_overlaps.view(approxs_per_octave, num_squares,
                                        num_gts).max(dim=0)
        # 对求出的overlaps再转置成
        overlaps = torch.transpose(overlaps, 0, 1)

        bboxes = squares[:, :4]
        # 对于标记为ignore的gt_bbox计算feature_map上的每个cell上的square bbox与其iof
        # 对于那些iof大于ignore_iof_thr的square, 在overlaps上找到它对应的位置并设为-1
        # (如此一来表示这个cell上的anchor没有匹配上对应的gt_bbox)
        if (self.ignore_iof_thr > 0) and (gt_bboxes_ignore is not None) and (
                gt_bboxes_ignore.numel() > 0):
            # 这里有两种算iof的方法，一种是以bboxes (square)的面积为分母
            # 另一种是以gt_bboxes_ignore的面积为分母
            if self.ignore_wrt_candidates:
                ignore_overlaps = bbox_overlaps(bboxes,
                                                gt_bboxes_ignore,
                                                mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = bbox_overlaps(gt_bboxes_ignore,
                                                bboxes,
                                                mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1
        # 根据算得的iou为shape_pred分配target
        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        return assign_result
```

#### __get_bboxes()__

```python
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   shape_preds,
                   loc_preds,
                   img_metas,
                   cfg,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(shape_preds) == len(
            loc_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        # get guided anchors
        _, guided_anchors, loc_masks = self.get_anchors(
            featmap_sizes,
            shape_preds,
            loc_preds,
            img_metas,
            use_loc_filter=not self.training)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            guided_anchor_list = [
                guided_anchors[img_id][i].detach() for i in range(num_levels)
            ]
            loc_mask_list = [
                loc_masks[img_id][i].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               guided_anchor_list,
                                               loc_mask_list, img_shape,
                                               scale_factor, cfg, rescale)
            result_list.append(proposals)
        return result_list

```



#### get_bboxes_single()

```python
    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          mlvl_masks,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors, mask in zip(cls_scores, bbox_preds,
                                                       mlvl_anchors,
                                                       mlvl_masks):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            # if no location is kept, end.
            if mask.sum() == 0:
                continue
            # reshape scores and bbox_pred
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            # filter scores, bbox_pred w.r.t. mask.
            # anchors are filtered in get_anchors() beforehand.
            scores = scores[mask, :]
            bbox_pred = bbox_pred[mask, :]
            if scores.dim() == 0:
                anchors = anchors.unsqueeze(0)
                scores = scores.unsqueeze(0)
                bbox_pred = bbox_pred.unsqueeze(0)
            # filter anchors, bbox_pred, scores w.r.t. scores
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = delta2bbox(anchors, bbox_pred, self.target_means,
                                self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        # multi class NMS
        det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                cfg.score_thr, cfg.nms,
                                                cfg.max_per_img)
        return det_bboxes, det_labels
```