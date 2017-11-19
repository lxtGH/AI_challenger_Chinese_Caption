# chinese_im2text.pytorch
# Notice 
This projetc is based on gujiuxiang's [chinese_im2text.pytorch](https://github.com/gujiuxiang/chinese_im2text.pytorch).
But there are some bugs in his repository, this project tries to fix them.
His project is based on ruotian's [neuraltalk2.pytorch](https://github.com/ruotianluo/neuraltalk2.pytorch).
Most of codes are from that amazing projects
## Requirements

### Software enviroment
Python 2.7 (because there is no [coco-caption](https://github.com/tylin/coco-caption) version for python 3), PyTorch 0.2 (along with torchvision). 

### Dataset
You need to download pretrained resnet model for both training and evaluation, and you need to register the ai challenger, and then download the training and validation dataset.

## Pretrained models

TODO

## Train your own network on AI Challenger
### Download AI Challenger dataset and preprocessing
First, download the 图像中文描述数据库 from [link](https://challenger.ai/datasets). We need training images (210,000) and val images (30,000). You should put the `ai_challenger_caption_train_20170902/` and `ai_challenger_caption_train_20170902/` in the same directory, denoted as `$IMAGE_ROOT`. Once we have these, we can now invoke the `json_preprocess.py` and `prepro_ai_challenger.py` script, which will read all of this in and create a dataset (two feature folders, a hdf5 label file and a json file).

```bash
$ python scripts/json_preprocess.py
$ python prepro_ai_challenger.py
```

`json_preprocess.py` will first transform the AI challenger Image Caption_json to mscoco json format. Then map all words that occur <= 5 times to a special `UNK` token, and create a vocabulary for all the remaining words. The image information and vocabulary are dumped into `coco_ai_challenger_raw.json`.
This file also generates the `coco_val_caption_validation_annotations_20170910.json` for evaluation metric calcuation, you can find the json files in the following folder:
```bash
# For metric calcuation
chinese_im2text.pytorch/caption_eval/data/coco_val_caption_validation_annotations_20170910.json
# For preprocessing
chinese_im2text.pytorch/caption_eval/data/coco_caption_validation_annotations_20170910.json
```

`prepro_ai_challenger.py` extract the resnet101 features (both fc feature and last conv feature) of each image. The features are saved in `coco_ai_challenger_talk_fc.h5` and `coco_ai_challenger_talk_att.h5`, and resulting files are about 359GB.


### Start training
The following training procedure are adopted from ruotian's project, and if you need REINFORCEMENT-based approach, you can clone from [here](https://github.com/ruotianluo/self-critical.pytorch). For ai challenger, they provide large number of validation size, you can set `--val_images_use` to a bigger size.

```bash
$ python train.py --id st --caption_model show_tell --input_json data/cocotalk.json --input_fc_h5 data/coco_ai_challenger_talk_fc.h5 --input_att_h5 data/coco_ai_challenger_talk_att.h5 --input_label_h5 data/coco_ai_challenger_talk_label.h5 --batch_size 10 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --checkpoint_path log_st --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 25
```

The train script will dump checkpoints into the folder specified by `--checkpoint_path` (default = `save/`). We only save the best-performing checkpoint on validation and the latest checkpoint to save disk space.

To resume training, you can specify `--start_from` option to be the path saving `infos.pkl` and `model.pth` (usually you could just set `--start_from` and `--checkpoint_path` to be the same).

If you have tensorflow, the loss histories are automatically dumped into `--checkpoint_path`, and can be visualized using tensorboard.

The current command use scheduled sampling, you can also set scheduled_sampling_start to -1 to turn off scheduled sampling.

If you'd like to evaluate BLEU/METEOR/CIDEr scores during training in addition to validation cross entropy loss, use `--language_eval 1` option, but don't forget to download the [coco-caption code](https://github.com/tylin/coco-caption) into `coco-caption` directory.

For more options, see `opts.py`. 

Currently, the training loss of my baseline model is as follows:
![](./vis/training_log_mine.png)

And I set the beam size to 5 during testing, and some predicted descriptions are as follows (image xxx, xxx is the image ID):
```bash
..
Beam size: 5, image 2550: 一个穿着裙子的女人走在道路上
Beam size: 5, image 2551: 房间里有一个穿着白色上衣的女人在给一个
Beam size: 5, image 2596: 一个穿着运动服的男人在运动场上奔跑
Beam size: 5, image 2599: 一个穿着裙子的女人站在广告牌前的红毯上
...
```
After 18,000 steps, I evaluated my model on the 1,0000 val images, and can achieve the following results:
```
Bleu_1: 0.754
Bleu_2: 0.630
Bleu_3: 0.522
Bleu_4: 0.432
METEOR: 0.369
ROUGE_L: 0.615
CIDEr: 1.234
```
However, when I try to upload my testing results (the test json file can be found in data folder), the online server always failed, and they did not tell me why! WHAT A MESS!

## Generate image captions

### Evaluate on raw images
Now place all your images of interest into a folder, e.g. `blah`, and run
the eval script:

```bash
$ python eval.py --model model.pth --infos_path infos.pkl --image_folder blah --num_images 10
```

This tells the `eval` script to run up to 10 images from the given folder. If you have a big GPU you can speed up the evaluation by increasing `batch_size`. Use `--num_images -1` to process all images. The eval script will create an `vis.json` file inside the `vis` folder, which can then be visualized with the provided HTML interface:

```bash
$ cd vis
$ python -m SimpleHTTPServer
```

Now visit `localhost:8000` in your browser and you should see your predicted captions.

### Evaluate on validation split

For evaluation, you can use the offical evaluation tool provide by AIChallenger. And I modified their code, and you can find it in
```bash
caption_eval
```
The GT annotations are also provided.

```bash
$ python eval.py --dump_images 0 --num_images 5000 --model model.pth --infos_path infos.pkl --language_eval 1 
```

The defualt split to evaluate is test. The default inference method is greedy decoding (`--sample_max 1`), to sample from the posterior, set `--sample_max 0`.

**Beam Search**. Beam search can increase the performance of the search for greedy decoding sequence by ~5%. However, this is a little more expensive. To turn on the beam search, use `--beam_size N`, N should be greater than 1 (we set beam size to 5 in our eval).

## Acknowledgements

Thanks the original [neuraltalk2](https://github.com/karpathy/neuraltalk2), and the pytorch-based [neuraltalk2.pytorch](https://github.com/ruotianluo/neuraltalk2.pytorch) and awesome PyTorch team.

## Paper

1. Jiuxiang Gu, Gang Wang, Jianfei Cai, and Tsuhan Chen. ["An Empirical Study of Language CNN for Image Captioning."](https://arxiv.org/pdf/1612.07086.pdf) ICCV, 2017.
```
@article{gu2016recurrent,
  title={An Empirical Study of Language CNN for Image Captioning},
  author={Gu, Jiuxiang and Wang, Gang and Cai, Jianfei and Chen, Tsuhan},
  journal={ICCV},
  year={2017}
}
```
2. Jiuxiang Gu, Jianfei cai, Gang Wang, and Tsuhan Chen. ["Stack-Captioning: Coarse-to-Fine Learning for Image Captioning."](https://arxiv.org/abs/1709.03376) arXiv preprint arXiv:1709.03376 (2017).
```
@article{gu2017stack_cap,
  title={Stack-Captioning: Coarse-to-Fine Learning for Image Captioning},
  author={Gu, Jiuxiang and Cai, Jianfei and Wang, Gang and Chen, Tsuhan},
  journal={arXiv preprint arXiv:1709.03376},
  year={2017}
}
```
