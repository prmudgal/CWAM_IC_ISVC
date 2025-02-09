# CWAM_IC
Official Pytorch Implementation for "Enhancing Learned Image Compression via Cross Window-based Attention", ISVC, 2024

![IC_Architecture_1](https://github.com/prmudgal/CWAM_IC_ISVC/blob/main/figures/CWAM_IC_Architecture.PNG)
**Figure:** *Our framework*

## Acknowledgement
The framework is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI), we add our model in compressai.models.ours, compressai.models.our_utils. We modify compressai.utils, compressai.zoo, compressai.layers and examples/train.py for usage.
Part of the codes benefit from [The Devil Is in the Details: Window-based Attention for Image Compression](https://github.com/Googolxx/STF), 
[Video Frame Interpolation with Transformer](https://github.com/zhshi0816/Video-Frame-Interpolation-Transformer), and
[Enhanced Invertible Encoding for Learned Image Compression](https://github.com/xyq7/InvCompress).

## Introduction
In this paper, we introduce a feature encoding and decoding module that improves CNNs’ ability to handle complex data representations. This module includes dense blocks and convolutional layers, which strengthen feature propagation and encourage feature reuse effectively. It is integrated in a residual manner for effectiveness. Then, we adopt a modular attention module that can be combined with neural networks to capture correlations among spatially neighboring elements while considering the wider receptive field. This component can be integrated with CNNs to further enhance their performance.

[[Paper](https://arxiv.org/abs/2410.21144)] 

![our_results](https://github.com/prmudgal/CWAM_IC_ISVC/blob/main/figures/Our_results.PNG)
**Figure:** *Our results*

## Installation
As mentioned in [CompressAI](https://github.com/InterDigitalInc/CompressAI), "A C++17 compiler, a recent version of pip (19.0+), and common python packages are also required (see setup.py for the full list)."
```bash
git clone https://github.com/prmudgal/CWAM_IC_ISVC.git
cd CWAM_IC_ISVC/codes/
conda create -n cwamic python=3.7 
conda activate cwamic
pip install -U pip && pip install -e .
conda install -c conda-forge tensorboard
```

### Evaluation
If you want evaluate with pretrained model, please download from [Google drive] and put in ./experiments/

### Pretrained Models
Pretrained models (optimized for MSE) trained from scratch using randomly chose 300k images from the OpenImages dataset.

| Loss | Lambda | Link                                                                                              |
| ---- |--------|---------------------------------------------------------------------------------------------------|
| MSE | 0.0045 | [mse_0045](https://drive.google.com/file/d/1ixytGS7Rg82V9vAxng5OAXnCreygcRbx/view?usp=drive_link)    |
| MSE | 0.00975 | [mse_00975](https://drive.google.com/file/d/1ugMgeTv_b0IDkNKHUpjKjGNXbe6OtVLS/view?usp=drive_link)    |
| MSE | 0.0175 | [mse_0175](https://drive.google.com/file/d/1MqDgnOIvsYrQjhwtMi5a75Z-c8JUFzdN/view?usp=drive_link)    |
| MSE | 0.0483  | [mse_0483](https://drive.google.com/file/d/1vr02qsJqavF5dt3s5zM3nQQOI7kz9AAi/view?usp=drive_link)     |
| MSE | 0.09 | [mse_09](https://drive.google.com/file/d/1s5P4qa0452Dn0tcQN-gfJq97CeOuJzqy/view?usp=drive_link) |
| MSE | 0.14 | [mse_14](https://drive.google.com/file/d/1IF-16-2LMP6AP0_195hn31kFbkRENKr-/view?usp=drive_link)    |
| MSSSIM | 873 | [msssim_873]() |
| MSSSIM | 1664  | [msssim_1664]()  |
| MSSSIM | 3184  | [msssim_3184]()     |
| MSSSIM | 6050 | [msssim_6050]() |


Further trained models shall be uploaded soon!!!!

Some evaluation dataset can be downloaded from 
[kodak dataset](http://r0k.us/graphics/kodak/), [CLIC](http://challenge.compression.cc/tasks/)

Note that as mentioned in original [CompressAI](https://github.com/InterDigitalInc/CompressAI), "Inference on GPU is not recommended for the autoregressive models (the entropy coder is run sequentially on CPU)." So for inference of our model, please run on CPU.
```bash
python -m compressai.utils.eval_model checkpoint $eval_data_dir -a invcompress -exp $exp_name -s $save_dir
```


An example: to evaluate model of quality 1 optimized with mse on kodak dataset. 
```bash
python -m compressai.utils.eval_model checkpoint ../data/kodak -a invcompress -exp exp_01_mse_q1 -s ../results/exp_01
```

If you want to evaluate your trained model on own data, please run update before evaluation. An example:
```bash
python -m compressai.utils.update_model -exp $exp_name -a invcompress
python -m compressai.utils.eval_model checkpoint $eval_data_dir -a invcompress -exp $exp_name -s $save_dir
```

### Train
We use the training dataset processed in the [repo](https://github.com/liujiaheng/CompressionData). We further preprocess with /codes/scripts/flicker_process.py
Training setting is detailed in the paper. You can also use your own data for training. 

```bash
python examples/train.py -exp $exp_name -m cwam -d $train_data_dir --epochs $epoch_num -lr $lr --batch-size $batch_size --cuda --gpu_id $gpu_id --lambda $lamvda --metrics $metric --save 
```

An example: to train model of quality 1 optimized with mse metric.
```bash
python examples/train.py -exp exp_01_mse_q1 -m cwam -d ../data/flicker --epochs 600 -lr 1e-4 --batch-size 8 --cuda --gpu_id 0 --lambda 0.00475 --metrics mse --save 

```



Other usage please refer to the original library [CompressAI](https://github.com/InterDigitalInc/CompressAI)




## Citation
If you find this work useful for your research, please cite:
```
@inproceedings{mudgal2024enhancing,
  title={Enhancing Learned Image Compression via Cross Window-Based Attention},
  author={Mudgal, Priyanka and Liu, Feng},
  booktitle={International Symposium on Visual Computing},
  pages={410--423},
  year={2024},
  organization={Springer}
}
```

## Contact
Feel free to contact us if there is any question. (pmudgal@pdx.edu)
