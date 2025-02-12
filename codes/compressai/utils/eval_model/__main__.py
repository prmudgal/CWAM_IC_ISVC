# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Evaluate an end-to-end compression model on an image dataset.
"""
import argparse
import json
import math
import os
import sys
import time

from collections import defaultdict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms
from deepspeed.profiling.flops_profiler import get_model_profile

import compressai

from compressai.zoo import models as pretrained_models
from compressai.zoo.image import model_architectures as architectures

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def collect_images(rootpath: str) -> List[str]:
    return [
        os.path.join(rootpath, f)
        for f in os.listdir(rootpath)
        if os.path.splitext(f)[-1].lower() in IMG_EXTENSIONS
    ]


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")

    # test_transforms = transforms.Compose(
    #     [transforms.CenterCrop(256), transforms.ToTensor()]
    # )
    # return test_transforms(img)

    return transforms.ToTensor()(img)

import matplotlib.pyplot as plt
from PIL import Image
@torch.no_grad()
def inference(f, model, x, savedir = "", idx = 1):
    x = x.unsqueeze(0)

    h, w = x.size(2), x.size(3)
    p = 64  # maximum 6 strides of 2
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    
    start = time.time()
    out_enc = model.compress(x_padded)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(
        out_dec["x_hat"], (-padding_left, -padding_right, -padding_top, -padding_bottom)
    )

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
    if savedir != "":
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        cur_psnr = psnr(x, out_dec["x_hat"])
        cur_ssim = ms_ssim(x, out_dec["x_hat"], data_range=1.0).item()
        tran1 = transforms.ToPILImage()

        cur_img = tran1(out_dec["x_hat"][0])
        cur_img.save(os.path.join(savedir, f+".png"))
        
        # print("new file", os.path.join(savedir, f+".png"))
        fname = os.path.join(savedir, f+".png")
        image = Image.open(fname).convert("L")
        arr = np.asarray(image)
        print('{:02d}'.format(idx)+"_"+'{:.2f}'.format(cur_psnr)+"_"+'{:.3f}'.format(bpp)+"_"+'{:.3f}'.format(cur_ssim))
        
    return {
       
        "psnr": psnr(x, out_dec["x_hat"]),
        "ms-ssim": ms_ssim(x, out_dec["x_hat"], data_range=1.0).item(),
        "bpp": bpp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    },  {"savedir": f}


@torch.no_grad()
def inference_entropy_estimation(model, x):
    x = x.unsqueeze(0)

    start = time.time()
    out_net = model.forward(x)
    # print(out_net['x_hat'][0,0,:5,:5])

    elapsed_time = time.time() - start

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )

    return {
        "psnr": psnr(x, out_net["x_hat"]),
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,  # broad estimation
        "decoding_time": elapsed_time / 2.0,
    }


def load_pretrained(model: str, metric: str, quality: int) -> nn.Module:
    return pretrained_models[model](
        quality=quality, metric=metric, pretrained=True
    ).eval()


def load_checkpoint(arch: str, checkpoint_path: str) -> nn.Module:
    print(checkpoint_path)
    return architectures[arch].from_state_dict(torch.load(checkpoint_path)).eval()

from pathlib import Path
def detailed_bpp(out):
    size = out['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    for name, values in out['likelihoods'].items():
        bpp_val = torch.log(values).sum() / (-math.log(2) * num_pixels)
        print(f'- "{name}" latent bit-rate: {bpp_val:.3f} bpp')

@torch.no_grad()
def inference_entropy_estimation_1(f, model, x, savedir = "", idx = 1):
    x = x.unsqueeze(0)

    start = time.time()
    out_net = model.forward(x)

    elapsed_time = time.time() - start

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )
    detailed_bpp(out_net)
    with torch.no_grad():
        y = model.g_a_func(x)
        y_hat, y_likelihoods = model.entropy_bottleneck(y)
        # print(y.size(), y_likelihoods.size())

    return out_net

def eval_model(model, filepaths, entropy_estimation=False, half=False, savedir = ""):
    device = next(model.parameters()).device
    metrics = defaultdict(float)
    for idx, f in enumerate(sorted(filepaths)):
        x = read_image(f).to(device)
        if not entropy_estimation:
            print(Path(f).stem)
            print('evaluating index', idx)
            if half:
                model = model.half()
                x = x.half()
            rv, dir = inference(Path(f).stem, model, x, savedir, idx)
                
        else:
            rv = inference_entropy_estimation(model, x)

        inference_entropy_estimation_1(Path(f).stem, model, x, savedir, idx)
        print('savedir', dir['savedir'])
        print('bpp', rv['bpp'])
        print('psnr', rv['psnr'])
        print('ms-ssim', rv['ms-ssim'])
        print()
        for k, v in rv.items():
            metrics[k] += v
    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    return metrics


def setup_args():
    parent_parser = argparse.ArgumentParser(
        add_help=False,
    )

    # Common options.
    parent_parser.add_argument("dataset", type=str, help="dataset path")
    parent_parser.add_argument(
        "-a",
        "--arch",
        type=str,
        choices=pretrained_models.keys(),
        help="model architecture",
        required=True,
    )
    parent_parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parent_parser.add_argument(
        "--cuda",
        action="store_true",
        help="enable CUDA",
    )
    parent_parser.add_argument(
        "--half",
        action="store_true",
        help="convert model to half floating point (fp16)",
    )
    parent_parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parent_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )
    parent_parser.add_argument(
        "-s",
        "--savedir",
        type=str,
        default="",
    )
    parent_parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser = argparse.ArgumentParser(
        description="Evaluate a model on an image dataset.", add_help=True
    )
    subparsers = parser.add_subparsers(
        help="model source", dest="source")#, required=True
   # )

    # Options for pretrained models
    pretrained_parser = subparsers.add_parser("pretrained", parents=[parent_parser])
    pretrained_parser.add_argument(
        "-m",
        "--metric",
        type=str,
        choices=["mse", "ms-ssim"],
        default="mse",
        help="metric trained against (default: %(default)s)",
    )
    pretrained_parser.add_argument(
        "-q",
        "--quality",
        dest="qualities",
        nargs="+",
        type=int,
        default=(1,),
    )

    checkpoint_parser = subparsers.add_parser("checkpoint", parents=[parent_parser])
    
    checkpoint_parser.add_argument("-exp", "--experiment", type=str, required=True, help="Experiment name")

    return parser


def main(argv):
    args = setup_args().parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    filepaths = collect_images(args.dataset)
    if len(filepaths) == 0:
        print("No images found in directory.")
        sys.exit(1)

    compressai.set_entropy_coder(args.entropy_coder)

    if args.source == "pretrained":
        runs = sorted(args.qualities)
        opts = (args.arch, args.metric)
        load_func = load_pretrained
        log_fmt = "\rEvaluating {0} | {run:d}"
    elif args.source == "checkpoint":
        # runs = args.paths
        checkpoint_updated_dir = os.path.join('../experiments', args.experiment, 'checkpoints')
        checkpoint_updated = os.path.join(checkpoint_updated_dir, os.listdir(checkpoint_updated_dir)[0])
        print(checkpoint_updated)
        runs = [checkpoint_updated]
        opts = (args.arch,)
        load_func = load_checkpoint
        log_fmt = "\rEvaluating {run:s}"

    results = defaultdict(list)
    for run in runs:
        if args.verbose:
            sys.stderr.write(log_fmt.format(*opts, run=run))
            sys.stderr.flush()
        model = load_func(*opts, run)
        if args.cuda and torch.cuda.is_available():
            model = model.to("cuda")
        metrics = eval_model(model, filepaths, args.entropy_estimation, args.half, args.savedir)
        for k, v in metrics.items():
            results[k].append(v)

    if args.verbose:
        sys.stderr.write("\n")
        sys.stderr.flush()

    description = (
        "entropy estimation" if args.entropy_estimation else args.entropy_coder
    )
    output = {
        "name": args.arch,
        "description": f"Inference ({description})",
        "results": results,
    }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
