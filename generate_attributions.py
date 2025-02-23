import torch.nn as nn
from torchvision import transforms as T
import torch
import numpy as np
import argparse
import torch
import random
from torch.utils.data import TensorDataset, DataLoader
import os
from torchvision.models import maxvit_t,vit_b_16,inception_v3,resnet50,vgg16,mobilenet_v2
from saliency.saliency_zoo import *
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(3407)


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='inception_v3')
parser.add_argument('--attr_method', type=str, default='agi')
parser.add_argument("--dataset",type=str,default="isa",choices=["attack","isa"])
parser.add_argument("--image_num",type=int,default=1000)
parser.add_argument("--single_softmax",action="store_true")
args = parser.parse_args()

perfix = f"attributions_{args.dataset}"
os.makedirs(perfix,exist_ok=True)

# 自带softmax的归因方法
attr_methods_with_softmax = ["mfaba","agi","ampe","isa"]

if args.attr_method == "deeplift":
    from resnet_mod import resnet50
    from vgg16_mod import vgg16


if __name__ == "__main__":
    import os
    if not os.path.exists(f"{perfix}/{args.model}_{args.attr_method}_attributions{'_singlesoftmax' if args.single_softmax else ''}.npy"):
        attr_method = eval(args.attr_method)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.dataset == "attack":
            img_batch = torch.load("img_batch.pt")[0:args.image_num]
            target_batch = torch.load("label_batch.pt")[0:args.image_num]
        else:
            img_batch = torch.load("ISA_dataset/img_batch.pt")[0:args.image_num]
            target_batch = torch.load("ISA_dataset/label_batch.pt")[0:args.image_num]

        dataset = TensorDataset(img_batch, target_batch)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        model = eval(f"{args.model}(pretrained=True).eval().to(device)")
        sm = nn.Softmax(dim=-1)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        norm_layer = T.Normalize(mean, std)
        starts_with = False
        for attr_name in attr_methods_with_softmax:
            if args.attr_method.startswith(attr_name):
                starts_with = True
                break
        if starts_with:
            model = nn.Sequential(norm_layer, model).to(device) if args.single_softmax else nn.Sequential(norm_layer, model, sm).to(device)
        else:
            model = nn.Sequential(norm_layer, model, sm).to(device) if args.single_softmax else nn.Sequential(norm_layer, model, sm,nn.Softmax(-1)).to(device)
        # if args.attr_method in ['fast_ig','guided_ig','guided_ig_gai','big']:
        model = nn.DataParallel(model, device_ids=[0,1])
        if args.attr_method.startswith('fast_ig') or args.attr_method.startswith('guided_ig') or args.attr_method.startswith('big'):
            batch_size = 1
        elif args.attr_method == "ig_gai":
            batch_size = 2
        elif args.attr_method.startswith('ig') or args.attr_method.startswith('ampe') or args.attr_method.startswith('eg') or args.attr_method.startswith("sg") or args.attr_method.startswith("deeplift"):
            batch_size = 4
        elif args.attr_method == "mfaba_alpha_rgb":
            batch_size = 32
        elif args.attr_method.startswith('agi') or args.attr_method.startswith('negflux') or args.attr_method.startswith('mfaba') or args.attr_method.startswith('isa') or args.attr_method.startswith('sm'):
            batch_size = 64
        elif args.attr_method == 'saliencymap':
            batch_size = 128
        if args.model == "maxvit_t" or args.model == "vit_b_16":
            if args.attr_method != "eg":
                if batch_size > 2:
                    batch_size = batch_size // 2
        attributions = []
        for i in tqdm(range(0, len(img_batch), batch_size)):
            img = img_batch[i:i+batch_size].to(device)
            target = target_batch[i:i+batch_size].to(device)
            if args.attr_method in ["ig_gai","ig_gai_kl","guided_ig_gai"]:
                attribution = attr_method(model, img, target, args.model)
            elif args.attr_method == "eg":
                attribution = attr_method(model, dataloader, img, target)
            else:
                attribution = attr_method(model, img, target)
            attributions.append(attribution)
        attributions = np.concatenate(attributions, axis=0)
        np.save(f"{perfix}/{args.model}_{args.attr_method}_attributions{'_singlesoftmax' if args.single_softmax else ''}.npy", attributions)