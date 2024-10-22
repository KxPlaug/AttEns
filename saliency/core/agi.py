import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
import numpy as np
from matplotlib import pyplot as plt

from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets
from PIL import Image

import json


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


def pre_processing(obs, torch_device):
    # rescale imagenet, we do mornalization in the network, instead of preprocessing
    # mean = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
    # std = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
    obs = obs / 255
    # obs = (obs - mean) / std
    obs = np.transpose(obs, (2, 0, 1))
    obs = np.expand_dims(obs, 0)
    obs = np.array(obs)
    # if cuda:
    #     torch_device = torch.device('cuda:0')
    # else:
    #     torch_device = torch.device('cpu')
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=torch_device)
    return obs_tensor

# %%


def fgsm_step(image, epsilon, data_grad_adv, data_grad_lab):
    # generate the perturbed image based on steepest descent
    grad_lab_norm = torch.norm(data_grad_lab, p=2)
    delta = epsilon * data_grad_adv.sign()

    # + delta because we are ascending
    perturbed_image = image + delta
    perturbed_rect = torch.clamp(perturbed_image, min=0, max=1)
    delta = perturbed_rect - image
    delta = - data_grad_lab * delta
    return perturbed_rect, delta
    # return perturbed_image, delta


def pgd_step(image, epsilon, model, init_pred, targeted, max_iter):
    """target here is the targeted class to be perturbed to"""
    perturbed_image = image.clone()
    # c_delta = 0  # cumulative delta
    leave_index = np.arange(image.shape[0]).tolist()
    for i in range(max_iter):
        # requires grads
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        # get the index of the max log-probability
        # pred = output.max(1, keepdim=True)[1]
        pred = output.argmax(-1)
        # if attack is successful, then break
        for j in leave_index:
            if pred[j] == targeted[j]:
                leave_index.remove(j)
        if len(leave_index) == 0:
            break
        # select the false class label
        output = F.softmax(output, dim=1)
        # loss = output[0, targeted.item()]
        loss = output[:, targeted].sum()

        model.zero_grad()
        loss.backward(retain_graph=True)
        data_grad_adv = perturbed_image.grad.data.detach().clone()

        # loss_lab = output[0, init_pred.item()]
        loss_lab = output[:, init_pred].sum()
        model.zero_grad()
        perturbed_image.grad.zero_()
        loss_lab.backward()
        data_grad_lab = perturbed_image.grad.data.detach().clone()
        perturbed_image, delta = fgsm_step(
            image, epsilon, data_grad_adv, data_grad_lab)
        # c_delta += delta
        if i == 0:
            c_delta = delta
        else:
            c_delta[leave_index] += delta[leave_index]

    return c_delta, perturbed_image


def pgd_step_kl_new(image, epsilon, model, init_pred, targeted, max_iter):
    """target here is the targeted class to be perturbed to"""
    perturbed_image = image.clone()
    # c_delta = 0  # cumulative delta
    leave_index = np.arange(image.shape[0]).tolist()
    for i in range(max_iter):
        # requires grads
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        # get the index of the max log-probability
        # pred = output.max(1, keepdim=True)[1]
        pred = output.argmax(-1)
        # if attack is successful, then break
        for j in leave_index:
            if pred[j] == targeted[j]:
                leave_index.remove(j)
        if len(leave_index) == 0:
            break
        # select the false class label
        output = F.softmax(output, dim=1)
        # loss = output[0, targeted.item()]
        loss = output[:, targeted].sum()

        model.zero_grad()
        loss.backward(retain_graph=True)
        data_grad_adv = perturbed_image.grad.data.detach().clone()

        # loss_lab = output[0, init_pred.item()]
        # loss_lab = output[:, init_pred].sum()
        loss_lab = F.kl_div(output.log(), torch.ones_like(output) / 1000, reduction="batchmean")
        model.zero_grad()
        perturbed_image.grad.zero_()
        loss_lab.backward()
        data_grad_lab = perturbed_image.grad.data.detach().clone()
        perturbed_image, delta = fgsm_step(
            image, epsilon, data_grad_adv, data_grad_lab)
        # c_delta += delta
        if i == 0:
            c_delta = delta
        else:
            c_delta[leave_index] += delta[leave_index]

    return c_delta, perturbed_image




def pgd_step_kl(image, epsilon, model, init_pred, max_iter):
    """target here is the targeted class to be perturbed to"""
    perturbed_image = image.clone()
    # c_delta = 0  # cumulative delta
    leave_index = np.arange(image.shape[0]).tolist()
    for i in range(max_iter):
        # requires grads
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        # get the index of the max log-probability
        # pred = output.max(1, keepdim=True)[1]
        pred = output.argmax(-1)
        # if attack is successful, then break
        # select the false class label
        output = F.softmax(output, dim=1)
        # loss_lab = output[0, init_pred.item()]
        loss_lab = F.kl_div(output.log(), torch.ones_like(output) / 1000, reduction="batchmean")
        model.zero_grad()
        
        loss_lab.backward()
        data_grad_lab = perturbed_image.grad.data.detach().clone()
        perturbed_image.grad.zero_()
        perturbed_image, delta = fgsm_step_gai(
            image, epsilon, data_grad_lab)
        # c_delta += delta
        if i == 0:
            c_delta = delta
        else:
            c_delta[leave_index] += delta[leave_index]

    return c_delta, perturbed_image


def pgd_step_gai(image, epsilon, model, init_pred, targeted, max_iter):
    """target here is the targeted class to be perturbed to"""
    perturbed_image = image.clone()
    # c_delta = 0  # cumulative delta
    leave_index = np.arange(image.shape[0]).tolist()
    for i in range(max_iter):
        # requires grads
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        # get the index of the max log-probability
        # pred = output.max(1, keepdim=True)[1]
        pred = output.argmax(-1)
        # if attack is successful, then break
        # for j in leave_index:
        #     if pred[j] == targeted[j]:
        #         leave_index.remove(j)
        if len(leave_index) == 0:
            break
        # select the false class label
        output = F.softmax(output, dim=1)
        # loss = output[0, targeted.item()]
        loss = output[:, targeted].sum()

        model.zero_grad()
        loss.backward(retain_graph=True)
        data_grad_adv = perturbed_image.grad.data.detach().clone()

        # loss_lab = output[0, init_pred.item()]
        loss_lab = output[:, init_pred].sum()
        model.zero_grad()
        perturbed_image.grad.zero_()
        loss_lab.backward()
        data_grad_lab = perturbed_image.grad.data.detach().clone()
        perturbed_image, delta = fgsm_step(
            image, epsilon, data_grad_adv, data_grad_lab)
        # c_delta += delta
        if i == 0:
            c_delta = delta
        else:
            c_delta[leave_index] += delta[leave_index]

    return c_delta, perturbed_image

def pgd_step_gai2(image, epsilon, model, init_pred, targeted, max_iter):
    """target here is the targeted class to be perturbed to"""
    perturbed_image = image.clone()
    # c_delta = 0  # cumulative delta
    leave_index = np.arange(image.shape[0]).tolist()
    for i in range(max_iter):
        # requires grads
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        # get the index of the max log-probability
        # pred = output.max(1, keepdim=True)[1]
        pred = output.argmax(-1)
        # if attack is successful, then break
        for j in leave_index:
            if pred[j] != init_pred[j]:
                leave_index.remove(j)
        if len(leave_index) == 0:
            break
        # select the false class label
        output = F.softmax(output, dim=1)
        # loss = output[0, targeted.item()]
        loss = output[:, targeted].sum()

        model.zero_grad()
        loss.backward(retain_graph=True)
        data_grad_adv = perturbed_image.grad.data.detach().clone()

        # loss_lab = output[0, init_pred.item()]
        loss_lab = output[:, init_pred].sum()
        model.zero_grad()
        perturbed_image.grad.zero_()
        loss_lab.backward()
        data_grad_lab = perturbed_image.grad.data.detach().clone()
        perturbed_image, delta = fgsm_step(
            image, epsilon, data_grad_adv, data_grad_lab)
        # c_delta += delta
        if i == 0:
            c_delta = delta
        else:
            c_delta[leave_index] += delta[leave_index]

    return c_delta, perturbed_image


def pgd_step_loss_balance(image, epsilon, model, init_pred, targeted, max_iter):
    """target here is the targeted class to be perturbed to"""
    perturbed_image = image.clone()
    # c_delta = 0  # cumulative delta
    leave_index = np.arange(image.shape[0]).tolist()
    start_losses = None
    final_losses = torch.zeros(image.shape[0])
    for i in range(max_iter):
        # requires grads
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        # get the index of the max log-probability
        # pred = output.max(1, keepdim=True)[1]
        pred = output.argmax(-1)
        # if attack is successful, then break
        for j in leave_index:
            if pred[j] == targeted[j]:
                leave_index.remove(j)
        if len(leave_index) == 0:
            break
        # select the false class label
        output = F.softmax(output, dim=1)
        # loss = output[0, targeted.item()]
        loss = output[:, targeted].sum()
        if i == 0:
            # start_losses = output[:, targeted].detach().clone()
            # final_losses = output[:, targeted].detach().clone()
            start_losses = torch.gather(output, 1, targeted.unsqueeze(1)).squeeze()
            final_losses = torch.gather(output, 1, targeted.unsqueeze(1)).squeeze()
        else:
            # final_losses[leave_index] = output[leave_index, targeted[leave_index]]
            final_losses[leave_index] = torch.gather(output[leave_index], 1, targeted[leave_index].unsqueeze(1)).squeeze()

        model.zero_grad()
        loss.backward(retain_graph=True)
        data_grad_adv = perturbed_image.grad.data.detach().clone()

        # loss_lab = output[0, init_pred.item()]
        loss_lab = output[:, init_pred].sum()
        model.zero_grad()
        perturbed_image.grad.zero_()
        loss_lab.backward()
        data_grad_lab = perturbed_image.grad.data.detach().clone()
        perturbed_image, delta = fgsm_step(
            image, epsilon, data_grad_adv, data_grad_lab)
        # c_delta += delta
        if i == 0:
            c_delta = delta
        else:
            c_delta[leave_index] += delta[leave_index]
    loss_diff = abs(final_losses - start_losses)
    # c_delta = c_delta / loss_diff
    c_delta = torch.stack([c_delta[i] / loss_diff[i] for i in range(len(c_delta))])
    return c_delta, perturbed_image


def fgsm_step_gai(image, epsilon, data_grad_lab):
    delta = - epsilon * data_grad_lab.sign()

    # + delta because we are ascending
    perturbed_image = image + delta
    perturbed_rect = torch.clamp(perturbed_image, min=0, max=1)
    delta = perturbed_rect - image
    delta = -data_grad_lab * delta
    return perturbed_rect, delta
    # return perturbed_image, delta

def pgd_step_gai3(image, epsilon, model, init_pred, max_iter):
    """target here is the targeted class to be perturbed to"""
    perturbed_image = image.clone()
    # c_delta = 0  # cumulative delta
    leave_index = np.arange(image.shape[0]).tolist()
    for i in range(max_iter):
        # requires grads
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        # get the index of the max log-probability
        # pred = output.max(1, keepdim=True)[1]
        pred = output.argmax(-1)
        # if attack is successful, then break
        # select the false class label
        output = F.softmax(output, dim=1)
        # loss_lab = output[0, init_pred.item()]
        loss_lab = output[:, init_pred].sum()
        model.zero_grad()
        
        loss_lab.backward()
        data_grad_lab = perturbed_image.grad.data.detach().clone()
        perturbed_image.grad.zero_()
        perturbed_image, delta = fgsm_step_gai(
            image, epsilon, data_grad_lab)
        # c_delta += delta
        if i == 0:
            c_delta = delta
        else:
            c_delta[leave_index] += delta[leave_index]

    return c_delta, perturbed_image



def fgsm_step_gai2(model,image,init_pred, epsilon, data_grad_adv, data_grad_lab):
    # generate the perturbed image based on steepest descent
    grad_lab_norm = torch.norm(data_grad_lab, p=2)
    delta = epsilon * data_grad_adv.sign()

    # + delta because we are ascending
    perturbed_image = image + delta
    perturbed_rect = torch.clamp(perturbed_image, min=0, max=1)
    perturbed_rect = perturbed_rect.detach().requires_grad_(True)
    output = model(perturbed_rect)
    loss = output[:, init_pred].sum()
    grad = torch.autograd.grad(loss, perturbed_rect)[0]
    delta = perturbed_rect - image
    data_grad_lab = (grad + data_grad_lab) / 2
    delta = - data_grad_lab * delta
    delta = delta.detach()
    return perturbed_rect, delta
    # return perturbed_image, delta

def pgd_step_gai4(image, epsilon, model, init_pred, targeted, max_iter):
    """target here is the targeted class to be perturbed to"""
    perturbed_image = image.clone()
    # c_delta = 0  # cumulative delta
    leave_index = np.arange(image.shape[0]).tolist()
    for i in range(max_iter):
        # requires grads
        perturbed_image.requires_grad = True
        output = model(perturbed_image)
        # get the index of the max log-probability
        # pred = output.max(1, keepdim=True)[1]
        pred = output.argmax(-1)
        # if attack is successful, then break
        for j in leave_index:
            if pred[j] == targeted[j]:
                leave_index.remove(j)
        if len(leave_index) == 0:
            break
        # select the false class label
        output = F.softmax(output, dim=1)
        # loss = output[0, targeted.item()]
        loss = output[:, targeted].sum()

        model.zero_grad()
        loss.backward(retain_graph=True)
        data_grad_adv = perturbed_image.grad.data.detach().clone()

        # loss_lab = output[0, init_pred.item()]
        loss_lab = output[:, init_pred].sum()
        model.zero_grad()
        perturbed_image.grad.zero_()
        loss_lab.backward()
        data_grad_lab = perturbed_image.grad.data.detach().clone()
        perturbed_image, delta = fgsm_step_gai2(model,
            image,init_pred, epsilon, data_grad_adv, data_grad_lab)
        # c_delta += delta
        if i == 0:
            c_delta = delta
        else:
            c_delta[leave_index] += delta[leave_index]

    return c_delta, perturbed_image

# Dummy class to store arguments
class Dummy():
    pass


# Function that opens image from disk, normalizes it and converts to tensor
read_tensor = transforms.Compose([
    lambda x: Image.open(x),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    lambda x: torch.unsqueeze(x, 0)
])


# Plots image from tensor
def tensor_imshow(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, **kwargs)
    if title is not None:
        plt.title(title)


# Given label number returns class name
def get_class_name(c):
    labels = json.load(open("imagenet_class_index.json"))
    # labels = np.loadtxt('synset_words.txt', str, delimiter='\t')
    return labels[str(c)][1]


# Image preprocessing function
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Normalization for ImageNet
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# Sampler for pytorch loader. Given range r loader will only
# return dataset[r] instead of whole dataset.
class RangeSampler(Sampler):
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        return iter(self.r)

    def __len__(self):
        return len(self.r)
