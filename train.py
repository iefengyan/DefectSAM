import argparse
import os
import cv2
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import models
import utils
import torch.nn as nn
import torch
from data_loader import get_loader
from torch.autograd import Variable
from sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from models import  models
device = "cuda"
def eval_psnr(test_image_root, test_gt_root, model):
    model.eval()
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()

    img_transform = transforms.Compose([
        transforms.Resize((1024,1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    images = [test_image_root + f for f in os.listdir(test_image_root)]
    gts = [test_gt_root + p for p in os.listdir(test_gt_root)]
    images = sorted(images)
    gts = sorted(gts)

    for index in range(len(images)):
        ori_image=Image.open(images[index]).convert("RGB")

        gt = cv2.imread(gts[index], cv2.IMREAD_GRAYSCALE)
        image = img_transform(ori_image).unsqueeze(0).cuda()
        H, W = gt.shape
        # low_inputs = F.upsample(image, size=(384, 384), mode='bilinear', align_corners=True)
        res=model(image)[0]
        # res = F.interpolate(res, size=(H, W), mode='bilinear', align_corners=False)
        res = torch.sigmoid(res).data.cpu().numpy().squeeze()

        pred = (res - res.min()) / (res.max() - res.min() + 1e-8)
        pred = Image.fromarray(pred * 255).convert('L')
        pred = pred.resize((W, H), resample=Image.BILINEAR)
        pred = np.array(pred)



        FM.step(pred=pred, gt=gt)
        WFM.step(pred=pred, gt=gt)
        SM.step(pred=pred, gt=gt)
        EM.step(pred=pred, gt=gt)
        M.step(pred=pred, gt=gt)
        #
    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = M.get_results()["mae"]

    curr_results = {
        "MAE": mae,
        "Smeasure": sm,
        "wFmeasure": wfm,
        "meanEm": em["curve"].mean(),
    }


    return mae,wfm

def total_loss(pred, mask):
    pred = torch.sigmoid(pred)
    bce_loss = nn.BCELoss()
    bce = bce_loss(pred, mask)

    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - inter/(union-inter)
    iou = iou.mean()
    return bce+iou


def prepare_training():
    if config.get('resume') is not None:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = config.get('resume') + 1
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    # if local_rank == 0:
    #     log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def reshapePos(pos_embed, img_size):
    token_size = int(img_size // 16)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
    return pos_embed


def reshapeRel(k, rel_pos_params, img_size):
    if not ('2' in k or '5' in k or '8' in k or '11' in k):
        return rel_pos_params

    token_size = int(img_size // 16)
    h, w = rel_pos_params.shape
    rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
    rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
    return rel_pos_params[0, 0, ...]
def load(net,ckpt, img_size):
    ckpt=torch.load(ckpt,map_location='cpu')
    from collections import OrderedDict
    dict=OrderedDict()
    for k,v in ckpt.items():
        #把pe_layer改名
        if 'pe_layer' in k:
            dict[k[15:]] = v
            continue
        if 'pos_embed' in k :
            dict[k] = reshapePos(v, img_size)
            continue
        if 'rel_pos' in k:
            dict[k] = reshapeRel(k, v, img_size)
        elif "image_encoder" in k:
            if "neck" in k:
                #Add the original final neck layer to 3, 6, and 9, initialization is the same.
                for i in range(4):
                    new_key = "{}.{}{}".format(k[:18], i, k[18:])
                    dict[new_key] = v
            else:
                dict[k]=v
        if "mask_decoder.transformer" in k:
            dict[k] = v
        if "mask_decoder.iou_token" in k:
            dict[k] = v
        if "mask_decoder.output_upscaling" in k:
            dict[k] = v
    state = net.load_state_dict(dict, strict=False)
    return state
def train(train_loader,optimizer, model):
    model.train()
    for batch in train_loader:
        inputs = batch['inp']
        labels = batch['gt']
        edges = batch["edge"]


        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        edges = edges.type(torch.FloatTensor)
        # ori = ori.type(torch.FloatTensor)
        # print(ori)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v, edges_v =  Variable(inputs.cuda(), requires_grad=False), \
                                            Variable(labels.cuda(), requires_grad=False),\
                                         Variable(edges.cuda(), requires_grad=False)


        optimizer.zero_grad()

        # low_inputs = F.upsample(inputs_v, size=(384, 384), mode='bilinear', align_corners=True)

        preds = model(inputs_v)
        loss1 = total_loss(preds[0], labels_v) + total_loss(preds[1], labels_v) + total_loss(preds[2],
                                                                                             labels_v) + total_loss(
            preds[3], labels_v)
        loss2 = total_loss(preds[4], labels_v) + total_loss(preds[5], labels_v) + total_loss(preds[6], labels_v)
        # loss = total_loss(preds[0], labels_v)
        loss = loss1+0.4*loss2
        # print(torch.cuda.memory_reserved() / 1024 ** 3)
        # print(loss)
        # sam hq
        loss.backward()

        optimizer.step()



import time


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
def main(config_,dataset_name,model_name, save_path, args):
    global config, log, writer, log_info
    config = config_
    # log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    file_dir= "D:\yanfeng\BASNet-master\SemanticData\process\\"


    train_image_root = os.path.join(file_dir,dataset_name+"\\train\images\\")
    train_gt_root = os.path.join(file_dir,dataset_name+"\\train\gt\\")
    test_image_root =os.path.join(file_dir,dataset_name+"\\test\images\\")
    test_gt_root = os.path.join(file_dir,dataset_name+"\\test\gt\\")

    train_loader =  get_loader(train_image_root, train_gt_root, batchsize=4, trainsize=1024, is_train=True)

    config['model']['name'] = model_name
    model = models.make(config['model']).cuda()
    #
    #
    # load(model, config['sam_checkpoint'],512)

    sam_checkpoint = torch.load(config['sam_checkpoint'])
    model.load_state_dict(sam_checkpoint, strict=False)

    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)
        elif "mask_decoder" in name:
            para.requires_grad_(False)

   
    if dataset_name=="mvtec":
        epoch_num = 150
        epoch_val = 40
    elif dataset_name=="CrackSeg9k":
        epoch_num= 60
        epoch_val = 30
    elif dataset_name == "ZJU-Leaper":
        epoch_num = 20
        epoch_val = 16
    elif dataset_name == "Magnetic-tile-defect-datasets":
        epoch_num = 300
        epoch_val = 100


    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    lr_scheduler = CosineAnnealingLR(optimizer, epoch_num, eta_min=config.get('lr_min'))
    best_mae = 1
    best_wf=0

    for epoch in range(1, epoch_num + 1):
        print(epoch)
        start_time = time.time()
        train(train_loader, optimizer, model)
        end_time = time.time()
        print('Cost time: {:.4f}'.format(end_time - start_time))
        lr_scheduler.step()
        if epoch >= epoch_val:
            mae, wf = eval_psnr(test_image_root, test_gt_root, model)
            model.train()
            if wf > best_wf:
                save(config, model, save_path, model_name + "-" + dataset_name + '-' + f'{wf:.4f}' + '.pth')
                best_wf = wf
                #print(best_)
            print("mae:%.4f, best_mae:%.4f, wf: %.4f" % (mae, best_wf, wf))


def save(config, model, save_path, name):

    torch.save(model.state_dict(), os.path.join(save_path, name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default="./sam-vit-b.yaml")
    parser.add_argument('--name', default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    save_path = "./save"
    dataset_name="CrackSeg9k"
    model="defectsam"


    main(config, dataset_name, model, save_path, args=args)
  
