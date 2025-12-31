import argparse
import os
import yaml
import torch
from tqdm import tqdm
from torchvision import transforms
from sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure
from data_loader import get_loader
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import  time




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def eval_psnr(test_image_root, test_gt_root, model, inp_size, pred_save):
    
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()
    img_transform = transforms.Compose([
        transforms.Resize((inp_size, inp_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    images = [test_image_root + f for f in os.listdir(test_image_root)]
    gts = [test_gt_root + p for p in os.listdir(test_gt_root)]
    images = sorted(images)
    gts = sorted(gts)
    model.eval()
    total_time=0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for index in tqdm(range(len(images))):

        ori_image=Image.open(images[index]).convert("RGB")
        gt = cv2.imread(gts[index], cv2.IMREAD_GRAYSCALE)
        image = img_transform(ori_image).unsqueeze(0).cuda()


        H, W = gt.shape
        # print(print_gpu_memory_usage())

        with torch.no_grad():
            start_time = time.time()
            res=model(image)[0]
            end_time = time.time()
    
  
        res = torch.sigmoid(res).data.cpu().numpy().squeeze()

        pred = (res - res.min()) / (res.max() - res.min() + 1e-8)
        pred = Image.fromarray(pred * 255).convert('L')
        pred = pred.resize((W, H), resample=Image.BILINEAR)
        img_name = gts[index].split("\\")[-1]
        imgIdx = img_name.split(".")[0]
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
    # print(len(images)/total_time)

    curr_results = {

        "MAE": '%.4f' % mae,
        "wFmeasure": '%.4f' % wfm,
        "Smeasure": '%.4f' % sm,
        "meanFm": '%.4f' % fm["curve"].mean(),
        "meanEm": '%.4f' % em["curve"].mean(),
    }
    print(len(images) / total_time)


    return curr_results



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default="./sam-vit-b.yaml")
    parser.add_argument('--model',default="D:\yanfeng\sam\save\defectsam\CrackSeg9k.pth")
    parser.add_argument('--prompt', default='none')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    inp_size=1024
    config['model']['name']="defect_sam"
    config['model']['args']['inp_size'] = inp_size

    model = models.make(config['model']).cuda()

    from mmcv.cnn import get_model_complexity_info

    model.eval()

    model = model.cuda()
    from thop import profile, clever_format


    from thop import profile, clever_format

    #
    input = torch.randn(1, 3, 512, 512).cuda()
    flops, params = profile(model, inputs=(input,))

    flops, params = clever_format([flops, params], "%.3f")
    print(flops)
    print(params)

    
    dataset_names = ["ZJU-Leaper"]
    for dataset_name in dataset_names:
        print(dataset_name)

        sam_checkpoint = torch.load("D:\yanfeng\sam\save\\mobilesam\\mobile_sam-ZJU-Leaper-0.7422.pth")
        model.load_state_dict(sam_checkpoint,strict=False)

        file_dir = "D:\yanfeng\BASNet-master\SemanticData\process\\"
        test_image_root = os.path.join(file_dir, dataset_name + "\\test\\images\\")
        test_gt_root = os.path.join(file_dir, dataset_name + "\\test\\gt\\")

        # test_loader = get_loader(test_image_root, test_gt_root, batchsize=1, trainsize=1024, is_train=False)

        pred_dir = "D:\yanfeng\project\save\preds\\"
        pred_save = os.path.join(pred_dir, "defectsam\\" + dataset_name+"\\")
        # pred_save = os.path.join(pred_dir,  dataset_name)
        # pred_save="D:\yanfeng\sam\save\\ablation0\zju\\nocnn_share\\"

        if not os.path.isdir(pred_save):
            os.makedirs(pred_save)

        metrics= eval_psnr(test_image_root, test_gt_root, model, inp_size, pred_save)

        print(metrics)
    #
