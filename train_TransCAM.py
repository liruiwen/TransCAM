import numpy as np
import torch
import os

from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
import voc12.data
from tool import pyutils, imutils
import argparse
import importlib
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=20, type=int)
    parser.add_argument("--network", default="network.conformer_CAM", type=str)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--arch", default='sm', type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--session_name", default="TransCAM", type=str)
    parser.add_argument("--crop_size", default=512, type=int)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--voc12_root", default='VOCdevkit/VOC2012', type=str)
    parser.add_argument("--tblog_dir", default='./tblog', type=str)
    parser.add_argument("--save_dir", default='./', type=str)
    args = parser.parse_args()

    pyutils.Logger(args.session_name + '.log')

    print(vars(args))

    model = getattr(importlib.import_module(args.network), 'Net_' + args.arch)()

    print(model)

    tblogger = SummaryWriter(args.tblog_dir)

    train_dataset = voc12.data.VOC12ClsDataset(args.train_list, voc12_root=args.voc12_root,
                                               transform=transforms.Compose([
                                                   imutils.RandomResizeLong(320, 640),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,
                                                                          hue=0.1),
                                                   np.asarray,
                                                   imutils.Normalize(),
                                                   imutils.RandomCrop(args.crop_size),
                                                   imutils.HWC_to_CHW,
                                                   torch.from_numpy
                                               ]))

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wt_dec, eps=1e-8)

    checkpoint = torch.load(args.weights, map_location='cpu')
    if 'model' in checkpoint.keys():
        checkpoint = checkpoint['model']
    else:
        checkpoint = checkpoint
    model_dict = model.state_dict()
    for k in ['trans_cls_head.weight', 'trans_cls_head.bias']:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint[k]
    for k in ['conv_cls_head.weight', 'conv_cls_head.bias']:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint[k]
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss')

    for ep in range(args.max_epoches):
        for iter, pack in enumerate(train_data_loader):
            _, img, label = pack
            N, C, H, W = img.size()
            bg_score = torch.ones((N, 1))
            label = torch.cat((bg_score, label), dim=1)
            label = label.cuda(non_blocking=True).unsqueeze(2).unsqueeze(3)

            logits_conv, logits_trans, cams = model('transcam', img)

            loss = F.multilabel_soft_margin_loss((logits_conv + logits_trans).unsqueeze(2).unsqueeze(3)[:, 1:, :, :], label[:, 1:, :, :])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meter.add({'loss': loss.item()})
        else:
            print('epoch: %5d' % ep,
                  'loss: %.4f' % avg_meter.get('loss'), flush=True)
            avg_meter.pop()

    torch.save(model.module.state_dict(), os.path.join(args.save_dir, args.session_name + '.pth'))


