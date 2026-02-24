import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

from dataloaders import utils
from dataloaders.dataset import BaseDataSets, RandomGenerator, TwoStreamBatchSampler
from utils import losses, metrics, ramps
from val_2D import test_single_volume_ds
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/URPC_Improved', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_urpc', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

# New hyperparameters for improvements
parser.add_argument('--focal_gamma', type=float, default=2.0,
                    help='gamma for focal loss')
parser.add_argument('--tversky_alpha', type=float, default=0.3,
                    help='alpha for tversky loss (emphasize FP)')
parser.add_argument('--tversky_beta', type=float, default=0.7,
                    help='beta for tversky loss (emphasize FN)')
parser.add_argument('--boundary_weight', type=float, default=1.0,
                    help='weight for boundary loss')
args = parser.parse_args()


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - handles class imbalance better than Dice
    alpha < beta: penalize false negatives more (good for small objects)
    alpha > beta: penalize false positives more
    """
    def __init__(self, n_classes, alpha=0.3, beta=0.7, smooth=1e-5):
        super(TverskyLoss, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha  # FP weight
        self.beta = beta    # FN weight
        self.smooth = smooth

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _tversky_index(self, score, target):
        target = target.float()
        tp = torch.sum(score * target)
        fp = torch.sum(score * (1 - target))
        fn = torch.sum((1 - score) * target)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return tversky

    def forward(self, inputs, target, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        
        loss = 0.0
        for i in range(self.n_classes):
            tversky = self._tversky_index(inputs[:, i], target[:, i])
            loss += (1 - tversky)
        return loss / self.n_classes


class BoundaryLoss(nn.Module):
    """
    Boundary Loss based on distance transform
    Helps improve segmentation near edges
    """
    def __init__(self, n_classes):
        super(BoundaryLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _compute_distance_map(self, label):
        """Compute distance transform for boundary loss"""
        label_np = label.cpu().numpy()
        batch_size = label_np.shape[0]
        dist_maps = []
        
        for b in range(batch_size):
            class_dist_maps = []
            for c in range(self.n_classes):
                binary_mask = (label_np[b, 0] == c).astype(np.float64)
                if binary_mask.sum() > 0:
                    # Distance transform from boundary
                    pos_dist = distance_transform_edt(binary_mask)
                    neg_dist = distance_transform_edt(1 - binary_mask)
                    # Signed distance: negative inside, positive outside
                    dist = neg_dist - pos_dist
                    # Normalize
                    dist = dist / (np.abs(dist).max() + 1e-8)
                else:
                    dist = np.ones_like(binary_mask)
                class_dist_maps.append(dist)
            dist_maps.append(np.stack(class_dist_maps, axis=0))
        
        dist_maps = np.stack(dist_maps, axis=0)
        return torch.from_numpy(dist_maps).float().to(label.device)

    def forward(self, inputs, target, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        
        # Compute distance maps
        target_unsqueeze = target.unsqueeze(1)
        dist_maps = self._compute_distance_map(target_unsqueeze)
        
        # Boundary loss: softmax output * distance map
        loss = torch.mean(inputs * dist_maps)
        return loss


class CombinedLoss(nn.Module):
    """Combined loss with Dice + Focal + Tversky + Boundary"""
    def __init__(self, n_classes, focal_gamma=2.0, tversky_alpha=0.3, 
                 tversky_beta=0.7, boundary_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.dice_loss = losses.DiceLoss(n_classes)
        self.focal_loss = FocalLoss(gamma=focal_gamma)
        self.tversky_loss = TverskyLoss(n_classes, alpha=tversky_alpha, beta=tversky_beta)
        self.boundary_loss = BoundaryLoss(n_classes)
        self.ce_loss = CrossEntropyLoss()
        self.boundary_weight = boundary_weight
        
    def forward(self, inputs, targets, inputs_soft=None, use_boundary=True):
        if inputs_soft is None:
            inputs_soft = torch.softmax(inputs, dim=1)
        
        # Standard losses
        loss_ce = self.ce_loss(inputs, targets.long())
        loss_dice = self.dice_loss(inputs_soft, targets.unsqueeze(1))
        loss_focal = self.focal_loss(inputs, targets.long())
        loss_tversky = self.tversky_loss(inputs, targets.unsqueeze(1), softmax=True)
        
        # Boundary loss (optional, can be expensive)
        if use_boundary and self.boundary_weight > 0:
            loss_boundary = self.boundary_loss(inputs, targets, softmax=True)
        else:
            loss_boundary = torch.tensor(0.0).to(inputs.device)
        
        # Combined: weighted sum
        # Dice + Focal handles class imbalance
        # Tversky helps with small objects
        # Boundary improves edges
        total_loss = (0.3 * loss_ce + 
                      0.3 * loss_dice + 
                      0.2 * loss_focal + 
                      0.1 * loss_tversky + 
                      self.boundary_weight * 0.1 * loss_boundary)
        
        return total_loss, {
            'ce': loss_ce.item(),
            'dice': loss_dice.item(),
            'focal': loss_focal.item(),
            'tversky': loss_tversky.item(),
            'boundary': loss_boundary.item() if isinstance(loss_boundary, torch.Tensor) else 0.0
        }


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    model = net_factory(net_type=args.model, in_chns=1,
                        class_num=num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    
    # Combined loss function with all improvements
    combined_loss_fn = CombinedLoss(
        n_classes=num_classes,
        focal_gamma=args.focal_gamma,
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        boundary_weight=args.boundary_weight
    )
    
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    kl_distance = nn.KLDivLoss(reduction='none')
    iterator = tqdm(range(max_epoch), ncols=70)
    
    # Flag to control boundary loss (expensive, enable after warmup)
    use_boundary = False
    boundary_start_iter = 5000  # Start boundary loss after 5000 iterations
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            outputs, outputs_aux1, outputs_aux2, outputs_aux3 = model(
                volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)
            outputs_aux2_soft = torch.softmax(outputs_aux2, dim=1)
            outputs_aux3_soft = torch.softmax(outputs_aux3, dim=1)

            # Enable boundary loss after warmup
            if iter_num >= boundary_start_iter:
                use_boundary = True

            # Combined loss for main output
            sup_loss_main, loss_dict_main = combined_loss_fn(
                outputs[:args.labeled_bs],
                label_batch[:args.labeled_bs],
                outputs_soft[:args.labeled_bs],
                use_boundary=use_boundary
            )
            
            # Simplified losses for auxiliary outputs (no boundary for efficiency)
            sup_loss_aux1, _ = combined_loss_fn(
                outputs_aux1[:args.labeled_bs],
                label_batch[:args.labeled_bs],
                outputs_aux1_soft[:args.labeled_bs],
                use_boundary=False
            )
            sup_loss_aux2, _ = combined_loss_fn(
                outputs_aux2[:args.labeled_bs],
                label_batch[:args.labeled_bs],
                outputs_aux2_soft[:args.labeled_bs],
                use_boundary=False
            )
            sup_loss_aux3, _ = combined_loss_fn(
                outputs_aux3[:args.labeled_bs],
                label_batch[:args.labeled_bs],
                outputs_aux3_soft[:args.labeled_bs],
                use_boundary=False
            )

            # Average supervised loss
            supervised_loss = (sup_loss_main + sup_loss_aux1 + sup_loss_aux2 + sup_loss_aux3) / 4

            # Consistency loss (unchanged from original URPC)
            preds = (outputs_soft + outputs_aux1_soft +
                     outputs_aux2_soft + outputs_aux3_soft) / 4

            variance_main = torch.sum(kl_distance(
                torch.log(outputs_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_main = torch.exp(-variance_main)

            variance_aux1 = torch.sum(kl_distance(
                torch.log(outputs_aux1_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux1 = torch.exp(-variance_aux1)

            variance_aux2 = torch.sum(kl_distance(
                torch.log(outputs_aux2_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux2 = torch.exp(-variance_aux2)

            variance_aux3 = torch.sum(kl_distance(
                torch.log(outputs_aux3_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
            exp_variance_aux3 = torch.exp(-variance_aux3)

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            
            consistency_dist_main = (
                preds[args.labeled_bs:] - outputs_soft[args.labeled_bs:]) ** 2
            consistency_loss_main = torch.mean(
                consistency_dist_main * exp_variance_main) / (torch.mean(exp_variance_main) + 1e-8) + torch.mean(variance_main)

            consistency_dist_aux1 = (
                preds[args.labeled_bs:] - outputs_aux1_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux1 = torch.mean(
                consistency_dist_aux1 * exp_variance_aux1) / (torch.mean(exp_variance_aux1) + 1e-8) + torch.mean(variance_aux1)

            consistency_dist_aux2 = (
                preds[args.labeled_bs:] - outputs_aux2_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux2 = torch.mean(
                consistency_dist_aux2 * exp_variance_aux2) / (torch.mean(exp_variance_aux2) + 1e-8) + torch.mean(variance_aux2)

            consistency_dist_aux3 = (
                preds[args.labeled_bs:] - outputs_aux3_soft[args.labeled_bs:]) ** 2
            consistency_loss_aux3 = torch.mean(
                consistency_dist_aux3 * exp_variance_aux3) / (torch.mean(exp_variance_aux3) + 1e-8) + torch.mean(variance_aux3)

            consistency_loss = (consistency_loss_main + consistency_loss_aux1 +
                                consistency_loss_aux2 + consistency_loss_aux3) / 4
            
            loss = supervised_loss + consistency_weight * consistency_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Cosine annealing with warm restarts (better than polynomial decay)
            if iter_num < 1000:
                # Warmup phase
                lr_ = base_lr * (iter_num / 1000)
            else:
                # Cosine annealing
                lr_ = base_lr * 0.5 * (1 + np.cos(np.pi * (iter_num - 1000) / (max_iterations - 1000)))
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            
            # Logging
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/supervised_loss', supervised_loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_dict_main['ce'], iter_num)
            writer.add_scalar('info/loss_dice', loss_dict_main['dice'], iter_num)
            writer.add_scalar('info/loss_focal', loss_dict_main['focal'], iter_num)
            writer.add_scalar('info/loss_tversky', loss_dict_main['tversky'], iter_num)
            writer.add_scalar('info/loss_boundary', loss_dict_main['boundary'], iter_num)
            writer.add_scalar('info/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight', consistency_weight, iter_num)
            
            logging.info(
                'iteration %d : loss : %f, ce: %f, dice: %f, focal: %f, tversky: %f' %
                (iter_num, loss.item(), loss_dict_main['ce'], loss_dict_main['dice'],
                 loss_dict_main['focal'], loss_dict_main['tversky']))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs_vis = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs_vis[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume_ds(
                        sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                logging.info(
                    'iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
