import sys, json, os, time, random
import os.path as osp
import numpy as np
import torch
from tqdm import trange
from utils.data_load import get_train_val_seg_loaders
from utils.metric_factory import fast_bin_dice
from utils.model_factory_seg import get_model
from utils.loss_factory import get_loss
def get_args_parser():
    import argparse

    def str2bool(v):
        # as seen here: https://stackoverflow.com/a/43357954/3208255
        if isinstance(v, bool):
            return v
        if v.lower() in ('true', 'yes'):
            return True
        elif v.lower() in ('false', 'no'):
            return False
        else:
            raise argparse.ArgumentTypeError('boolean value expected.')

    parser = argparse.ArgumentParser(description='Training for 2d Biomedical Image Segmentation')
    parser.add_argument('--csv_path_tr', type=str, default='data/tr_f1.csv', help='csv path training data')
    parser.add_argument('--model_name', type=str, default='fpn_resnet18', help='architecture')
    parser.add_argument('--loss1', type=str, default='bce', help='1st loss')
    parser.add_argument('--loss2', type=str, default=None, help='2nd loss')
    parser.add_argument('--load_path', type=str, default='', help='path to weight of pretrained model, if any')
    parser.add_argument('--alpha1', type=float, default=1., help='multiplier in alpha1*loss1+alpha2*loss2')
    parser.add_argument('--alpha2', type=float, default=0., help='multiplier in alpha1*loss1+alpha2*loss2')
    parser.add_argument('--im_size', type=str, default='128/128', help='im size/spatial xy dimension')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--tr_pctg', type=float, default=1.0, help='fraction of the tr set to use')
    parser.add_argument('--vl_pctg', type=float, default=1.0, help='fraction of the vl set to use')
    parser.add_argument('--optimizer', type=str, default='nadam', choices=('sgd', 'adamw', 'nadam'), help='optimizer choice')
    parser.add_argument('--lr', type=float, default=1e-4, help='max learning rate')
    parser.add_argument('--n_epochs', type=int, default=20, help='training epochs')
    parser.add_argument('--vl_interval', type=int, default=5, help='how often we check performance and maybe save')
    parser.add_argument('--cyclical_lr', type=str2bool, nargs='?', const=True, default=True, help='re-start lr each vl_interval epochs')
    parser.add_argument('--metric', type=str, default='dsc', help='which metric to use for monitoring progress (AUC)')
    parser.add_argument('--save_path', type=str, default='delete', help='path to save model (defaults to date/time')
    parser.add_argument('--seed', type=int, default=None, help='fixes random seed (slower!)')
    parser.add_argument('--num_workers', type=int, default=8, help='number of parallel (multiprocessing) workers')
    args = parser.parse_args()

    return args


def set_seeds(seed_value, use_cuda):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def validate(model, loader, loss_fn=torch.nn.functional.binary_cross_entropy_with_logits):
    model.eval()
    thresh = 0.5
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    dscs, losses = [], []
    with trange(len(loader)) as t:
        n_elems, running_loss = 0, 0
        for (i_batch, batch_data) in enumerate(loader):
            inputs, labels = batch_data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            losses.append(loss.item())
            outputs = outputs.sigmoid().cpu().squeeze().numpy() > thresh
            labels = labels.cpu().squeeze().numpy().astype(bool)
            for j in range(len(outputs)):
                dsc_score = fast_bin_dice(labels[j], outputs[j])
                dscs.append(100*dsc_score)

            n_elems += 1
            running_loss += loss
            run_loss = running_loss / n_elems
            t.set_postfix(LOSS='{:.2f}'.format(100 * run_loss))
            t.update()

    loss = np.mean(np.array(losses))
    dsc =  np.mean(np.array(dscs))
    return dsc, loss

def train_one_epoch(model, loader, loss_fn, optimizer, scheduler):
    model.train()
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    with trange(len(loader)) as t:
        n_elems, running_loss = 0, 0
        for (i_batch, batch_data) in enumerate(loader):
            inputs, labels = batch_data
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            lr=get_lr(optimizer)
            running_loss += loss.detach().item() * inputs.shape[0]
            n_elems += inputs.shape[0]  # total nr of items processed
            run_loss = running_loss / n_elems
            t.set_postfix(LOSS_lr='{:.4f}/{:.6f}'.format(run_loss, lr))
            t.update()

def set_tr_info(tr_info, epoch=0, ovft_metrics=None, vl_metrics=None, best_epoch=False):
    # I customize this for each project.
    # Here tr_info contains f1, auc, bacc, loss values.
    # Also, and vl_metrics contain (in this order) f1, auc, bacc, loss
    if best_epoch:
        tr_info['best_tr_dsc'] = tr_info['tr_dscs'][-1]
        tr_info['best_vl_dsc'] = tr_info['vl_dscs'][-1]
        tr_info['best_tr_loss'] = tr_info['tr_losses'][-1]
        tr_info['best_vl_loss'] = tr_info['vl_losses'][-1]
        tr_info['best_epoch'] = epoch
    else:
        tr_info['tr_dscs'].append(ovft_metrics[0])
        tr_info['vl_dscs'].append(vl_metrics[0])
        tr_info['tr_losses'].append(ovft_metrics[-1])
        tr_info['vl_losses'].append(vl_metrics[-1])

    return tr_info

def init_tr_info():
    # I customize this function for each project.
    tr_info = dict()
    tr_info['tr_dscs'], tr_info['vl_dscs'] = [], []
    tr_info['tr_losses'], tr_info['vl_losses'] = [], []

    return tr_info

def get_eval_string(tr_info, epoch, finished=False, vl_interval=1):
    # I customize this function for each project.
    # Pretty prints first three values of train/val metrics to a string and returns it
    # Used also by the end of training (finished=True)
    ep_idx = len(tr_info['tr_losses'])-1
    if finished:
        ep_idx = epoch
        epoch = (epoch+1) * vl_interval - 1

    s = 'Ep. {}: Train||Val DSC: {:.2f}||{:.2f} - Loss: {:.4f}||{:.4f}'.format(str(epoch+1).zfill(3),
         tr_info['tr_dscs'][ep_idx], tr_info['vl_dscs'][ep_idx], tr_info['tr_losses'][ep_idx], tr_info['vl_losses'][ep_idx])
    return s
def train_model(model, optimizer, loss_fn, tr_loader, ovft_loader, vl_loader, scheduler, metric, n_epochs, vl_interval, save_path):
    best_metric, best_epoch = 0, 0
    tr_info = init_tr_info()

    for epoch in range(n_epochs):
        print('Epoch {:d}/{:d}'.format(epoch + 1, n_epochs))
        # train one epoch
        train_one_epoch(model, tr_loader, loss_fn, optimizer, scheduler)
        if (epoch + 1) % vl_interval == 0:
            with torch.no_grad():
                ovft_metrics = validate(model, ovft_loader)
                vl_metrics = validate(model, vl_loader)
            tr_info = set_tr_info(tr_info, epoch, ovft_metrics, vl_metrics)
            s = get_eval_string(tr_info, epoch)
            print(s)
            with open(osp.join(save_path, 'train_log.txt'), 'a') as f:
                print(s, file=f)
            # check if performance was better than anyone before and checkpoint if so
            if metric == 'dsc': curr_metric = tr_info['vl_dscs'][-1]
            elif metric =='bce': curr_metric = tr_info['vl_losses'][-1]
            else: sys.exit('bad metric')

            if curr_metric > best_metric:
                print('-------- Best {} attained. {:.2f} --> {:.2f} --------'.format(metric, best_metric, curr_metric))
                torch.save(model.state_dict(), osp.join(save_path, 'best_model.pth'))
                best_metric, best_epoch = curr_metric, epoch + 1
                tr_info = set_tr_info(tr_info, epoch+1, best_epoch=True)
            else:
                print('-------- Best {} so far {:.2f} at epoch {:d} --------'.format(metric, best_metric, best_epoch))
    del model, tr_loader, vl_loader
    # maybe this works also? tr_loader.dataset._fill_cache
    torch.cuda.empty_cache()
    return tr_info

if __name__ == '__main__':

    args = get_args_parser()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # reproducibility
    seed_value = 0
    set_seeds(seed_value, use_cuda)

    # logging
    save_path = osp.join('experiments', args.save_path)
    os.makedirs(save_path, exist_ok=True)
    config_file_path = osp.join(save_path, 'config.cfg')
    with open(config_file_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # gather parser parameters
    model_name = args.model_name
    optimizer_choice = args.optimizer
    lr, bs = args.lr, args.batch_size
    n_epochs, vl_interval, metric = args.n_epochs, args.vl_interval, args.metric
    csv_path_tr, nw = args.csv_path_tr, args.num_workers

    im_size = args.im_size.split('/')
    im_size = tuple(map(int, im_size))

    print('* Instantiating a {} model'.format(model_name))
    n_classes = 1
    in_c = 3
    model = get_model(args.model_name, n_classes=n_classes, in_c=in_c)
    if args.load_path != '':
        model.load_state_dict(torch.load(osp.join('experiments', args.load_path, 'best_model.pth')))
        print('* Loaded pretrained weights')

    print('* Creating Dataloaders, batch size = {}, workers = {}'.format(bs, nw))
    tr_loader, ovft_loader, vl_loader = get_train_val_seg_loaders(csv_path_tr, bs, im_size, nw, args.tr_pctg, args.vl_pctg)
    imgs, segs = next(iter(tr_loader))
    # print(imgs.shape, segs.shape)
    # print(segs.unique())

    model = model.to(device)
    print('Total params: {0:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    if optimizer_choice == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif optimizer_choice == 'nadam':
        optimizer = torch.optim.NAdam(model.parameters(), lr=args.lr)
    elif optimizer_choice == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=3e-5, momentum=0.99, nesterov=True)
    else: sys.exit('please choose between sgd, adam or nadam optimizers')

    if args.cyclical_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=vl_interval*len(tr_loader), eta_min=0)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs*len(tr_loader), eta_min=0)

    loss_fn = get_loss(args.loss1, args.loss2, args.alpha1, args.alpha2)
    print('* Instantiating loss function {:.2f}*{} + {:.2f}*{}'.format(args.alpha1, args.loss1, args.alpha2, args.loss2))


    print('* Starting to train\n', '-' * 10)
    start = time.time()
    tr_info = train_model(model, optimizer, loss_fn, tr_loader, ovft_loader, vl_loader, scheduler, metric, n_epochs, vl_interval, save_path)
    end = time.time()

    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print('Training time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds))

    with (open(osp.join(save_path, 'log.txt'), 'a') as f):
        print('Best epoch = {}/{}: Tr/Vl DSC={:.2f}/{:.2f} - Loss = {:.4f}/{:.4f}\n'.format(
      tr_info['best_epoch'], n_epochs, tr_info['best_tr_dsc'], tr_info['best_vl_dsc'],
            tr_info['best_tr_loss'], tr_info['best_vl_loss']), file=f)
        for j in range(n_epochs//vl_interval):
            s = get_eval_string(tr_info, epoch=j, finished=True, vl_interval=vl_interval)
            print(s, file=f)
        print('\nTraining time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds), file=f)

    print('Done. Training time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds))