import os
import yaml
import torch
import argparse
import numpy as np
import skimage.io

from collections import OrderedDict
from models import MobileUNet, SoftDiceLoss, IoU, utils_pytorch
from dataset import *

import warnings;
warnings.simplefilter('ignore')


def get_configs(args):
    with open(args.data, encoding='ascii', errors='ignore') as f:
        cfg = yaml.safe_load(f)

    ## Dataset config
    data_cfg = {
        'labels': cfg.get('labels', ['cv', 'pv']),
        'image_reader': skimage.io.imread, 
        'masks_reader': skimage.io.imread,
        'rescale_factor': cfg.get('rescale_factor', 0.5), 
        'dapi_only': cfg.get('dapi_only', True), 
    }

    ## Train Config
    train_cfg = utils_pytorch.TrainConfig(
        GPU_IDS = args.gpu_ids if args.gpu_ids is not None else [],
        BATCH_SIZE = args.batch_size,
        CLASS_WEIGHTS = [1] * len(data_cfg['labels']),
        OPTIMIZER = ('Adam', {'lr': 1e-3, 'betas': (0.9, 0.999)}), # ('SGD', {'lr': 0.01, 'momentum': 0.9}), # 
        SAVE_FREQUENCY = 1,
        EVAL_FREQUENCY = 1,
        LR_SCHEDULER = ('MultiStepLR', {'milestones': [200, 400, 600, 800], 'gamma': 0.8}),
        REDUCE_LR_ON_PLATEAU = ('loss/val', {'mode': 'min', 'factor': 0.5, 
                                'patience': 100, 'verbose': False, 'threshold': 1e-4})
    )
    
    return cfg, data_cfg, train_cfg


def load_cfg(cfg):
    if isinstance(cfg, dict):
        yaml = cfg
    elif isinstance(cfg, str):
        with open(cfg, encoding='ascii', errors='ignore') as f:
            yaml = yaml.safe_load(f)

    return yaml


class Trainer(utils_pytorch.TorchModel):
    def build_model(self, **kwargs):
        """ Build a pytorch model. 
            function should return: torch.nn.Module, (layer_regex)
        """
        return kwargs['net'], {}

    def get_criterion(self, config):
        weight = config.CLASS_WEIGHTS
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float)
        return {'dice_loss': SoftDiceLoss(weight=weight), 
                'cent_loss': torch.nn.BCELoss(weight=weight),
                'iou': IoU(mode='iou')}

    def _forward(self, batch_data):
        X, y_true = batch_data
        batch_size = X.size(0)
        y_pred = self.net(X)[:,1:]

        ## loss and metric
        # cent = self.criterion['cent_loss'](y_pred, y_true)
        loss = dice = self.criterion['dice_loss'](y_pred, y_true)
        ious = self.criterion['iou'](y_pred, y_true)
        ious = ious.cpu().detach().numpy()
        class_ious = [(x/y if y > 0 else 0, y) 
                      for x, y in zip(np.sum(ious, axis=0), np.sum(ious > 0, axis=0))]

        batch_stats = OrderedDict({
            'loss': (loss, batch_size),
            'iou_ave': (np.nanmean(ious), batch_size),
            **{'iou_{}'.format(i): _ for i, _ in enumerate(class_ious)},
        })

        return loss, batch_stats


def build_model(num_classes=None, in_channels=1, weights_path=None):
    try:
        model = torch.jit.load(weights_path)
    except:
        if weights_path is not None:
            weights = torch.load(weights_path, map_location='cpu')
            in_channels = in_channels or list(weights.values())[0].shape[1]
            num_classes = num_classes or list(weights.values())[-1].shape[0]
        else:
            weights = None
            assert num_classes is not None, f"Please specify num_classes if weights_path is not given."
            assert in_channels is not None, f"Please specify in_channels if weights_path is not given."

        encoder = {'architecture': 'mobilenet_v2', 'pretrained': True,
                   'width_mult': 1.0, 'in_channels': in_channels}
        model = MobileUNet(num_classes=num_classes, encoder=encoder)
        if weights is not None:
            model.load_state_dict(weights, strict=False)

    return model


def main(args):
    cfg, data_cfg, train_cfg = get_configs(args)
    in_channels = 1 if data_cfg['dapi_only'] else 3
    train_shape = (in_channels, 1024, 1024)
    inference_shape = (in_channels, int(2048*data_cfg['rescale_factor']), int(4096*data_cfg['rescale_factor']))

    ## Training dataset and dataloader
    train_dataset = SegDataset(cfg['train'], cfg['root'], processor=train_processor, 
                               output_shape=train_shape, **data_cfg)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                               num_workers=args.num_workers, shuffle=True)

    ## Validation dataset and dataloader
    val_dataset = SegDataset(cfg['val'], cfg['root'], processor=val_processor, 
                             output_shape=inference_shape, **data_cfg)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2, num_workers=2, shuffle=False)
    print(f"No. validation samples = {len(val_dataset)}")

    ## Test dataset and dataloader
    if 'test' in data_cfg:
        test_dataset = SegDataset(cfg['test'], cfg['root'], processor=val_processor, 
                                  output_shape=inference_shape, **data_cfg)
        print(f"No. test samples = {len(test_dataset)}")
        eval_loaders = {
            'test': torch.utils.data.DataLoader(test_dataset, batch_size=2, num_workers=2, shuffle=False),
        }
    else:
        test_dataset, eval_loaders = None, {}

    # train model
    model = build_model(1 + len(data_cfg['labels']), in_channels, weights_path=args.weights)
    trainer = Trainer(model_dir=args.exp_dir, model_name='GoZDeep', net=model)

    trainer.train(
        train_loader, val_loader, 
        epochs=args.num_epochs, config=train_cfg, 
        eval_loaders=eval_loaders, verbose=2,
    )

    # script = model.to_script(torch.ones((1,) + inference_shape))
    # script.save(os.path.join(model.log_dir, "{}.last.trace.pt".format(model.model_name)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GoZDeep training with MobileUNet.', add_help=False)
    parser.add_argument('--data', required=True, type=str, help="Config file with image and mask paths and hyperparameters.")
    parser.add_argument('--weights', default=None, type=str, help="Model file or weights path." )
    parser.add_argument('--exp_dir', default='exp', type=str, help="Experiment folder.")
    parser.add_argument('--gpu_ids', nargs='*', type=int, help='GPU device ids for training, script will use cpu if not given.')
    parser.add_argument('--num_epochs', default=800, type=int, help='Number of epochs.')
    parser.add_argument('--batch_size', default=4, type=int, help='Number of batch size.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers for data loader.')

    args = parser.parse_args()
    main(args)
