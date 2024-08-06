from torchvision import transforms
from utils import *
from datetime import datetime
import torch.nn as nn
class setting_config:
    """
    the config of training setting.
    """
    model_name = 'RepViT'
    model_config = {
        'num_classes': 12,    #7 or 12
        'input_channels': 3,
    }
    test_weights = ''
    datasets = 'normal'
    if datasets == 'normal':
        data_path = ''
    elif datasets == 'extreme':
        data_path = ''
    else:
        raise Exception('datasets in not right!')
    criterion = nn.CrossEntropyLoss()
    num_classes = 12   #7 or 12
    input_size_h = 224
    input_size_w = 224
    input_channels = 3
    num_workers = 0
    seed = 42
    batch_size = 32
    epochs = 100

    work_dir = 'results/' + network + '_' + datasets + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    print_interval = 20
    val_interval = 30
    save_interval = 100
    threshold = 0.5

    if model_name == "ConvNext":
        net = convnext_tiny()
    elif model_name == "DiSpecNet":
        net = DiSpecNet(each_cl_loss=True,glo=True)
    elif model_name == "EfficientNet":
        net = EfficientNet()
    elif model_name == "FasterNet":
        net = FasterNet()
    elif model_name == "GFNet":
        net = GFNet(
                img_size=224,
                patch_size=16, embed_dim=256, depth=12, mlp_ratio=4,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
    elif model_name == "MambaOut":
        net = mambaout_kobe()
    elif model_name == "Parallel_ViT":
        net = Parallel_ViT(
                image_size = 224,
                patch_size = 16,
                num_classes = 7,
                 dim = 1024,
                depth = 6,
                heads = 8,
                mlp_dim = 2048,
                num_parallel_branches = 2,  # in paper, they claimed 2 was optimal
                dropout = 0.1,
                emb_dropout = 0.1
            )
    elif model_name == "Patch_Merger":
        net = Patch_Merger(
                image_size = 224,
                patch_size = 16,
                num_classes = 7,
                dim = 1024,
                depth = 12,
                heads = 8,
                patch_merge_layer = 6,        # at which transformer layer to do patch merging
                patch_merge_num_tokens = 8,   # the output number of tokens from the patch merge
                mlp_dim = 2048,
                dropout = 0.1,
                emb_dropout = 0.1
        )
    elif model_name == "RepViT":
        net = repvit_m0_6()
    elif model_name == "SepViT":
        net = SepViT(
                num_classes = 7,
                dim = 32,               # dimensions of first stage, which doubles every stage (32, 64, 128, 256) for SepViT-Lite
                dim_head = 32,          # attention head dimension
                heads = (1, 2, 4, 8),   # number of heads per stage
                depth = (1, 2, 6, 2),   # number of transformer blocks per stage
                window_size = 7,        # window size of DSS Attention block
                dropout = 0.1           # dropout
            )
    elif model_name == "STViT":
        net = stvit_small()
    elif model_name == "SwinTransformer":
        net = SwinTransformer()
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    opt = 'Adam'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop',
                   'SGD'], 'Unsupported optimizer!'
    if opt == 'Adadelta':
        lr = 0.01  # default: 1.0 – coefficient that scale delta before it is applied to the parameters
        rho = 0.9  # default: 0.9 – coefficient used for computing a running average of squared gradients
        eps = 1e-6  # default: 1e-6 – term added to the denominator to improve numerical stability
        weight_decay = 0.05  # default: 0 – weight decay (L2 penalty)
    elif opt == 'Adagrad':
        lr = 0.01  # default: 0.01 – learning rate
        lr_decay = 0  # default: 0 – learning rate decay
        eps = 1e-10  # default: 1e-10 – term added to the denominator to improve numerical stability
        weight_decay = 0.05  # default: 0 – weight decay (L2 penalty)
    elif opt == 'Adam':
        lr = 0.001  # default: 1e-3 – learning rate
        eps = 1e-8  # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 0.0001  # default: 0 – weight decay (L2 penalty)
        amsgrad = False  # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond
    elif opt == 'AdamW':
        lr = 0.001  # default: 1e-3 – learning rate
        eps = 1e-8  # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 1e-2  # default: 1e-2 – weight decay coefficient
        amsgrad = False  # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond
    elif opt == 'Adamax':
        lr = 2e-3  # default: 2e-3 – learning rate
        eps = 1e-8  # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 0  # default: 0 – weight decay (L2 penalty)
    elif opt == 'ASGD':
        lr = 0.01  # default: 1e-2 – learning rate
        lambd = 1e-4  # default: 1e-4 – decay term
        alpha = 0.75  # default: 0.75 – power for eta update
        t0 = 1e6  # default: 1e6 – point at which to start averaging
        weight_decay = 0  # default: 0 – weight decay
    elif opt == 'RMSprop':
        lr = 1e-2  # default: 1e-2 – learning rate
        momentum = 0  # default: 0 – momentum factor
        alpha = 0.99  # default: 0.99 – smoothing constant
        eps = 1e-8  # default: 1e-8 – term added to the denominator to improve numerical stability
        centered = False  # default: False – if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
        weight_decay = 0  # default: 0 – weight decay (L2 penalty)
    elif opt == 'Rprop':
        lr = 1e-2  # default: 1e-2 – learning rate
        etas = (0.5,
                1.2)  # default: (0.5, 1.2) – pair of (etaminus, etaplis), that are multiplicative increase and decrease factors
        step_sizes = (1e-6, 50)  # default: (1e-6, 50) – a pair of minimal and maximal allowed step sizes
    elif opt == 'SGD':
        lr = 0.01  # – learning rate
        momentum = 0.9  # default: 0 – momentum factor
        weight_decay = 0.05  # default: 0 – weight decay (L2 penalty)
        dampening = 0  # default: 0 – dampening for momentum
        nesterov = False  # default: False – enables Nesterov momentum

    sch = 'CosineAnnealingLR'
    if sch == 'StepLR':
        step_size = epochs // 5  # – Period of learning rate decay.
        gamma = 0.5  # – Multiplicative factor of learning rate decay. Default: 0.1
        last_epoch = -1  # – The index of last epoch. Default: -1.
    elif sch == 'MultiStepLR':
        milestones = [60, 120, 150]  # – List of epoch indices. Must be increasing.
        gamma = 0.1  # – Multiplicative factor of learning rate decay. Default: 0.1.
        last_epoch = -1  # – The index of last epoch. Default: -1.
    elif sch == 'ExponentialLR':
        gamma = 0.99  # – Multiplicative factor of learning rate decay.
        last_epoch = -1  # – The index of last epoch. Default: -1.
    elif sch == 'CosineAnnealingLR':
        T_max = 20  # – Maximum number of iterations. Cosine function period.
    elif sch == 'ReduceLROnPlateau':
        mode = 'min'  # – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
        factor = 0.1  # – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
        patience = 10  # – Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
        threshold = 0.0001  # – Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
        threshold_mode = 'rel'  # – One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’.
        cooldown = 0  # – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
        min_lr = 0  # – A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
        eps = 1e-08  # – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.
    elif sch == 'CosineAnnealingWarmRestarts':
        T_0 = 50  # – Number of iterations for the first restart.
        T_mult = 2  # – A factor increases T_{i} after a restart. Default: 1.
        eta_min = 1e-6  # – Minimum learning rate. Default: 0.
        last_epoch = -1  # – The index of last epoch. Default: -1.
    elif sch == 'WP_MultiStepLR':
        warm_up_epochs = 10
        gamma = 0.1
        milestones = [125, 225]
    elif sch == 'WP_CosineLR':
        warm_up_epochs = 20