# MODEL OPTS
def model_args(parser):
    group = parser.add_argument_group('Model', 'Arguments control Model')
    group.add_argument('--arch', default='wrn', type=str, choices=['wrn', 'resnet', 'pre_resnet'],
                       help='model architecture')
    group.add_argument('--depth', default=16, type=int, 
                       help='depth of the model')
    group.add_argument('--width', default=10, type=int, 
                       help='widen factor for WideResNet')
    group.add_argument('--gpu', default='0,1', type=str, 
                       help='gpu id')
    group.add_argument('--seed', default=233, type=int,
                       help='random seed')


# DATALOADING OPTS
def data_args(parser):
    group = parser.add_argument_group('Data', 'Arguments control Data and loading for training')
    group.add_argument('--data-dir', type=str, default='./data',
                       help='Dataset directory')
    group.add_argument('--batch-size', type=int, default=128,
                       help='batch size of the train loader')


# BASE TRAINING ARGS
def base_train_args(parser):
    group = parser.add_argument_group('Base_Training', 'Base arguments to configure training')
    group.add_argument('--epochs', default=200, type=int, 
                       help='number of training epochs')
    group.add_argument('--lr', default=0.1, type=float, 
                       help='learning rate')
    group.add_argument('--lr-min', default=0., type=float, 
                       help='minimal learning rate for cyclic schedule')
    group.add_argument('--lr-max', default=0.2, type=float, 
                       help='maximal learning rate for cyclic schedule')
    group.add_argument('--lr-sch', default='multistep', choices=['cyclic', 'multistep'],
                       help='learning rate schedule type')
    group.add_argument('--sch-intervals', nargs='*', default=[], type=int, 
                       help='learning scheduler milestones for multistep schedule')
    group.add_argument('--lr-gamma', default=0.1, type=float, 
                       help='learning rate decay ratio')
    group.add_argument('--weight-decay', default=5e-4, type=float,
                       help='weight decay')
    group.add_argument('--momentum', default=0.9, type=float,
                       help='momentum for SGD')
    group.add_argument('--donot-test-robust', action='store_false', dest='test_robust',
                       help='whether test robust accuracy during the training')
    group.add_argument('--donot-save-eps', action="store_false", dest='save_eps',
                       help='whether save the epsilon value for each sample per epoch')
    group.add_argument('--donot-save-loss', action="store_false", dest='save_loss',
                       help='whether save the loss value for each sample per epoch')
    group.add_argument('--donot-save-fosc', action="store_false", dest='save_fosc',
                       help='whether save the fosc value for each sample per epoch')
    group.add_argument('--eval-when-attack', action='store_true',
                       help='whether set model to eval mode when generating adv examples')
    group.add_argument('--amp', action='store_true',
                       help='whether use automatic mixed precision from Apex')


# AMP ARGS
# adapted from https://github.com/locuslab/fast_adversarial/blob/master/CIFAR10/train_fgsm.py
def amp_args(parser):
    group = parser.add_argument_group('AMP', 'Arguments to configure Automatic Mixed Precision training')
    group.add_argument('--opt-level', default='O1', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    group.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    group.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')


# MADRY ADVERSARIAL TRAINING ARGS
def madry_advt_args(parser):
    group = parser.add_argument_group('Madry Adversarial_Training', 'Arguments to configure Madry adversarial training')
    group.add_argument('--eps', default=8, type=int, 
                       help='perturbation budget for adversarial training')
    group.add_argument('--alpha', default=2, type=int, 
                       help='step size for adversarial training')
    group.add_argument('--steps', default=10, type=int, 
                       help='number of steps for adversarial training')
    group.add_argument('--increase-steps', action='store_true',
                       help='whether increase the number of steps during the training')
    group.add_argument('--more-steps', nargs='*', default=[], type=int,
                       help='other numbers of steps that would be used to generate adv examples')
    group.add_argument('--steps-intervals', nargs='*', default=[], type=int,
                       help='the timespot to change the number of steps')
    group.add_argument('--increase-eps', action='store_true',
                       help='whether increase the number of steps during the training')
    group.add_argument('--more-eps', nargs='*', default=[], type=int,
                       help='other numbers of steps that would be used to generate adv examples')
    group.add_argument('--eps-intervals', nargs='*', default=[], type=int,
                       help='the timespot to change the number of steps')


# CUSTOMIZED ADVERSARIAL TRAINING ARGS
# https://arxiv.org/pdf/2002.06789.pdf
def cat_args(parser):
    group = parser.add_argument_group('Adversarial_Training', 'Arguments to configure customized adversarial training')
    group.add_argument('--inner-max', default='cat_code', type=str, choices=['cat_code', 'cat_paper'],
                       help='specify a type of inner maximazation')
    group.add_argument('--donot-use-distance-for-eps', action='store_false', dest='use_distance_for_eps',
                       help='whether use |x_adv - x| as the updated eps')
    group.add_argument('--eps', default=8, type=int, 
                       help='perturbation budget for cutomized adversarial training')
    group.add_argument('--alpha', default=2, type=int, 
                       help='step size for cutomized adversarial training')
    group.add_argument('--fixed-alpha', action='store_true',
                       help='whether use fixed step size for cutomized adversarial training')
    group.add_argument('--adapt-alpha', default=2., type=float, 
                       help='step size will be adapt_alpha times eps divided by number of steps')
    group.add_argument('--steps', default=10, type=int, 
                       help='number of steps for cutomized adversarial training')
    group.add_argument('--rs', action="store_true",
                       help='whether use random start for the attack')
    group.add_argument('--eta', default=0.005, type=float, 
                       help='epsilon scheduling parameter for cutomized adversarial training')
    group.add_argument('--c', default=10, type=int, 
                       help='weighting parameter for cutomized adversarial training')
    group.add_argument('--no-label-smoothing', action="store_false", dest="label_smoothing",
                       help='do not use label smoothing')


# FAST ADVERSARIAL TRAINING ARGS
# https://openreview.net/pdf?id=BJx040EFvH
def fast_advt_args(parser):
    group = parser.add_argument_group('Fast Adversarial_Training', 'Arguments to configure fast adversarial training')
    group.add_argument('--eps', default=8, type=int, 
                       help='perturbation budget for adversarial training')
    group.add_argument('--alpha', default=10, type=int, 
                       help='step size for adversarial training')


# WBOX EVALUATION ARGS
def wbox_eval_args(parser):
    group = parser.add_argument_group('White-box_Evaluation', 'Arguments to configure evaluation of white-box robustness')
    group.add_argument('--model-file', default=None, type=str,
                       help='Path to the model checkpoint')
    group.add_argument('--subset-num', default=0, type=int, 
                       help='number of samples of the subset, will use the full test set if zero')
    group.add_argument('--trainset', action="store_true",
                       help='use train samples or test samples')
    group.add_argument('--random-start', default=1, type=int, 
                       help='number of random starts for PGD')
    group.add_argument('--steps', default=20, type=int, 
                       help='number of steps for PGD')
    group.add_argument('--loss-fn', default='xent', type=str, choices=['xent', 'cw'],
                       help='which loss function to use')
    group.add_argument('--cw-conf', default=50., type=float,
                       help='confidence for cw loss function')
    group.add_argument('--benchmark', action='store_true',
                       help='test the robustness under eps=8/255 alpha=2/255')
    group.add_argument('--early-stop', action="store_true", 
                       help='whether jump over the following evaluation for larger epsilon when the accuracy is alreday zero')
    group.add_argument('--save-to-csv', action="store_true",
                       help='whether save the results to a csv file')
    group.add_argument('--overwrite', action="store_false", dest="append_out",
                       help='when saving results, whether use append mode')
    group.add_argument('--convergence-check', action="store_true", 
                       help='whether perform sanity check to make sure the attack converges')
    group.add_argument('--save-adv', action="store_true",
                       help='whether save the generated adversarial examples')


# FOSC EVALUATION ARGS
def fosc_on_others_eval_args(parser):
    group = parser.add_argument_group('FOSC_Evaluation', 'Arguments to configure evaluation of FOSC')
    group.add_argument('--model-file', default=None, type=str,
                       help='Path to the model checkpoint')
    group.add_argument('--data-file', default=None, type=str,
                       help='Path to the adversarial examples')
    group.add_argument('--subset-num', default=0, type=int, 
                       help='number of samples of the subset, will use the full test set if zero')
    group.add_argument('--save-to-csv', action="store_true",
                       help='whether save the results to a csv file')
    group.add_argument('--overwrite', action="store_false", dest="append_out",
                       help='when saving results, whether use append mode')


# FOSC EVALUATION ARGS
def fosc_eval_args(parser):
    group = parser.add_argument_group('FOSC_Evaluation', 'Arguments to configure evaluation of FOSC')
    group.add_argument('--model-file', default=None, type=str,
                       help='Path to the model checkpoint')
    group.add_argument('--subset-num', default=0, type=int, 
                       help='number of samples of the subset, will use the full test set if zero')
    group.add_argument('--trainset', action="store_true",
                       help='use train samples or test samples')
    group.add_argument('--save', action="store_true",
                       help='whether save the results')
    group.add_argument('--eps', default=8, type=int)
    group.add_argument('--alpha', default=2, type=int)
    group.add_argument('--steps', default=20, type=int)