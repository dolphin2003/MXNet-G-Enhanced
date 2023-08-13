from load_model import load_checkpoint
from save_model import save_checkpoint


def combine_model(prefix1, epoch1, prefix2, epoch2, prefix_out, epoch_out):
    args1, auxs1 = load_checkpoint(prefix1, epoch1)
    args2, auxs2 = load_checkpoint(prefix2, epoch2)
    arg_names = args1.keys() + args2.keys()
    aux_names = auxs1.keys() + auxs2.keys()
    args = dict()
    for arg in arg_names:
        