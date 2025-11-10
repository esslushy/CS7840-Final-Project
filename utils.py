import torch
from MI import info_nce
import numpy as np

def equiv_error_calc(net, images):
    images_rot90 = torch.rot90(images, dims=(-2, -1))
    images_rot180 = torch.rot90(images_rot90, dims=(-2, -1))
    images_rot270 = torch.rot90(images_rot180, dims=(-2, -1))
    net_errors = []
    
    with torch.inference_mode():
        output, _, layers = net(images)
        output_rot90, _, layers_rot90 = net(images_rot90)
        output_rot180, _, layers_rot180 = net(images_rot180)
        output_rot270, _, layers_rot270 = net(images_rot270)

    for rep, rep90, rep180, rep270 in zip(layers, layers_rot90, layers_rot180, layers_rot270):
        baseline = info_nce(rep, rep)
        net_errors.append(np.mean([
            (info_nce(rep90, rep) - baseline).item(),
            (info_nce(rep180, rep) - baseline).item(),
            (info_nce(rep270, rep) - baseline).item(),
        ]))
    return net_errors