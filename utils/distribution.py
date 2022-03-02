import functools
import torch

def get_distribution(data, indices):
        def prod(x): return functools.reduce(lambda a, b: a * b, x)
        shape = data[0][0].shape
        pixels = prod(shape[1:])
        print('Calculating Mean...')
        mean_unscaled = torch.zeros(shape[0])
        for index in indices:
            x = data[index][0].flatten(1)
            mean_unscaled += torch.sum(x, 1)
        mean = mean_unscaled / (len(data) * pixels)
        print('Calculating Std...')
        std_unscaled = torch.zeros_like(mean)
        for index in indices:
            x = data[index][0].flatten(1)
            std_unscaled += torch.sum(torch.square(x -
                                      mean.view(shape[0], 1)), 1)
        std = torch.sqrt(std_unscaled / (len(data) * pixels))
        mean = mean.item() if prod(mean.shape) == 1 else mean
        std = std.item() if prod(std.shape) == 1 else std
        print(f'Mean: {mean}, Std: {std}')
        return mean, std