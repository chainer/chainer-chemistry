from chainer import cuda


class GaussianNoiseSampler(object):
    """Default noise sampler class to calculate SmoothGrad"""

    def __init__(self, mode='relative', scale=0.15):
        self.mode = mode
        self.scale = scale

    def sample(self, target_array):
        xp = cuda.get_array_module(target_array)
        noise = xp.random.normal(
            0, self.scale, target_array.shape)
        if self.mode == 'absolute':
            # `scale` is used as is
            pass
        elif self.mode == 'relative':
            # `scale_axis` is used to calculate `max` and `min` of target_array
            # As default, all axes except batch axis are used.
            scale_axis = tuple(range(1, target_array.ndim))
            vmax = xp.max(target_array, axis=scale_axis, keepdims=True)
            vmin = xp.min(target_array, axis=scale_axis, keepdims=True)
            noise = noise * (vmax - vmin)
        else:
            raise ValueError("[ERROR] Unexpected value mode={}"
                             .format(self.mode))
        return noise
