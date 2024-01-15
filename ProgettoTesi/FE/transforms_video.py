import numbers
from FE import functional_video as F


class ResizeVideo:
    def __init__(self, size, interpolation_mode="bilinear"):
        self.size = size
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        return F.resize(clip, self.size, self.interpolation_mode)


class CenterCropVideo:
    def __init__(self, crop_size):
        if isinstance(crop_size, numbers.Number):
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            self.crop_size = crop_size

    def __call__(self, clip):
        return F.center_crop(clip, self.crop_size)

    def __repr__(self):
        return self.__class__.__name__ + f"(crop_size={self.crop_size})"


class NormalizeVideo:
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        return F.normalize(clip, self.mean, self.std, self.inplace)

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(mean={self.mean}, std={self.std}, inplace={self.inplace})"
        )


class ToTensorVideo:
    def __init__(self):
        pass

    def __call__(self, clip):
        return F.to_tensor(clip)

    def __repr__(self):
        return self.__class__.__name__



