import numpy as np
import cv2
from . import methods



class Resize(object):
    def __init__(self, dsize, interpolation=methods.INTER_LINEAR):
        self.h = dsize[0]
        self.w = dsize[1]
        self.dsize = dsize
        self.interpolation = interpolation

    def __call__(self, image):
        shape = [image.shape[0], image.shape[1]]
        if shape != self.dsize:
            image = cv2.resize(image, [self.w, self.h],
                               interpolation=self.interpolation)
        return image


class Scale(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, image):
        image = image * self.scale
        return image


class Standardization(object):
    def __init__(self, mean, std, method=methods.z_score):
        self.method = method
        self.mean = np.array(mean)
        self.std = np.array(std).astype(float)

    def __call__(self, image):
        if self.method == methods.z_score:
            image = image - self.mean
            image = image / self.std

        return image


class Flip(object):
    def __init__(self, prob=0.5, method=methods.horizontal_flip):
        self.prob = prob
        self.method = method

    def __call__(self, image):
        _prob = np.random.uniform(0.0, 1.0)
        if _prob < self.prob:
            image = cv2.flip(image, self.method)
        return image


class Brightness(object):
    def __init__(self, delta, random=True):
        self.delta = delta
        self.random = random

    def __call__(self, image):
        image = image.astype(np.float32)

        if self.random:
            assert self.delta >= 0
            delta = np.random.uniform(-self.delta, self.delta)
        else:
            delta = self.delta

        image = np.clip((image + delta), 0.0, 255.0).astype(np.uint8)
        return image


class Contrast(object):
    def __init__(self, lower=None, upper=None, delta=None, random=True):
        self.lower = lower
        self.upper = upper
        self.delta = delta
        self.random = random

    def __call__(self, image):
        image = image.astype(np.float32)

        assert self.delta >= 0
        if self.random:
            delta = np.random.uniform(self.lower, self.upper)
        else:
            delta = self.delta

        image = np.clip((image * delta), 0.0, 255.0).astype(np.uint8)
        return image


class Saturation(object):
    def __init__(self, lower=None, upper=None, delta=None, random=True):
        self.lower = lower
        self.upper = upper
        self.delta = delta
        self.random = random

    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = image.astype(np.float32)

        assert self.delta >= 0
        if self.random:
            delta = np.random.uniform(self.lower, self.upper)
        else:
            delta = self.delta

        image[:, :, 1] = image[:, :, 1] * delta
        image[image > 255] = 255
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image


class Hue(object):
    def __init__(self, delta, random=True):
        self.delta = delta
        self.random = random

    def __call__(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = image.astype(np.float32)

        if self.random:
            assert self.delta >= 0
            delta = np.random.randint(-self.delta, self.delta)
        else:
            delta = self.delta

        image[:, :, 0] = (image[:, :, 0] + delta) % 180
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image


class ReorderChannels(object):
    def __init__(self, channels=None, random=True):
        self.channels = channels
        self.random = random

    def __call__(self, image):
        if self.random:
            channels = [0, 1, 2]
            np.random.shuffle(channels)
        else:
            channels = self.channels

        image = image[:, :, channels]
        return image


class Crop(object):
    def __init__(self, ratio=0.9, prob=0.5):
        self.ratio = ratio
        self.prob = prob

    def __call__(self, image):
        _prob = np.random.uniform(0.0, 1.0)
        if _prob < self.prob:
            h = image.shape[0]
            w = image.shape[1]
            h_ratio = np.random.uniform(self.ratio, 1.0)
            w_ratio = np.random.uniform(self.ratio, 1.0)
            new_h = int(h * h_ratio)
            new_w = int(w * w_ratio)

            if new_h == h and new_w == w:
                return image

            top  = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            image = image[top: top + new_h,
                          left: left + new_w]
            return image
        else:
            return image


def mean(image, batch=False):
    if batch:
        _mean = np.mean(image, axis=(0, 1, 2))
    else:
        _mean = np.mean(image, axis=(0, 1))
    return _mean

def std(image, batch=False):
    if batch:
        _std = np.std(image, axis=(0, 1, 2))
    else:
        _std = np.std(image, axis=(0, 1))
    return _std

def meanStd(image, batch=False):
    _mean = mean(image, batch)
    _std = std(image, batch)
    return _mean, _std

