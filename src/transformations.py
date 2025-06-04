import numpy as np


class RandomRotate:
    def __init__(self, max_angle=2 * np.pi):
        self.max_angle = max_angle

    def __call__(self, points):
        theta = np.random.uniform(0, self.max_angle)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        return points @ rot


class RandomTranslate:
    def __init__(self, range_val=1.0):
        self.range_val = range_val

    def __call__(self, points):
        shift = np.random.uniform(-self.range_val, self.range_val, size=(1, 2)).astype(np.float32)
        return points + shift


class RandomPermute:
    def __call__(self, points):
        perm = np.random.permutation(len(points))
        return points[perm]
