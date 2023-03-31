import numpy as np


class RCWAMaterialMatrix(object):
    def __init__(self):
        self.model = None
        self.settings = None
        self.conv = None
        self.dist = None

    def loadRCWAModel(self, model):
        self.model = model

    def loadRCWASettings(self, settings):
        self.settings = settings

    def load_material_matrix(self, material_matrix):
        self.dist = material_matrix.dist
        self.conv = material_matrix.conv

    def get_convolution(self, data):
        shape = data.shape
        ft = np.fft.fft2(data)
        data_fft = np.fft.fftshift(ft) / shape[0] / shape[1]

        m, n = self.settings.harmonics
        M = np.arange(-m, m + 1, 1)
        N = np.arange(-n, n + 1, 1)
        mn = (2 * m + 1) * (2 * n + 1)
        conv_matrix = np.zeros([mn, mn], dtype=np.complex128)

        cx = round((shape[1] - 1) / 2)
        cy = round((shape[0] - 1) / 2)

        for jj in N:
            for ii in M:
                row = (jj + N[-1]) * M.size + ii + M[-1]
                for j in N:
                    for i in M:
                        col = (j + N[-1]) * M.size + i + M[-1]
                        conv_matrix[row, col] = data_fft[cy + jj - j, cx + ii - i]

        return np.matrix(conv_matrix)

    def compute_material_matrix(self):
        mn = (2 * self.settings.harmonics[0] + 1) * (2 * self.settings.harmonics[1] + 1)
        self.conv = np.zeros([mn, mn, 4, len(self.model.layers) - 2], dtype=np.complex128)
        self.dist = np.zeros([self.settings.fft_resolution[0], self.settings.fft_resolution[1], 2, len(self.model.layers) - 2], dtype=np.complex128)

        num = 0
        for layer in self.model.layers[1: -1]:
            rel_er = np.ones(self.settings.fft_resolution)
            rel_ur = np.ones(self.settings.fft_resolution)

            if layer.type == "homogeneous":
                rel_er *= layer.epsilon
                rel_ur *= layer.mu

            elif layer.type == "1d grating":
                m, n = self.settings.fft_resolution
                rel_er[:, :round(layer.dc * m)] = layer.epsilon1
                rel_er[:, round(layer.dc * m):] = layer.epsilon2
                rel_ur[:, :round(layer.dc * m)] = layer.mu1
                rel_ur[:, round(layer.dc * m):] = layer.mu2

            elif layer.type == "distribution":
                rel_er = layer.epsilon
                rel_ur = layer.mu

            self.dist[:, :, 0, num] = rel_er
            self.dist[:, :, 1, num] = rel_ur

            self.conv[:, :, 0, num] = self.get_convolution(rel_er)
            self.conv[:, :, 1, num] = self.get_convolution(rel_ur)
            self.conv[:, :, 2, num] = np.mat(self.conv[:, :, 0, num]).I
            self.conv[:, :, 3, num] = np.mat(self.conv[:, :, 1, num]).I
            num += 1

    def solve(self, model, settings):
        self.loadRCWAModel(model)
        self.loadRCWASettings(settings)
        self.compute_material_matrix()
