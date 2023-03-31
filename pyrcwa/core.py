from numpy import pi
from .matrix import *
from .material import *


class RCWABase(object):
    def __init__(self):
        self.kx = None
        self.ky = None
        self.P = None
        self.Q = None
        self.W = None
        self.V = None
        self.Lambda = None


class RCWAFreeSpace(RCWABase):
    def __init__(self):
        super().__init__()

    def loadWaveVector(self, kx, ky):
        self.kx, self.ky = kx, ky

    def getPQMatrix(self):
        if self.kx.shape != self.ky.shape:
            raise Exception

        idt = np.mat(np.identity(self.kx.shape[0]))
        pq11 = self.kx * self.ky
        pq12 = idt - self.kx ** 2
        pq21 = self.ky ** 2 - idt
        pq22 = - self.kx * self.ky
        self.P = self.Q = np.bmat([[pq11, pq12], [pq21, pq22]])

    def solve(self, kx, ky):
        self.loadWaveVector(kx, ky)
        self.getPQMatrix()
        self.W, self.V, self.Lambda = solve_PQMatrix(self.P, self.Q)


class RCWARefSide(RCWABase):
    def __init__(self):
        super().__init__()
        self.layer = None
        self.FreeSpace = None
        self.A = None
        self.B = None
        self.SMatrix = SMatrix()

    def loadWaveVector(self, kx, ky):
        self.kx, self.ky = kx, ky

    def loadLayer(self, layer):
        self.layer = layer

    def loadFreeSpace(self, FreeSpace):
        self.FreeSpace = FreeSpace

    def getPQMatrix(self):
        if self.kx.shape != self.ky.shape:
            raise Exception

        epsilon = self.layer.epsilon
        mu = self.layer.mu

        idt = np.mat(np.identity(self.kx.shape[0]))
        pq11 = self.kx * self.ky
        pq12 = epsilon * mu * idt - self.kx ** 2
        pq21 = self.ky ** 2 - epsilon * mu * idt
        pq22 = -self.ky * self.kx
        self.P = np.bmat([[pq11, pq12], [pq21, pq22]]) / epsilon
        self.Q = np.bmat([[pq11, pq12], [pq21, pq22]]) / mu

    def solve(self, kx, ky, layer, FreeSpace):
        self.loadWaveVector(kx, ky)
        self.loadLayer(layer)
        self.loadFreeSpace(FreeSpace)
        self.getPQMatrix()
        self.W, self.V, self.Lambda = solve_PQMatrix(self.P, self.Q)

        self.A = self.FreeSpace.W.I * self.W + self.FreeSpace.V.I * self.V
        self.B = self.FreeSpace.W.I * self.W - self.FreeSpace.V.I * self.V

        s11 = - self.A.I * self.B
        s12 = 2 * self.A.I
        s21 = 0.5 * (self.A - self.B * self.A.I * self.B)
        s22 = self.B * self.A.I
        self.SMatrix.loadByPart(s11, s12, s21, s22)


class RCWATrnSide(RCWABase):
    def __init__(self):
        super().__init__()
        self.layer = None
        self.FreeSpace = None
        self.A = None
        self.B = None
        self.SMatrix = SMatrix()

    def loadWaveVector(self, kx, ky):
        self.kx, self.ky = kx, ky

    def loadLayer(self, layer):
        self.layer = layer

    def loadFreeSpace(self, FreeSpace):
        self.FreeSpace = FreeSpace

    def getPQMatrix(self):
        if self.kx.shape != self.ky.shape:
            raise Exception

        epsilon = self.layer.epsilon
        mu = self.layer.mu

        idt = np.mat(np.identity(self.kx.shape[0]))
        pq11 = self.kx * self.ky
        pq12 = epsilon * mu * idt - self.kx ** 2
        pq21 = self.ky ** 2 - epsilon * mu * idt
        pq22 = -self.ky * self.kx
        self.P = np.bmat([[pq11, pq12], [pq21, pq22]]) / epsilon
        self.Q = np.bmat([[pq11, pq12], [pq21, pq22]]) / mu

    def solve(self, kx, ky, layer, FreeSpace):
        self.loadWaveVector(kx, ky)
        self.loadLayer(layer)
        self.loadFreeSpace(FreeSpace)
        self.getPQMatrix()
        self.W, self.V, self.Lambda = solve_PQMatrix(self.P, self.Q)

        self.A = self.FreeSpace.W.I * self.W + self.FreeSpace.V.I * self.V
        self.B = self.FreeSpace.W.I * self.W - self.FreeSpace.V.I * self.V

        s11 = self.B * self.A.I
        s12 = 0.5 * (self.A - self.B * self.A.I * self.B)
        s21 = 2 * self.A.I
        s22 = - self.A.I * self.B
        self.SMatrix.loadByPart(s11, s12, s21, s22)


class RCWASingleLoop(RCWABase):
    def __init__(self):
        super().__init__()
        self.source = None
        self.layer = None
        self.FreeSpace = None
        self.A = None
        self.B = None
        self.Xi = None
        self.SMatrix = SMatrix()
        self.material_matrix_conv = None

    def loadWaveVector(self, kx, ky):
        self.kx, self.ky = kx, ky

    def loadSource(self, source):
        self.source = source

    def loadLayer(self, layer):
        self.layer = layer

    def loadFreeSpace(self, FreeSpace):
        self.FreeSpace = FreeSpace

    def loadMaterialMatrix(self, material_matrix):
        self.material_matrix_conv = material_matrix

    def getPQMatrix(self):
        if self.kx.shape != self.ky.shape:
            raise Exception

        eps_conv = self.material_matrix_conv[:, :, 0]
        mu_conv = self.material_matrix_conv[:, :, 1]
        eps_conv_inv = self.material_matrix_conv[:, :, 2]
        mu_conv_inv = self.material_matrix_conv[:, :, 3]

        p11 = self.kx * eps_conv_inv * self.ky
        p12 = mu_conv - self.kx * eps_conv_inv * self.kx
        p21 = self.ky * eps_conv_inv * self.ky - mu_conv
        p22 = -self.ky * eps_conv_inv * self.kx
        self.P = np.bmat([[p11, p12], [p21, p22]])

        q11 = self.kx * mu_conv_inv * self.ky
        q12 = eps_conv - self.kx * mu_conv_inv * self.kx
        q21 = self.ky * mu_conv_inv * self.ky - eps_conv
        q22 = -self.ky * mu_conv_inv * self.kx
        self.Q = np.bmat([[q11, q12], [q21, q22]])

    def solve(self, kx, ky, source, layer, FreeSpace, material_matrix):
        self.loadWaveVector(kx, ky)
        self.loadSource(source)
        self.loadLayer(layer)
        self.loadFreeSpace(FreeSpace)
        self.loadMaterialMatrix(material_matrix)
        self.getPQMatrix()
        self.W, self.V, self.Lambda = solve_PQMatrix(self.P, self.Q)

        A = self.A = self.W.I * self.FreeSpace.W + self.V.I * self.FreeSpace.V
        B = self.B = self.W.I * self.FreeSpace.W - self.V.I * self.FreeSpace.V
        k0 = 2 * pi / self.source.wavelength
        Xi = self.Xi = np.matrix(np.diag(np.exp(-k0 * layer.thickness * np.diagonal(self.Lambda))))

        s11 = (A - Xi * B * A.I * Xi * B).I * (Xi * B * A.I * Xi * A - B)
        s12 = (A - Xi * B * A.I * Xi * B).I * Xi * (A - B * A.I * B)
        s21 = s12
        s22 = s11
        self.SMatrix.loadByPart(s11, s12, s21, s22)


class RCWAMainLoop(object):
    def __init__(self):
        self.kx = None
        self.ky = None
        self.source = None
        self.model = None
        self.settings = None
        self.FreeSpace = None
        self.A = None
        self.B = None
        self.Xi = None
        self.SMatrix = SMatrix()
        self.material_matrix = RCWAMaterialMatrix()

    def loadRCWAModel(self, model):
        self.model = model

    def solve(self, kx, ky, source, model, FreeSpace, material_matrix):
        self.loadRCWAModel(model)
        self.material_matrix.load_material_matrix(material_matrix)
        loop = RCWASingleLoop()

        if len(self.model.layers) == 3:
            loop.solve(kx, ky, source, self.model.layers[1], FreeSpace, self.material_matrix.conv[:, :, :, 0])
            self.SMatrix.loadByTotal(loop.SMatrix.total)

        elif len(self.model.layers) > 3:
            theSMatrix = SMatrix()
            loop.solve(kx, ky, source, self.model.layers[1], FreeSpace, self.material_matrix.conv[:, :, :, 0])
            theSMatrix.loadByTotal(loop.SMatrix.total)
            idx = 1
            for ky in self.model.layers[2: -1]:
                loop.solve(kx, ky, source, self.model.layers[idx + 1], FreeSpace, self.material_matrix.conv[:, :, :, idx])
                idx += 1
                theSMatrix.loadByTotal(star_product(theSMatrix, loop.SMatrix).total)

            self.SMatrix.loadByTotal(theSMatrix.total)
