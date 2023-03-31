from .core import *
from numpy import cos, sin


class RCWASolver(object):
	def __init__(self, source, model, settings):
		self.source = self.loadSource(source)
		self.model = self.loadModel(model)
		self.settings = self.loadSettings(settings)
		self.Kx = self.Ky = self.Kz = self.K = self.KRef = self.KTrn = None
		self.FreeSpace = RCWAFreeSpace()
		self.RefSide = RCWARefSide()
		self.TrnSide = RCWATrnSide()
		self.MaterialMatrix = RCWAMaterialMatrix()
		self.MainLoop = RCWAMainLoop()
		self.SMatrix = SMatrix()
		self.results = self.init_results()

	def loadSource(self, source):
		return source

	def loadModel(self, model):
		self._check_layer(model.layers)
		return model

	def loadSettings(self, settings):
		return settings

	@staticmethod
	def init_results():
		return {
			"R": np.NAN,
			"T": np.NAN,
			"Total R": np.NAN,
			"Total T": np.NAN,
			"Total Energy": np.NAN
		}

	@staticmethod
	def _check_layer(layers):
		if layers[0].type != "Ref":
			raise Exception
		elif layers[-1].type != "Trn":
			raise Exception
		else:
			ly_id = 1
			for ly in layers[1: -1]:
				if ly.type not in ["homogeneous", "1d grating", "distribution"]:
					raise Exception("{}".format(ly_id))
				ly_id += 1

	def compute_wave_vector(self):
		wavelength = self.source.wavelength
		layers = self.model.layers
		spatial_period = self.model.spatial_period
		harmonics = self.settings.harmonics
		k0 = 2 * pi / wavelength
		K = k0 * np.sqrt(layers[0].epsilon) * np.array(
			[
				sin(self.source.alpha) * cos(self.source.theta),
				sin(self.source.alpha) * sin(self.source.theta),
				cos(self.source.alpha)
			]
		)
		m, n = harmonics
		M = np.arange(-m, m + 1, 1)
		N = np.arange(-n, n + 1, 1)
		LAMBDA_x, LAMBDA_y = spatial_period
		kx = K[0] - 2 * pi / LAMBDA_x * M
		ky = K[1] - 2 * pi / LAMBDA_y * N

		kx2d, ky2d = np.meshgrid(kx, ky, indexing='xy')
		kz2d = np.conj(np.lib.scimath.sqrt(k0**2-kx2d**2-ky2d**2))
		kRef2d = -np.conj(np.lib.scimath.sqrt(k0**2*layers[0].epsilon-kx2d**2-ky2d**2))
		kTrn2d = np.conj(np.lib.scimath.sqrt(k0**2*layers[-1].epsilon-kx2d**2-ky2d**2))
		self.KRef = np.matrix(np.diag(np.concatenate(kRef2d.reshape((-1, 1), order='C')) / k0))
		self.KTrn = np.matrix(np.diag(np.concatenate(kTrn2d.reshape((-1, 1), order='C')) / k0))
		self.Kx = np.matrix(np.diag(np.concatenate(kx2d.reshape((-1, 1), order='C')) / k0))
		self.Ky = np.matrix(np.diag(np.concatenate(ky2d.reshape((-1, 1), order='C')) / k0))
		self.Kz = np.matrix(np.diag(np.concatenate(kz2d.reshape((-1, 1), order='C')) / k0))
		self.K = K

	def solveFreeSpace(self):
		self.FreeSpace.solve(self.Kx, self.Ky)

	def solveRefSide(self):
		self.RefSide.solve(
			kx=self.Kx,
			ky=self.Ky,
			layer=self.model.layers[0],
			FreeSpace=self.FreeSpace
		)

	def solveTrnSide(self):
		self.TrnSide.solve(
			kx=self.Kx,
			ky=self.Ky,
			layer=self.model.layers[-1],
			FreeSpace=self.FreeSpace
		)

	def solveMaterialMatrix(self):
		self.MaterialMatrix.solve(
			model=self.model,
			settings=self.settings
		)

	def solveMainLoop(self):
		self.MainLoop.solve(
			kx=self.Kx,
			ky=self.Ky,
			source=self.source,
			model=self.model,
			FreeSpace=self.FreeSpace,
			material_matrix=self.MaterialMatrix
		)

	def getTotalSMatrix(self):
		MA = self.RefSide.SMatrix
		MB = self.MainLoop.SMatrix
		MC = self.TrnSide.SMatrix
		self.SMatrix.loadByTotal(star_product(star_product(MA, MB), MC).total)

	def getDiffractionEfficiency(self):
		self.compute_diffraction_efficiency(
			WRef=self.RefSide.W,
			WTrn=self.TrnSide.W,
			SMatrix_Total=self.SMatrix
		)

	def solve_new(self):
		self.compute_wave_vector()
		self.solveFreeSpace()
		self.solveRefSide()
		self.solveTrnSide()
		self.solveMaterialMatrix()
		self.solveMainLoop()
		self.getTotalSMatrix()
		self.getDiffractionEfficiency()

	def load_material_matrix(self, material_matrix):
		self.MaterialMatrix.load_material_matrix(material_matrix)

	def solve_load_material_matrix(self, material_matrix):
		self.compute_wave_vector()
		self.solveFreeSpace()
		self.solveRefSide()
		self.solveTrnSide()
		self.load_material_matrix(material_matrix)
		self.solveMainLoop()
		self.getTotalSMatrix()
		self.getDiffractionEfficiency()

	def compute_diffraction_efficiency(self, WRef, WTrn, SMatrix_Total):
		alpha = self.source.alpha
		theta = self.source.theta
		phi = self.source.phi
		phase = self.source.phase

		ate = np.array([sin(theta), -cos(theta), 0])
		atm = np.array([-cos(alpha)*cos(theta), -cos(alpha)*sin(theta), sin(alpha)])
		EP = ate * sin(phi) + atm * cos(phi) * np.exp(1j*phase)
		Kx = self.Kx
		Ky = self.Ky
		KRef = self.KRef
		KTrn = self.KTrn
		k0 = 2 * pi / self.source.wavelength
		kz = self.K[2] / k0
		mu_ref = self.model.layers[0].mu
		mu_trn = self.model.layers[-1].mu
		delta = np.zeros([np.shape(Kx)[0], 1])
		delta[round((np.shape(Kx)[0] - 1) / 2), 0] = 1
		s_inc = np.r_[EP[0]*delta, EP[1]*delta]
		c_inc = WRef.I*s_inc
		c_ref = SMatrix_Total.s11*c_inc
		c_trn = SMatrix_Total.s21*c_inc
		r_T = WRef * c_ref
		t_T = WTrn * c_trn
		r_X = r_T[:np.shape(Kx)[0]]
		r_Y = r_T[np.shape(Kx)[0]:]
		r_Z = -np.matrix(self.KRef).I * (Kx * r_X + Ky * r_Y)
		t_X = t_T[:np.shape(Kx)[0]]
		t_Y = t_T[np.shape(Kx)[0]:]
		t_Z = -np.matrix(self.KTrn).I * (Kx * t_X + Ky * t_Y)

		r2 = np.abs(r_X.A) ** 2 + np.abs(r_Y.A) ** 2 + np.abs(r_Z.A) ** 2
		t2 = np.abs(t_X.A) ** 2 + np.abs(t_Y.A) ** 2 + np.abs(t_Z.A) ** 2

		self.results["R"] = np.abs(np.real(KRef/kz)*r2)
		self.results["T"] = np.abs(mu_ref/mu_trn*np.real(KTrn/kz)*t2)
		self.results["Total R"] = np.sum(self.results["R"])
		self.results["Total T"] = np.sum(self.results["T"])
		self.results["Total Energy"] = self.results["Total R"] + self.results["Total T"]
		return

	def show(self):
		layer_info = ""
		num = 0
		for ly in self.model.layers:
			layer_info += "Layer{}: {}\n".format(num, ly.type)
			num += 1
		print(
			"*****************************************\n"
			"*************RCWA Solver Result**********\n"
			"*****************************************\n"
			"-----------------------------------------\n"
			"1. Source settings: \n" +
			"wavelength: {:.2f} nm\n".format(self.source.wavelength*1e9) +
			"alpha: {:.2f} deg.\n".format(self.source.alpha/pi*180) +
			"theta: {:.2f} deg.\n".format(self.source.theta/pi*180) +
			"phi: {:.2f} deg.\n".format(self.source.phi/pi*180) +
			"phase: {:.2f} deg.\n".format(self.source.phase/pi*180) +
			"-----------------------------------------\n"
			"2. Layers settings: \n" +
			layer_info +
			"-----------------------------------------\n"
			"3. Results: \n" +
			"R: {}\n".format(self.results["R"]) +
			"T: {}\n".format(self.results["T"]) +
			"Total R: {:.4f}%\n".format(self.results["Total R"]*100) +
			"Total T: {:.4f}%\n".format(self.results["Total T"]*100) +
			"Total Energy: {:.2f}%\n".format(self.results["Total Energy"]*100)
		)
