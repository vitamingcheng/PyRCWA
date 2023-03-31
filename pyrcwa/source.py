from .utils import deg2rad


class RCWASource(object):
    """
        Class for defining monochromatic excitation source
        :param wavelength: The wavelength, in meters
        :param alpha: Angle with respect to the vector normal to the layer stack, in radians
        :param theta: Rotation angle amount the vector normal to the layer stack
        :param phi: Angle with respect to the vector normal to the layer stack, in radians
        :param phase: Phase difference between the TE/TM polarization vectors
        """
    def __init__(self, wavelength, alpha, theta, phi, phase):
        self.wavelength = wavelength
        self.alpha = deg2rad(alpha)
        self.theta = deg2rad(theta)
        self.phi = deg2rad(phi)
        self.phase = deg2rad(phase)
    #
    # @property
    # def wavelength(self):
    #     return self._wavelength
    #
    # @wavelength.setter
    # def wavelength(self, _wavelength):
    #     self._wavelength = _wavelength
    #
    # @wavelength.getter
    # def wavelength(self):
    #     return self._wavelength
