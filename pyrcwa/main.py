from source import RCWASource
from model import RCWALayer, RCWAModel
from settings import RCWASettings
from solver import RCWASolver

theSource = RCWASource(wavelength=0.5e-6, alpha=0, theta=0.0, phi=90, phase=0)
theRefLayer = RCWALayer(layer_type="Ref", epsilon=1, mu=1)
theGratingLayer = RCWALayer(
    layer_type="1d grating",
    epsilon1=1.5*1.5,
    epsilon2=1,
    mu1=1,
    mu2=1,
    thickness=0.5e-6,
    duty_cycle=0.5,
    period=1e-6
)
theTrnLayer = RCWALayer(layer_type="Trn", epsilon=1.5*1.5, mu=1)

theModel = RCWAModel(
    layers=[theRefLayer, theGratingLayer, theTrnLayer],
    spatial_period=(theGratingLayer.period, theGratingLayer.period)
)

theSettings = RCWASettings(
    harmonics=(1, 0),
    fft_resolution=(1001, 1001)
)

theRCWASolver = RCWASolver(
    source=theSource,
    model=theModel,
    settings=theSettings
)

theRCWASolver.solve_new()
theRCWASolver.show()



















