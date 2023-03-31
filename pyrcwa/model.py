class RCWALayer(object):
    def __init__(self, layer_type, **kwargs):
        if layer_type in ["Ref", "Trn"]:
            self.type = layer_type
            self.epsilon = kwargs["epsilon"]
            self.mu = kwargs["mu"]
        elif layer_type == "homogeneous":
            self.type = layer_type
            self.epsilon = kwargs["epsilon"]
            self.mu = kwargs["mu"]
            self.thickness = kwargs["thickness"]
        elif layer_type == "distribution":
            self.type = layer_type
            self.epsilon = kwargs["epsilon"]
            self.mu = kwargs["mu"]
            self.thickness = kwargs["thickness"]
        elif layer_type == "1d grating":
            self.type = layer_type
            self.epsilon1 = kwargs["epsilon1"]
            self.epsilon2 = kwargs["epsilon2"]
            self.mu1 = kwargs["mu1"]
            self.mu2 = kwargs["mu2"]
            self.thickness = kwargs["thickness"]
            self.dc = kwargs["duty_cycle"]
            self.period = kwargs["period"]


class RCWAModel(object):
    def __init__(self, layers, spatial_period):
        self.layers = layers
        self.spatial_period = spatial_period
