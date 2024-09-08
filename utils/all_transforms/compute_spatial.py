

class ComputeSpatial(object):

    def __init__(self, network):
        self.network = network
    
    def __call__(self, data):
       return self.network.forward_spatial(data)