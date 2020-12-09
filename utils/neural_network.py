import numpy as np

class Net(object):
  
    def __init__(self, nInput, nHidden, nOutput):
        super(Net, self).__init__()
        self.nInput = nInput
        self.nHidden = nHidden
        self.nOutput = nOutput
        self.nVariables = nInput*nHidden + nHidden*nOutput
        self.fc1 = None
        self.fc2 = None

    def feedforward(self, x):
        x = np.tanh(np.matmul(self.fc1, x))
        x = np.tanh(np.matmul(self.fc2, x))
        return x

    def init_weight(self, weights):
        if weights.size != self.nVariables:
            raise ValueError('Error length of variables!')
        self.fc1 = weights[:self.nInput*self.nHidden].reshape(self.nHidden, self.nInput)
        self.fc2 = weights[self.nInput*self.nHidden:].reshape(self.nOutput, self.nHidden)

    def get_nVariables(self):
        return self.nVariables

    def evaluate(self, cart, sLen):
        cart.__init__(sLen)
        while True:
            state = cart.get_state()
            cart.applied_force = 10*self.feedforward(state)
            cart.update_state()
            cart.update_state()
            if cart.failed:
                return cart.time
            elif (cart.time - 2000) > -0.00001:
                return cart.time