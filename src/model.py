import sys
import os
import numpy as np
import torch
import torch.nn as nn

# ************************************************ Neural Network Model ************************************************ #


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    params:
        tolerance: int, number of epochs to wait before stopping training
        min_delta: float, minimum difference between new loss and old loss to be considered as an improvement
    returns:
        None
    """
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
                
class MLP(nn.Module):
    """
    A simple multilayer perceptron
    params:
        layers: list of ints, the number of nodes in each layer
    returns:
        out: tensor, the output of the neural network
    """
    def __init__(self, layers):
        super().__init__() 
        self.layers = layers
        self.activation = nn.Tanh() #Tanh(), ReLU(), LeakyReLU(), Sigmoid(), Softplus()
        self.linears = nn.ModuleList([nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)])
        for i in range(len(self.layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0) # set weights to xavier normal
            nn.init.zeros_(self.linears[i].bias.data) # set biases to zero

    def forward(self, x):
        if torch.is_tensor(x) is not True:         
            x = torch.from_numpy(x)  
        
        act_hid_out = x.float() # input layer

        for i in range(len(self.layers)-2): # input -- penultimate layer. apply activation to all but the last layer
            hid_out = self.linears[i](act_hid_out)    
            act_hid_out = self.activation(hid_out)
        
        out = nn.functional.softplus(self.linears[-1](act_hid_out)) # ensure output is positive
        return out
    

def predict(data: np.array, layer_model: list =[6, 16, 16, 16, 3], dir: str = os.getcwd()):
    '''Predict the output 
    params:
        data: numpy array, the input data
        dir: str, the parent directory of the model
    returns:
        output: torch tensor, the output of the neural network
    '''
    device = 'cpu' # trained on cpu
    
    model_list = []
    for index in range(len(layer_model)-1):
        model = MLP(np.array(layer_model)).to(device)
        weight = torch.load(f'{dir}/model_{index}.pth')
        model.load_state_dict(weight)
        model.eval()
        model_list.append(model)
    
    output_list = []
    with torch.no_grad():
        for model in model_list:
            output = model(data)
            output_list.append(output)
    
    return output_list
