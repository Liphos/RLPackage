"""Model utilitary class"""
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """DQN Network"""
    def __init__(self, observation_shape:int, action_shape:int, embed_size:int, output_activation:str="linear"):
        super(MLP, self).__init__()
        self.layers = []
        self.dense1 = nn.Linear(observation_shape, embed_size)
        self.layers.append(self.dense1)
        self.dense2 = nn.Linear(embed_size, action_shape)
        self.layers.append(self.dense2)
        self.activation = F.relu
        if output_activation == "relu":
            self.output_activation = F.relu #type:Callable
        elif output_activation == "softmax":
            self.output_activation = F.softmax
        elif output_activation == "linear":
            self.output_activation = lambda x:x
        else:
            raise ValueError("Unknwon activation function")

    def forward(self, x:torch.Tensor):
        """return values"""
        x = self.activation(self.dense1(x))
        x = self.output_activation(self.dense2(x))
        return x

    def save_txt(self, filename:str):
        """Save the layers in a txt

        Args:
            filename (str): path to the txt file
        """
        with open(filename, 'w', encoding="utf-8") as file:
            for layer in self.layers:
                file.write(str(layer._get_name) + "\n")
        file.close()

if __name__ == '__main__':
    model = MLP(1, 2)
    model(torch.tensor([]))
