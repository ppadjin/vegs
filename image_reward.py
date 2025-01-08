# pip install image-reward
import ImageReward as RM
model = RM.load("ImageReward-v1.0")

#rewards = model.score("an image of the street and the car in front", ["/workspace/vegs/0000004109.png"])

#print(rewards)

import torch

a = torch.tensor([1, 2, 3]).float()

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(3, 64*64*3)

    def forward(self, x):
        return self.fc(x).view(-1, 3, 64, 64)

simple_model = SimpleModel()
res = simple_model(a)

reward = model.score("an image of the street and the car in front", res)

print(reward)

