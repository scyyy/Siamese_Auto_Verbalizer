import torch.nn as nn
import torch.nn.functional as F
import torch


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, anchor, positive, negative):
        # print(anchor.shape, positive.shape, negative.shape)

        output1 = self.embedding_net(anchor)
        output2 = self.embedding_net(positive)
        output3 = self.embedding_net(negative)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()

        self.fc = nn.Sequential(nn.Linear(300, 100),
                                nn.PReLU(),
                                nn.Linear(100, 50))

        # self.fc1 = nn.Linear(300, 100)
        # self.fc2 = nn.Linear(100, 50)

    def forward(self, x):
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = nn.PReLU(x)
        # output = self.fc2(x)
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class TripletCosLoss(nn.Module):
    def __init__(self, margin):
        super(TripletCosLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        sim = torch.nn.CosineSimilarity()
        distance_positive = 1 - sim(anchor, positive)
        distance_negative = 1 - sim(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()