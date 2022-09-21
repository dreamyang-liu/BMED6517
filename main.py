import torch.nn as nn
import torch
from Backbone import Backbone
from argparser import args

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.model = Backbone(backbone=args.backbone, input_channel=args.input_channel, pretrained=args.pretrained).to(DEVICE)
        self.loss = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
    
    def train(self, train_loader, vali_loader):
        print("Training")
        for epoch in range(self.args.epochs):
            print("Epoch: ", epoch)
            for i, (x, y) in enumerate(train_loader):
                out = self.model(x.to(DEVICE))
                loss = self.loss(out, y.to(DEVICE))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            if i % 10 == 0:
                self.eval(vali_loader)
        torch.save(self.model.state_dict(), self.args.save_path)
        return 0
    
    def eval(self, test_loader, return_pred=False):
        self.model.eval()
        print("Evaluating")
        for i, (x, y) in enumerate(test_loader):
            out = self.model(x.to(DEVICE))
            loss = self.loss(out, y.to(DEVICE))
        self.model.train()
        return 0

if __name__ == "__main__":
    trainer = Trainer(args)