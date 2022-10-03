
import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl
from torchvision import models
from sklearn.metrics import f1_score, accuracy_score

class BaselineModule(pl.LightningModule):

    def __init__(self, model, input_channel=3, pretrained=True):
        super().__init__()
        self.save_hyperparameters()
        if model == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
        if model == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
        elif model == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
        elif model == 'resnet152':
            self.model = models.resnet152(pretrained=pretrained)
        self.model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.output = nn.Linear(self.model.fc.out_features, 2)
    
    def forward(self, x):
        x = self.model(x)
        x = self.output(x)
        return torch.softmax(x, dim=1)
    
    def training_step(self, batch, batch_idx): 
        x, y = batch
        y_hat = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        accuracy = accuracy_score(y_hat.argmax(dim=1).cpu(), y.cpu())
        f1score = f1_score(y_hat.argmax(dim=1).cpu(), y.cpu())
        # self.log("val_loss", loss, prog_bar=True)
        # self.log("val_accuracy", accuracy, prog_bar=True)
        # self.log("val_f1_score", f1score, prog_bar=True)
        return {"val_loss": loss, "accuracy": accuracy, "f1_score": f1score}
    
    def validation_epoch_end(self, val_batch_outputs):
        avg_loss = torch.stack([x['val_loss'] for x in val_batch_outputs]).mean()
        avg_accuracy = torch.tensor([x['accuracy'] for x in val_batch_outputs]).mean()
        avg_f1_score = torch.tensor([x['f1_score'] for x in val_batch_outputs]).mean()
        self.log("final_val_loss", avg_loss, sync_dist=True)
        self.log("final_val_accuracy", avg_accuracy, sync_dist=True)
        self.log("final_val_f1_score", avg_f1_score, sync_dist=True)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        y_hat = self.forward(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        accuracy = accuracy_score(y_hat.argmax(dim=1).cpu(), y.cpu())
        f1score = f1_score(y_hat.argmax(dim=1).cpu(), y.cpu())
        return {"test_loss": loss, "accuracy": accuracy, "f1_score": f1score}
    
    def test_epoch_end(self, test_batch_outputs):
        avg_loss = torch.stack([x['test_loss'] for x in test_batch_outputs]).mean()
        avg_accuracy = torch.tensor([x['accuracy'] for x in test_batch_outputs]).mean()
        avg_f1_score = torch.tensor([x['f1_score'] for x in test_batch_outputs]).mean()
        self.log("test_loss", avg_loss)
        self.log("final_accuracy", avg_accuracy)
        self.log("final_f1_score", avg_f1_score)
        # all_preds = torch.stack(test_batch_outpus)
        # print(all_preds)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer
