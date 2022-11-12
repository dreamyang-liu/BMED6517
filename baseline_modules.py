import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix

class BaselineModule(pl.LightningModule):

    def __init__(self, model, input_channel=1, lr=1e-3, pretrained=True, args=None):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        if model == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
        if model == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
        if model == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
        elif model == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
        elif model == 'resnet152':
            self.model = models.resnet152(pretrained=pretrained)
        
        self.model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.output = nn.Linear(self.model.fc.out_features, 2)
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x):
        x = self.model(x)
        x = self.output(x)
        return torch.softmax(x, dim=1)
    
    def training_step(self, batch, batch_idx): 
        x, y = batch # N * C * H * W, N * H * W
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        # return loss
        return {'loss': loss, 'y_hat': y_hat.argmax(dim=1), 'y': y.clone().detach()}
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return {'val_loss': loss, 'y_hat': y_hat.argmax(dim=1), 'y': y.clone().detach()}
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return {'test_loss': loss, 'y_hat': y_hat.argmax(dim=1), 'y': y.clone().detach()}
    
    def training_epoch_end(self, train_batch_outputs):
        loss = torch.stack([x['loss'] for x in train_batch_outputs]).mean()
        y_hat = torch.cat([x['y_hat'] for x in train_batch_outputs], dim=0).flatten()
        y = torch.cat([x['y'] for x in train_batch_outputs], dim=0).flatten()
        report = classification_report(y.cpu().numpy(), y_hat.cpu().numpy(), output_dict=True, zero_division=0)
        self.log('train_loss', loss)
        self.log('train_acc', report['accuracy'])
        self.log('train_precision', report['macro avg']['precision'])
        self.log('train_recall', report['macro avg']['recall'])
        self.log('train_f1', report['macro avg']['f1-score'])
        self.log('train_precision_w', report['weighted avg']['precision'])
        self.log('train_recall_w', report['weighted avg']['recall'])
        self.log('train_f1_w', report['weighted avg']['f1-score'])
    
    def validation_epoch_end(self, val_batch_outputs):
        loss = torch.stack([x['val_loss'] for x in val_batch_outputs]).mean()
        y_hat = torch.cat([x['y_hat'] for x in val_batch_outputs], dim=0).flatten()
        y = torch.cat([x['y'] for x in val_batch_outputs], dim=0).flatten()
        report = classification_report(y.cpu().numpy(), y_hat.cpu().numpy(), output_dict=True, zero_division=0)
        self.log('val_loss', loss)
        self.log('val_acc', report['accuracy'])
        self.log('val_precision', report['macro avg']['precision'])
        self.log('val_recall', report['macro avg']['recall'])
        self.log('val_f1', report['macro avg']['f1-score'])


    def test_epoch_end(self, test_batch_outputs):
        loss = torch.stack([x['test_loss'] for x in test_batch_outputs]).mean()
        y_hat = torch.cat([x['y_hat'] for x in test_batch_outputs], dim=0).flatten()
        y = torch.cat([x['y'] for x in test_batch_outputs], dim=0).flatten()
        print(classification_report(y.cpu().numpy(), y_hat.cpu().numpy()))
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        return optimizer
