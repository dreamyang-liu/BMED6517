import pytorch_lightning as pl
import torch.nn as nn
from torchvision import models

class CondensationModule(pl.Module):

    def __int(self, sample_per_class=100, model='resnet50', **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.sample_per_class = sample_per_class
        self.model = model
        self.kwargs = kwargs
        self.model = self.model(**self.kwargs)