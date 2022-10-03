import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from dataloader import XRayDataModule
from modules import BaselineModule

models = ['resnet18']
for model in models:
    xd = XRayDataModule(64)
    model = BaselineModule(model, 1)
    ddp = DDPStrategy(process_group_backend="nccl", find_unused_parameters=False)
    train_trainer = pl.Trainer(accelerator='gpu', 
                                devices=6, 
                                strategy=ddp, 
                                max_epochs=50, 
                                val_check_interval=0.5,
                                reload_dataloaders_every_n_epochs=2,
                                log_every_n_steps=5,
                                auto_lr_find=True,
                                )
    train_trainer.fit(model=model, datamodule=xd)
    train_trainer.test(model=model, datamodule=xd)