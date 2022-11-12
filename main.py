import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from modules import XRayDataModule
from modules import BaselineModule
from argparser import args
random_seed = 42
pl.seed_everything(random_seed)
models = ['resnet18']
for model in models:
    xd = XRayDataModule(args.batch_size)
    model = BaselineModule(model, 1, args.lr, args=args)
    # ddp = DDPStrategy(process_group_backend="nccl", find_unused_parameters=False)
    train_trainer = pl.Trainer(accelerator='gpu', 
                                devices=[args.gpu], 
                                # strategy=ddp, 
                                max_epochs=args.epochs,
                                reload_dataloaders_every_n_epochs=2,
                                log_every_n_steps=10,
                                check_val_every_n_epoch=2,
                                )
    train_trainer.fit(model=model, datamodule=xd)
    train_trainer.test(model=model, datamodule=xd)