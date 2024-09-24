import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import ImageNetVidDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import argparse

# Configs
resume_path = 'models/v1-5-pruned_controlnet.ckpt'
batch_size = 4
logger_freq = 100
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = ImageNetVidDataset(path= "/data/nak168/spatial_temporal/stream_img/data/fpe-westbrook/", path_weather= "/data/nak168/spatial_temporal/stream_img/data/", len_seq=1)
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(accelerator="gpu", precision=32, callbacks=[logger], devices=[1], max_epochs=200, default_root_dir="ckpt")
# Train!
trainer.fit(model, dataloader)

# dataset = ImageNetVidDataset(path= "/data/nak168/spatial_temporal/stream_img/data/fpe-westbrook/", path_weather= "/data/nak168/spatial_temporal/stream_img/data/", len_seq=1, phase="test")
# dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
# logger = ImageLogger(batch_frequency=1)
# trainer = pl.Trainer(accelerator="gpu", precision=32, callbacks=[logger], devices=[1], max_epochs=1, default_root_dir="ckpt")
# # Test
# trainer.test(model, dataloaders=dataloader)
