import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# from taming.modules.diffusionmodules.model import Encoder, Decoder
# from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
# from taming.modules.vqvae.quantize import GumbelQuantize
# from taming.modules.vqvae.quantize import EMAVectorQuantizer

from leonardo_da_vqgan.models.diffusion_models import Encoder, Decoder
from leonardo_da_vqgan.models.quantize import VectorQuantizer

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 lr=4.5e-6,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw (breadth, height, width?)
                 ):
        super().__init__()
        self.learning_rate = lr
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig) # TODO: build manually
        self.decoder = Decoder(**ddconfig) # TODO: build manually
        self.loss = instantiate_from_config(lossconfig) # TODO: build manually 
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape) # TODO: Build Manually
    
        # quantile convolutions
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        # post quantile convolutions
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        # load model weights if given
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        self.image_key = image_key

        # if flagged, color is not part of model weights
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        
        # display?
        if monitor is not None:
            self.monitor = monitor

    # load the model weights
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    # call the encoder, compute quant convolution, return quantization and embedded loss
    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    # compute the post_quant_convolution, decode the result, and return it
    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    # embed the decoded input, decode it again(?)
    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b) #TODO: Understand this by building VectorQuantizer
        dec = self.decode(quant_b)
        return dec

    # encode the input (returning quantization and loss), decode quantization, return decoded input and embedding_loss
    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    # get K'th sample from batch, reshape to contiguous float tensor
    def get_input(self, batch, k):
        x = batch
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 1, 2, 3).to(memory_format=torch.contiguous_format)
        return x.float()

    def training_step(self, batch, batch_idx, optimizer_index):
        x = self.get_input(batch, self.image_key)
        xrec , qloss = self(x) # call forward on the input image
        if optimizer_index == 0:
            # autoencode, compute auto_encode loss
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_index, self.global_step, last_layer=self.get_last_layer(), split="train")

            # log the loss (pytorch lightnight by default, redo in vanilla pytorch)
            # self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            # self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss
        
        if optimizer_index == 1:
            # discriminator loss
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_index, self.global_step,last_layer=self.get_last_layer(), split="train")

            # self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            # self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        # self.log("val/rec_loss", rec_loss,
        #            prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        # self.log("val/aeloss", aeloss,
        #            prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        # self.log_dict(log_dict_ae)
        # self.log_dict(log_dict_disc)
        
        return log_dict_ae, log_dict_disc

    # set learning rate and optimizers of whole model
    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    # return the decoder's output layer weights
    def get_last_layer(self):
        return self.decoder.conv_out.weight

    # create a dictionary of inputs and outputs
    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    # return the RGB version of an input
    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        # if we haven't already, remove colorizer weights from the model
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x