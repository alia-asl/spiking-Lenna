import torch
from PIL import Image
import numpy as np
from conex.helpers.filters import DoGFilter, GaborFilter
from conex.helpers.transforms import Conv2dFilter
from encoding import TTFSEncoder

from typing import Literal

class Processor:
    def load(self, image_name) -> None:
        """
        Params:
        -----
        `image`: the path to the image
        """
        image = Image.open(f"images/{image_name}.tif")
        self.image = torch.tensor(np.array(image)).unsqueeze(0).type(torch.float32)
    
    def get_filter(self, filter_name:Literal['dog', 'gabor'], params):
        filter = {"gabor": GaborFilter, "dog": DoGFilter}[filter_name]
        the_filter = filter(**params).reshape(1, 1,  params['size'],  params['size'])
        return the_filter
    
    def apply_filter(self, filter_name:Literal['dog', 'gabor'], params):
        conved = Conv2dFilter(self.get_filter(filter_name, params))(self.image).squeeze()
        return conved
    
    def apply_encoding(self, encoder_name:Literal['ttfs'], encoder_params, filter_name:Literal['dog', 'gabor'], filter_params):
        encoder = {'ttfs': TTFSEncoder}[encoder_name](**encoder_params)
        encoded:torch.Tensor = encoder(self.apply_filter(filter_name=filter_name, params=filter_params))
        return encoded


