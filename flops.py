from ptflops import get_model_complexity_info
import torchvision.models as models
from models import *
from config import *

configs = configs()
net =  configs.net
flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
print(f"FLOPS: {flops}")
print(f"Params: {params}")