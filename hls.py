import hls4ml
import torch
import pydot, graphviz
from model import FC_Net
from torchinfo import summary

model = torch.load('models/fc_trained_model.pth', map_location='cpu')
# print(summary(model, (1,1,28,28)))

config = hls4ml.utils.config.config_from_pytorch_model(model)
# # print(config)
# print("-----------------------------------")
# print("Configuration")
# print_dict(config)
# print("-----------------------------------")
hls_model = hls4ml.converters.convert_from_pytorch_model(model,
                                                        input_shape = (1,1,28,28),
                                                        hls_config=config,
                                                        output_dir='hls4ml_output',
                                                        fpga_part='xcu250-figd2104-2L-e')

# hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)
hls_model.compile()
