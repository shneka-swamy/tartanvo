from TartanVO import TartanVO
import argparse
from pathlib import Path

import torch
import torchvision

def commandParser():
    parser = argparse.ArgumentParser(description='TartanVO Exporter')
    parser.add_argument('--model-name', default='',
                        help='name of pretrained model')
    parser.add_argument('--export-dir', default='', type=Path,
                        help='export directory')
    
    args = parser.parse_args()
    return args

def main(args):
    modelHolder = TartanVO(args.model_name)
    example = [[(1, 3, 448, 640), (1, 3, 448, 640), (1, 2, 112, 160)]]
    exampleTensors = torch.rand(example[0][0]), torch.rand(example[0][1]), torch.rand(example[0][2])
    exampleTensors = exampleTensors[0].cuda(), exampleTensors[1].cuda(), exampleTensors[2].cuda()
    traced_script_module = modelHolder.onnx_model(exampleTensors)
    print("finished exporting model")

    args.export_dir.mkdir(parents=True, exist_ok=True)

    traced_script_module.save(str(args.export_dir / 'model.pt'))

    # load the model from the file to verify it worked
    loaded = torch.jit.load(args.export_dir / 'model.pt')
    print(loaded)
    

if __name__ == '__main__':
    main(commandParser())