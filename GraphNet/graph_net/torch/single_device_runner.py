from . import utils
import argparse
import importlib.util
import inspect
import torch
from pathlib import Path
from typing import Type, Any
import sys
from graph_net.torch.extractor import extract


def load_class_from_file(file_path: str, class_name: str) -> Type[torch.nn.Module]:
    spec = importlib.util.spec_from_file_location("unnamed", file_path)
    unnamed = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(unnamed)
    model_class = getattr(unnamed, class_name, None)
    return model_class

def main(args):
    model_path = args.model_path
    model_class = load_class_from_file(f"{model_path}/model.py", class_name="GraphModule") #加载model.py中的 GraphModule 类，并实例化模型对象
    assert model_class is not None
    model = model_class()
    print(f'{model_path=}')
    if args.enable_extract:
        assert args.extract_name is not None
        model = extract(name=args.extract_name)(model)

    # 加载权重和输入张量的描述信息, 字典中包含了 input_info:input_meta.py 和 weight_info:weght_meta.py
    inputs_params = utils.load_converted_from_text(f'{model_path}')
    params = inputs_params["weight_info"]
    state_dict = {
        k: utils.replay_tensor(v) for k, v in params.items()
    }
    
    y = model(**state_dict)[0]

    #打印输出张量的最小/最大索引和形状，便于验证模型行为。
    print("Output tensor min/max indices and shape:")
    print(torch.argmin(y), torch.argmax(y))
    print(y.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="load and run model")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to folder e.g '../../samples/torch/resnet18'")
    parser.add_argument("--enable-extract", type=bool, required=False, default=False, help="Enable extract")
    parser.add_argument("--extract-name", type=str, required=False, default=None, help="Extracted graph's name")
    args = parser.parse_args()
    main(args=args)
