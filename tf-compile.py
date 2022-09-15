from idiom.cc.onnx import compile
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Compile the ONNX model for Envise')
    parser.add_argument('--onnx-path',
                        type=str, 
                        help='Path to ResNet18 model ONNX file', required=True)
    args = parser.parse_args()

    return args


compile_dir = 'compile_dir'
# onnx_path = '/models/home/auro/idiom-ml-tf2/idiom-ml-tf/examples/rn18/inference/model_4_bs16.onnx' 
batch_size = 1

compile_flags = [
    f'--onnx-define-symbol=unk__185,{batch_size}',
]


args = parse_arguments()
print('Compiling ONNX model...')
compile(compile_dir, args.onnx_path, batch_size, compile_flags)


