
from idiom.cc.onnx import compile

compile_dir = 'compile_dir'
onnx_file = '/models/home/auro/idiom-ml-tf2/idiom-ml-tf/examples/rn18/inference/model_4_bs16.onnx' 
batch_size = 1

compile_flags = [
    f'--onnx-define-symbol=unk__185,{batch_size}',
]

print('Compiling ONNX model...')
compile(compile_dir, onnx_file, batch_size, compile_flags)


