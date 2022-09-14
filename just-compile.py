
from idiom.cc.onnx import compile

compile_dir = 'compile_dir'
onnx_file = '/models/home/auro/idiom-ml-tf2/idiom-ml-tf/examples/rn18/inference/model_4.onnx' 

# compile_flags = [
#     f'--onnx-declare-input=input:[16, 320, 320, 3]',
# ]

compile_flags = [
    f'--onnx-define-symbol=unk__185,16',
]


batch_size = 16
print('Compiling ONNX model...')
compile(compile_dir, onnx_file, batch_size, compile_flags)
# compile(compile_dir, onnx_file, batch_size)

