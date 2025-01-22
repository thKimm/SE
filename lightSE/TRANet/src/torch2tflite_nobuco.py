import nobuco
from nobuco import ChannelOrder, ChannelOrderingStrategy
from nobuco.layers.weight import WeightLayer

import torch
from torch import nn

import tensorflow as tf
from tensorflow.lite.python.lite import TFLiteConverter
import keras

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import torchaudio
import glob
import numpy as np
import librosa as rs

from utils.hparams import HParam

from common_porting import get_model
# def update_state_dict_keys(old_state_dict, hp): # TRUNet
#     new_state_dict = {}
#     for old_key in old_state_dict:
#         if old_key.startswith("helper.enc"):
#             try :
#                 number = int(old_key[10])
#             except : 
#                 number = int(old_key[11]) + 1
#             if number == 1:
#                 new_state_dict[old_key] = old_state_dict[old_key]
#                 continue
#         if 'conv_depth.weight' in old_key:
#             big_tensor = old_state_dict[old_key]
#             for i in range(big_tensor.size(0)):
#                 # Determine new_key for depthwise convolutions
#                 new_key1 = old_key.replace(f'enc{number}.conv_depth.weight', f'enc{number}.conv_depth.{i}.weight')
#                 new_key2 = old_key.replace(f'enc.{number-1}.conv_depth.weight', f'enc.{number-1}.conv_depth.{i}.weight')
#                 new_state_dict[new_key1] = big_tensor[i:i+1, :, :, :]
#                 new_state_dict[new_key2] = big_tensor[i:i+1, :, :, :]
#         elif 'conv_depth.bias' in old_key:
#             big_tensor = old_state_dict[old_key]
#             for i in range(big_tensor.size(0)):
#                 # Determine new_key for depthwise convolutions
#                 new_key1 = old_key.replace(f'enc{number}.conv_depth.bias', f'enc{number}.conv_depth.{i}.bias')
#                 new_key2 = old_key.replace(f'enc.{number-1}.conv_depth.bias', f'enc.{number-1}.conv_depth.{i}.bias')
#                 new_state_dict[new_key1] = big_tensor[i:i+1]
#                 new_state_dict[new_key2] = big_tensor[i:i+1]
#         else:
#             # Copy other parameters as they are
#             new_state_dict[old_key] = old_state_dict[old_key]
#     return new_state_dict
def update_state_dict_keys(old_state_dict, hp): # MappingNet
    new_state_dict = {}
    for old_key in old_state_dict:
        # if 'FBlocks' in old_key and 'conv' in old_key and 'weight' in old_key:
        if 'FBlocks' in old_key and 'conv' in old_key:
            WandB = old_key.split('.')[-1]
            big_tensor = old_state_dict[old_key]
            groups = hp.model.architecture["freq_modeling"]["conv1"]["groups"]
            groups_in = hp.model.architecture["freq_modeling"]["conv1"]["in_channels"] // groups
        #     for i in range(groups):
        #         # Determine new_key for depthwise convolutions
        #         new_wight = old_key.replace('.weight', f'.{i}.weight')
        #         new_state_dict[new_wight] = big_tensor[i*groups_in:(i+1)*groups_in]
        # elif 'FBlocks' in old_key and 'conv' in old_key and 'bias' in old_key:
            # big_tensor = old_state_dict[old_key]
            # groups = hp.model.architecture["freq_modeling"]["conv1"]["groups"]
            # groups_in = hp.model.architecture["freq_modeling"]["conv1"]["in_channels"] // groups
            for i in range(groups):
                # Determine new_key for depthwise convolutions
                new_bias = old_key.replace(f'.{WandB}', f'.{i}.{WandB}')
                new_state_dict[new_bias] = big_tensor[i*groups_in:(i+1)*groups_in]
        else:
            # Copy other parameters as they are
            new_state_dict[old_key] = old_state_dict[old_key]
    return new_state_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--default', type=str, required=True,
                        help="default configuration")
    parser.add_argument('--version_name', '-v', type=str, required=True,
                        help="version of current training")
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--name',type=str,required=False,default='test',
                        help="name of tf-lite model")
    args = parser.parse_args()
    
    # hp = HParam(args.config,args.default,merge_except=["architecture"])
    hp = HParam(args.config)
    print("NOTE::Loading configuration {} based on {}".format(args.config,args.default))
    
    device = torch.device("cpu")
    version = args.version_name
    # torch.cuda.set_device(device)
    
    ## load
    model = get_model(hp,device=device)
    # if args.chkpt is None:
    #     modelsave_path = hp.log.root +'/'+'chkpt' + '/' + version
    #     chkpt = modelsave_path + '/bestmodel.pt'
    # else :
    #     chkpt = args.chkpt
    # print('NOTE::Loading pre-trained model : '+ chkpt)
    # try : 
    #     state_dict = torch.load(chkpt, map_location=device)["model"]
    # except KeyError :
    #     state_dict = torch.load(chkpt, map_location=device)
    # try : 
    #     model.load_state_dict(state_dict,strict=True)
    # except :
    #     state_dict = update_state_dict_keys(state_dict,hp)
    #     model.load_state_dict(state_dict,strict=True)
    # h0 = torch.rand(1,3,257,64).to(device)
    h1 = torch.rand(1,257,64).to(device)
    h2 = torch.rand(1,257,64).to(device)
    h3 = torch.rand(1,257,64).to(device)
    h4 = torch.rand(1,257,64).to(device)
    h5 = torch.rand(1,257,64).to(device)
    h6 = torch.rand(1,257,64).to(device)
    x = torch.randn(1,257,5,6).to(device)
    # Pytorch model
    # Pytorch_model = model.helper
    Pytorch_model = model.eval()
    
    keras_model = nobuco.pytorch_to_keras(
                        Pytorch_model, 
                        args = [x, h1, h2, h3, h4, h5, h6],
                        input_shapes = {x:(1, 257, 5, 6), h1:(1, 257, 64), h2:(1, 257, 64), h3:(1, 257, 64), h4:(1, 257, 64), h5:(1, 257, 64), h6:(1, 257, 64)},
                        input_names={x:'x', h1:'h1', h2:'h2', h3:'h3', h4:'h4', h5:'h5', h6:'h6'},
                        output_names={0:'Y', 1:'tgru_state_out1', 2:'tgru_state_out2', 3:'tgru_state_out3', 4:'tgru_state_out4', 5:'tgru_state_out5', 6:'tgru_state_out6'},
                        # args = [x, h0],
                        inputs_channel_order=ChannelOrder.PYTORCH,
                        outputs_channel_order=ChannelOrder.PYTORCH
                    )
    model_dir = './Convert'
    if os.path.exists(model_dir) == False:
        os.mkdir(model_dir)
    tf_name = args.name
    model_path = os.path.join(model_dir, tf_name)
    
    # 서명을 추가한 TensorFlow 함수 정의
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, 257, 5, 6], dtype=tf.float32),
        tf.TensorSpec(shape=[1, 257, 64], dtype=tf.float32),
        tf.TensorSpec(shape=[1, 257, 64], dtype=tf.float32),
        tf.TensorSpec(shape=[1, 257, 64], dtype=tf.float32),
        tf.TensorSpec(shape=[1, 257, 64], dtype=tf.float32),
        tf.TensorSpec(shape=[1, 257, 64], dtype=tf.float32),
        tf.TensorSpec(shape=[1, 257, 64], dtype=tf.float32)
    ])
    # def serve_model(x, h1, h2, h3):
    def serve_model(x, h1, h2, h3, h4, h5, h6):
        # outputs = keras_model(inputs=(x, h1, h2, h3))
        outputs = keras_model(inputs=(x, h1, h2, h3, h4, h5, h6))
        return {
            'Y': outputs[0],
            'tgru_state_out1': outputs[1],
            'tgru_state_out2': outputs[2],
            'tgru_state_out3': outputs[3],
            'tgru_state_out4': outputs[4],
            'tgru_state_out5': outputs[5],
            'tgru_state_out6': outputs[6]
    }
    
    # keras_model.save(model_path + '.h5')
    tf.saved_model.save(keras_model, model_path, signatures={'serving_default': serve_model})
    print('Keras Model saved')
    
    # custom_objects = {'WeightLayer': WeightLayer}
    
    # keras_model_restored = keras.models.load_model(model_path + '.h5', custom_objects=custom_objects)
    print('Model loaded')

    # converter = TFLiteConverter.from_keras_model_file(model_path + '.h5', custom_objects=custom_objects)
    # converter = tf.lite.TFLiteConverter.from_keras_model(keras_model, custom_objects=custom_objects)
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_ops = [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    with open(model_path + '.tflite', 'wb') as f:
        f.write(tflite_model)
