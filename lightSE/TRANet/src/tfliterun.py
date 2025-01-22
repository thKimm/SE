'''
DATA PROCESSING
'''

import torch
import torch.nn.functional as F
import librosa
import numpy as np
import soundfile as sf

class AudioProcessor:
    def __init__(self, frame_size=512, hop_size=128):
        self.frame_size = frame_size
        self.hop_size = hop_size

    def _to_spec(self, x):
        B, L = x.shape
        if self.frame_size == self.hop_size:
            if L % self.frame_size != 0:
                x = F.pad(x, (0, self.frame_size - (L % self.frame_size)))
            X = torch.reshape(x, (x.shape[0], self.frame_size, -1))
            X = torch.fft.rfft(X, dim=1)
            X = torch.stack([X.real, X.imag], dim=-1)
        else:
            X = torch.stft(x, n_fft=self.frame_size, hop_length=self.hop_size,
                           window=torch.hann_window(self.frame_size).to(x.device), return_complex=False)
        return X

    def _to_signal(self, stft):
        # stft.size() = [B, F, T, 2]
        stft = stft[..., 0] + 1j * stft[..., 1]  # stft.shape (B, F, T)
        if self.frame_size == self.hop_size:
            out_signal = torch.fft.irfft(stft, dim=1)
            out_signal = torch.reshape(out_signal, (out_signal.shape[0], -1))
        else:
            out_signal = torch.istft(stft, n_fft=self.frame_size, hop_length=self.hop_size,
                                     window=torch.hann_window(self.frame_size).to(stft.device))
        return out_signal  # out_signal.shape == (B, N), N = num_samples

    def process_audio(self, x):
        if len(x.shape) == 3:
            B, C, L = x.shape
            X = []
            for c in range(C):
                X.append(self._to_spec(x[:, c, :]))
                X.append(torch.abs(torch.stft(x[:, c, :],
                                n_fft=self.frame_size,
                                hop_length=self.hop_size,
                                window=torch.hann_window(self.frame_size).to(x.device),
                                return_complex=True)).unsqueeze(dim=-1))
            X = torch.cat(X, dim=-1)
        else:
            B, L = x.shape
            X = self._to_spec(x)
        return X

# WAV 파일을 읽고 처리하는 함수
def load_and_process_wav(file_path, sample_rate=16000):
    # Load the wav file using librosa
    audio, sr = librosa.load(file_path, sr=sample_rate, mono=False)
    
    # If the audio is mono, make it 2 channels by duplicating the data
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)

    # Ensure the audio data is in np.float32 format
    audio = audio.astype(np.float32)

    # Convert the audio data to a PyTorch tensor with float32 dtype
    audio_tensor = torch.tensor(audio, dtype=torch.float32)
    
    # Add batch dimension
    audio_tensor = audio_tensor.unsqueeze(0)  # Shape: [1, 2, L]
    
    # Process the audio to get the STFT
    processor = AudioProcessor(frame_size=512, hop_size=128)
    stft_tensor = processor.process_audio(audio_tensor)

    # Convert the processed STFT tensor back to numpy array with float32 dtype
    stft_numpy = stft_tensor.detach().cpu().numpy().astype(np.float32)

    return stft_numpy, audio

# STFT --> wav and save
def save_wav_from_stft(stft_np, file_path, sample_rate=16000, frame_size=512, hop_size=128):
    processor = AudioProcessor(frame_size=frame_size, hop_size=hop_size)
    
    # STFT numpy --> torch tensor
    stft_tensor = torch.tensor(stft_np, dtype=torch.float32)
    
    signal_tensor = processor._to_signal(stft_tensor)
    
    # remove batch dimension
    signal_np = signal_tensor.squeeze(0).cpu().numpy()
    
    sf.write(file_path, signal_np, sample_rate)

if __name__ == "__main__":
    import argparse
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    parser = argparse.ArgumentParser(description='Convert .wav file to .tflite file')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the .tflite model')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to the input .wav file')
    parser.add_argument('--output_path', type=str, required=False, default='./',
                        help='Path to save the output .wav file')
    args = parser.parse_args()
    
    # Example usage
    file_path = args.input_path
    stft, original_audio = load_and_process_wav(file_path)
    zero_pad = np.zeros((1, 257, 2, 6), dtype=np.float32)
    stft = np.concatenate([zero_pad, stft, zero_pad], axis=2)
    print(f"STFT shape: {stft.shape}")
    print(f"Original audio shape: {original_audio.shape}")

    import tensorflow as tf
    import numpy as np


    '''
    MODEL SETTING
    '''
    model_path = args.model_path
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    signature_runner = interpreter.get_signature_runner('serving_default')
    # interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Assume the GRU model has the following input details:
    # input_details[0] -> input data
    # input_details[1] -> initial hidden state

    # Prepare the input data
    # Example input data shape: (batch_size, time_steps, input_dim)
    
    # input_shape = input_details[0]['shape']
    # input_data = np.random.random_sample(input_shape).astype(np.float32)
    input_data = np.random.random_sample([1, 257, 1, 6]).astype(np.float32)

    # Prepare the initial hidden state
    # Example hidden state shape: (num_layers, batch_size, hidden_dim)
    # hidden_state1_shape = input_details[1]['shape']
    hidden_state1 = np.zeros([1, 257, 64], dtype=np.float32)

    # hidden_state2_shape = input_details[2]['shape']
    hidden_state2 = np.zeros([1, 257, 64], dtype=np.float32)
    
    # hidden_state3_shape = input_details[3]['shape']
    hidden_state3 = np.zeros([1, 257, 64], dtype=np.float32)
    
    # hidden_state4_shape = input_details[4]['shape']
    hidden_state4 = np.zeros([1, 257, 64], dtype=np.float32)
    
    # hidden_state5_shape = input_details[5]['shape']
    hidden_state5 = np.zeros([1, 257, 64], dtype=np.float32)
    
    # hidden_state6_shape = input_details[6]['shape']
    hidden_state6 = np.zeros([1, 257, 64], dtype=np.float32)
    '''
    RUN MODEL
    '''
    B,F,T,C = stft.shape
    output = np.zeros((B,F,T,2), dtype=np.float32)
    import tqdm
    for i in tqdm.tqdm(range(stft.shape[2]-4)):
        input_data = stft[:,:,i:i+5,:]
            

        # # Set the tensor to point to the input data and hidden state
        # interpreter.set_tensor(input_details[0]['index'], input_data)
        # interpreter.set_tensor(input_details[1]['index'], hidden_state1)
        # interpreter.set_tensor(input_details[2]['index'], hidden_state2)
        # interpreter.set_tensor(input_details[3]['index'], hidden_state3)
        # # Run the model
        # interpreter.invoke()
        outputs = signature_runner(x=input_data, h1=hidden_state1, h2=hidden_state2, h3=hidden_state3, h4=hidden_state4, h5=hidden_state5, h6=hidden_state6)

        # Get the output data
        # output_data = interpreter.get_tensor(output_details['Y']['index'])
        # # print("Output:", output_data)
        # hidden_state1 = interpreter.get_tensor(output_details['tgru_state_out1']['index'])
        # hidden_state2 = interpreter.get_tensor(output_details['tgru_state_out2']['index'])
        # hidden_state3 = interpreter.get_tensor(output_details['tgru_state_out3']['index'])
        # print("Final hidden state:", hidden_state)
        output_data = outputs['Y']
        hidden_state1 = outputs['tgru_state_out1']
        hidden_state2 = outputs['tgru_state_out2']
        hidden_state3 = outputs['tgru_state_out3']
        hidden_state4 = outputs['tgru_state_out4']
        hidden_state5 = outputs['tgru_state_out5']
        hidden_state6 = outputs['tgru_state_out6']
        output[:,:,i:i+1,:] = output_data

    # Save the output as a wav file
    output_file_path = args.output_path
    save_wav_from_stft(output, output_file_path)

    
