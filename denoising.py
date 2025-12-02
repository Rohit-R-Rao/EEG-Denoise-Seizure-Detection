# -*- coding: utf-8 -*-
# Copyright (c) 2022, Kwanhyung Lee, AITRICS. All rights reserved.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pyedflib import highlevel, EdfReader
from scipy.io.wavfile import write
from scipy import signal as sci_sig
from scipy.spatial.distance import pdist
from scipy.signal import stft, hilbert, butter, freqz, filtfilt, find_peaks, iirnotch
from builder.utils.process_util import run_multi_process
from builder.utils.utils import search_walk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import sys
import argparse
import torch
import glob
import pickle
import random
import mne 
from mne.io.edf.edf import _read_annotations_edf, _read_edf_header
from itertools import groupby

GLOBAL_DATA = {}
label_dict = {}
sample_rate_dict = {}
sev_label = {}

# Global variable for DeepSeparator model (lazy loading)
_deepseparator_model = None
_deepseparator_device = None


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply bandpass filter to remove frequencies outside EEG range"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def notch_filter(data, freq, fs, quality=30):
    """Apply notch filter to remove line noise (50/60 Hz)"""
    b, a = iirnotch(freq, quality, fs)
    y = filtfilt(b, a, data)
    return y


def load_deepseparator_model(checkpoint_path):
    """Load DeepSeparator model from checkpoint (lazy loading, only loads once)"""
    global _deepseparator_model, _deepseparator_device
    
    if _deepseparator_model is not None:
        return _deepseparator_model, _deepseparator_device
    
    try:
        # Resolve checkpoint path to absolute path first
        if not os.path.isabs(checkpoint_path):
            # Try to resolve relative to current working directory
            checkpoint_path = os.path.abspath(checkpoint_path)
            # If still doesn't exist, try relative to script directory
            if not os.path.exists(checkpoint_path):
                if '__file__' in globals() and __file__:
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                else:
                    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
                checkpoint_path = os.path.join(script_dir, checkpoint_path)
                checkpoint_path = os.path.abspath(checkpoint_path)
        
        # Get DeepSeparator code directory from checkpoint path
        # Checkpoint is at: DeepSeparator/code/checkpoint/DeepSeparator.pkl
        # So code dir is: DeepSeparator/code/
        checkpoint_dir = os.path.dirname(checkpoint_path)  # .../checkpoint
        deepseparator_code_dir = os.path.dirname(checkpoint_dir)  # .../code
        
        # Verify the code directory exists and has network.py
        if not os.path.exists(deepseparator_code_dir):
            # Try alternative: if checkpoint_path was relative, try finding it
            if '__file__' in globals() and __file__:
                script_dir = os.path.dirname(os.path.abspath(__file__))
            else:
                script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
            
            # Try relative to script
            deepseparator_code_dir = os.path.join(script_dir, 'DeepSeparator', 'code')
            deepseparator_code_dir = os.path.abspath(deepseparator_code_dir)
            
            if not os.path.exists(deepseparator_code_dir):
                deepseparator_code_dir = os.path.join(script_dir, '..', 'DeepSeparator', 'code')
                deepseparator_code_dir = os.path.abspath(deepseparator_code_dir)
        
        # Verify network.py exists
        network_file = os.path.join(deepseparator_code_dir, 'network.py')
        if not os.path.exists(network_file):
            raise ImportError(f"network.py not found in {deepseparator_code_dir}")
        
        # Add to Python path
        if deepseparator_code_dir not in sys.path:
            sys.path.insert(0, deepseparator_code_dir)
        
        from network import DeepSeparator
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = DeepSeparator()
        model.to(device)
        
        if os.path.exists(checkpoint_path):
            print(f"Loading DeepSeparator model from {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()  # Set to evaluation mode
            _deepseparator_model = model
            _deepseparator_device = device
            print("DeepSeparator model loaded successfully")
            return model, device
        else:
            raise FileNotFoundError(f"DeepSeparator checkpoint not found at {checkpoint_path}")
    except Exception as e:
        print(f"Error loading DeepSeparator model: {e}")
        raise


def denoise_with_deepseparator(signal, model, device, batch_size=10000):
    """
    Denoise a single-channel EEG signal using DeepSeparator model.
    
    Args:
        signal: 1D numpy array of EEG signal
        model: Loaded DeepSeparator model
        device: torch device (cpu or cuda)
        batch_size: Batch size for processing (default: 10000, matching training)
    
    Returns:
        Denoised signal as 1D numpy array
    """
    # Standardize the signal (as done in training)
    mu = np.mean(signal)
    sigma = np.std(signal)
    if sigma == 0:
        return signal.copy()  # Return original if no variance
    
    standardized_signal = (signal - mu) / sigma
    
    # Convert to tensor
    signal_tensor = torch.from_numpy(standardized_signal).float().to(device)
    
    # Process in batches if signal is very long
    signal_length = len(signal)
    denoised_parts = []
    
    if signal_length <= batch_size:
        # Process entire signal at once
        signal_batch = signal_tensor.unsqueeze(0)  # Add batch dimension: [1, length]
        indicator = torch.tensor([[0.0]], device=device)  # 0 for denoising
        
        with torch.no_grad():
            denoised_batch = model(signal_batch, indicator)
        
        denoised_signal = denoised_batch.cpu().numpy()[0]  # Remove batch dimension
    else:
        # Process in chunks
        for start_idx in range(0, signal_length, batch_size):
            end_idx = min(start_idx + batch_size, signal_length)
            signal_chunk = signal_tensor[start_idx:end_idx].unsqueeze(0)
            indicator = torch.tensor([[0.0]], device=device)
            
            with torch.no_grad():
                denoised_chunk = model(signal_chunk, indicator)
            
            denoised_parts.append(denoised_chunk.cpu().numpy()[0])
        
        denoised_signal = np.concatenate(denoised_parts)
    
    # Un-standardize
    denoised_signal = denoised_signal * sigma + mu
    
    return denoised_signal


def denoise_signal(signal, sample_rate, apply_bandpass=True, apply_notch=True, 
                   line_noise_freq=50, use_deepseparator=False, 
                   deepseparator_checkpoint=None):
    """
    Apply denoising filters to EEG signal.
    
    Args:
        signal: 1D numpy array of EEG signal
        sample_rate: Sampling rate in Hz
        apply_bandpass: Whether to apply bandpass filter
        apply_notch: Whether to apply notch filter
        line_noise_freq: Line noise frequency (50 or 60 Hz)
        use_deepseparator: Whether to use DeepSeparator model for denoising
        deepseparator_checkpoint: Path to DeepSeparator checkpoint file
    
    Returns:
        Denoised signal
    """
    denoised = signal.copy()
    
    # Apply traditional filters first (always, unless explicitly disabled)
    # When using DeepSeparator, traditional filters are applied first by default
    # unless apply_traditional_filters_first is False
    should_apply_traditional = True
    if use_deepseparator:
        # If using DeepSeparator, check if we should skip traditional filters
        should_apply_traditional = GLOBAL_DATA.get('apply_traditional_filters_first', True)
    
    if should_apply_traditional:
        # Bandpass filter: typical EEG range is 0.5-70 Hz
        if apply_bandpass:
            denoised = butter_bandpass_filter(denoised, lowcut=0.5, highcut=70.0, 
                                             fs=sample_rate, order=4)
        
        # Notch filter: remove 50 Hz (Europe) or 60 Hz (US) line noise
        if apply_notch:
            if sample_rate > 2 * line_noise_freq:  # Nyquist criterion
                denoised = notch_filter(denoised, line_noise_freq, sample_rate)
    
    # Apply DeepSeparator if requested (after traditional filters)
    if use_deepseparator:
        if deepseparator_checkpoint is None:
            raise ValueError("deepseparator_checkpoint must be provided when use_deepseparator=True")
        
        try:
            model, device = load_deepseparator_model(deepseparator_checkpoint)
            denoised = denoise_with_deepseparator(denoised, model, device)
        except Exception as e:
            print(f"Warning: DeepSeparator denoising failed: {e}. Using traditional filters only.")
            # Fall back to traditional filters if DeepSeparator fails
            if not apply_bandpass and not apply_notch:
                # If no traditional filters were applied, apply them now as fallback
                if apply_bandpass:
                    denoised = butter_bandpass_filter(denoised, lowcut=0.5, highcut=70.0, 
                                                     fs=sample_rate, order=4)
                if apply_notch and sample_rate > 2 * line_noise_freq:
                    denoised = notch_filter(denoised, line_noise_freq, sample_rate)
    
    return denoised


def label_sampling_tuh(labels, feature_samplerate):
    y_target = ""
    remained = 0
    feature_intv = 1/float(feature_samplerate)
    for i in labels:
        # Skip empty lines
        if not i.strip():
            continue
            
        parts = i.strip().split()
        # Need at least 3 parts: begin, end, label
        if len(parts) < 3:
            print(f"Warning: Malformed label line (expected at least 3 space-separated values): '{i.strip()}'")
            continue
            
        begin, end, label = parts[0], parts[1], parts[2]

        # Handle binary label type: map any non-bckg label to 'seiz'
        if GLOBAL_DATA['label_type'] == 'tse_bi' and label not in GLOBAL_DATA['disease_labels']:
            if label != 'bckg':
                label = 'seiz'
        
        # Auto-detect: if we see 'seiz' label but label_type is 'tse', treat it as binary
        if GLOBAL_DATA['label_type'] == 'tse' and label == 'seiz':
            # Map 'seiz' to 'bckg' (0) for now, or skip - actually, we should map to a valid label
            # Since 'seiz' means any seizure, and we're in 'tse' mode, we can't map it properly
            # So we'll skip it with a more informative message
            print(f"Warning: Found 'seiz' label but label_type is 'tse'. Consider using --label_type tse_bi. Skipping this interval.")
            continue
        
        # If label still not found, skip or use default
        if label not in GLOBAL_DATA['disease_labels']:
            print(f"Warning: Unknown label '{label}' found. Skipping this interval.")
            continue

        try:
            intv_count, remained = divmod(float(end) - float(begin) + remained, feature_intv)
            y_target += int(intv_count) * str(GLOBAL_DATA['disease_labels'][label])
        except (ValueError, TypeError) as e:
            print(f"Warning: Error parsing time values (begin={begin}, end={end}): {e}. Skipping this interval.")
            continue
    return y_target

def read_label_file(file_name, label_type):
    """Read label file, handling both .tse_bi and .csv_bi formats"""
    # Try multiple extensions in order of preference
    possible_extensions = []
    
    if label_type == 'tse_bi':
        possible_extensions = ['.tse_bi', '.csv_bi']
    elif label_type == 'tse':
        # For 'tse', try .tse first, then .tse_bi, then .csv_bi
        possible_extensions = ['.tse', '.tse_bi', '.csv_bi']
    else:
        possible_extensions = [f'.{label_type}']
    
    label_file_path = None
    for ext in possible_extensions:
        test_path = file_name + ext
        if os.path.exists(test_path):
            label_file_path = test_path
            break
    
    if label_file_path is None:
        return None, None
    
    try:
        label_file = open(label_file_path, 'r')
    except FileNotFoundError:
        return None, None
    
    y = label_file.readlines()
    y = list(y[2:])  # Skip first 2 header lines
    
    # Handle CSV format (comma-separated) vs TSE format (space-separated)
    if label_file_path.endswith('.csv_bi'):
        # CSV format: skip CSV header line, then parse comma-separated values
        y = [line for line in y if line.strip() and not line.strip().startswith('#')]
        if y and y[0].startswith('channel,'):  # Skip CSV header
            y = y[1:]
        # Parse CSV lines safely, checking for enough columns
        parsed_lines = []
        y_labels = []
        for line in y:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:  # Need at least 4 columns: channel, begin, end, label
                parsed_lines.append(f"{parts[1]} {parts[2]} {parts[3]}")
                y_labels.append(parts[3])
            else:
                print(f"Warning: CSV line has insufficient columns (expected at least 4): '{line.strip()}'")
        y = parsed_lines
        y_labels = list(set(y_labels))
    else:
        # TSE format: space-separated (handle multiple spaces)
        parsed_lines = []
        y_labels = []
        for i in y:
            if not i.strip():
                continue
            parts = i.strip().split()  # split() handles multiple spaces
            if len(parts) >= 3:  # Need at least 3 parts: begin, end, label
                parsed_lines.append(i.strip())
                y_labels.append(parts[2])
            else:
                print(f"Warning: TSE line has insufficient columns (expected at least 3): '{i.strip()}'")
        y = parsed_lines
        y_labels = list(set(y_labels))
    
    return y, y_labels


def generate_training_data_leadwise_tuh_train(file):
    sample_rate = GLOBAL_DATA['sample_rate']    # EX) 200Hz
    file_name = ".".join(file.split(".")[:-1])  # EX) $PATH_TO_EEG/train/01_tcp_ar/072/00007235/s003_2010_11_20/00007235_s003_t000
    data_file_name = file_name.split("/")[-1]   # EX) 00007235_s003_t000
    signals, signal_headers, header = highlevel.read_edf(file)
    label_list_c = []
    for idx, signal in enumerate(signals):
        label_noref = signal_headers[idx]['label'].split("-")[0]    # EX) EEG FP1-ref or EEG FP1-LE --> EEG FP1
        label_list_c.append(label_noref)   

    ############################# part 1: labeling  ###############################
    y, y_labels = read_label_file(file_name, GLOBAL_DATA['label_type'])
    if y is None:
        print(f"Warning: Label file not found for {file_name}. Skipping this file.")
        return  # Skip this file if the label file doesn't exist
    signal_sample_rate = int(signal_headers[0]['sample_frequency'])
    if sample_rate > signal_sample_rate:
        return
    if not all(elem in label_list_c for elem in GLOBAL_DATA['label_list']): # if one or more of ['EEG FP1', 'EEG FP2', ... doesn't exist
        return
    # if not any(elem in y_labels for elem in GLOBAL_DATA['disease_type']): # if non-patient exist
    #     return
    y_sampled = label_sampling_tuh(y, GLOBAL_DATA['feature_sample_rate'])
    
    ############################# part 2: input data filtering #############################
    signal_list = []
    signal_label_list = []
    signal_final_list_raw = []

    for idx, signal in enumerate(signals):
        label = signal_headers[idx]['label'].split("-")[0]
        if label not in GLOBAL_DATA['label_list']:
            continue

        if int(signal_headers[idx]['sample_frequency']) > sample_rate:
            secs = len(signal)/float(signal_sample_rate)
            samps = int(secs*sample_rate)
            x = sci_sig.resample(signal, samps)
            # Apply denoising after resampling
            x = denoise_signal(x, sample_rate, 
                             apply_bandpass=GLOBAL_DATA.get('apply_bandpass', True),
                             apply_notch=GLOBAL_DATA.get('apply_notch', True),
                             use_deepseparator=GLOBAL_DATA.get('use_deepseparator', False),
                             deepseparator_checkpoint=GLOBAL_DATA.get('deepseparator_checkpoint', None))
            signal_list.append(x)
            signal_label_list.append(label)
        else:
            # Apply denoising to signals that don't need resampling
            x = denoise_signal(signal, sample_rate,
                             apply_bandpass=GLOBAL_DATA.get('apply_bandpass', True),
                             apply_notch=GLOBAL_DATA.get('apply_notch', True),
                             use_deepseparator=GLOBAL_DATA.get('use_deepseparator', False),
                             deepseparator_checkpoint=GLOBAL_DATA.get('deepseparator_checkpoint', None))
            signal_list.append(x)
            signal_label_list.append(label)

    if len(signal_label_list) != len(GLOBAL_DATA['label_list']):
        print("Not enough labels: ", signal_label_list)
        return 
    
    for lead_signal in GLOBAL_DATA['label_list']:
        signal_final_list_raw.append(signal_list[signal_label_list.index(lead_signal)])

    new_length = len(signal_final_list_raw[0]) * (float(GLOBAL_DATA['feature_sample_rate']) / GLOBAL_DATA['sample_rate'])
    
    new_length = int(new_length)  # Convert to int for indexing and string operations
    if len(y_sampled) > new_length:
        y_sampled = y_sampled[:new_length]
    elif len(y_sampled) < new_length:
        diff = int(new_length - len(y_sampled))
        if len(y_sampled) > 0:
            y_sampled += y_sampled[-1] * diff
        else:
            # If y_sampled is empty (all labels were skipped), pad with '0' (background)
            print(f"Warning: y_sampled is empty for {data_file_name}. Padding with background labels.")
            y_sampled = '0' * int(new_length)

    # Check if y_sampled is still empty after padding
    if len(y_sampled) == 0:
        print(f"Warning: No valid labels found for {data_file_name}. Skipping this file.")
        return

    y_sampled_np = np.array(list(map(int,y_sampled)))
    new_labels = []
    new_labels_idxs = []

    ############################# part 3: slicing for easy training  #############################
    y_sampled = ["0" if l not in GLOBAL_DATA['selected_diseases'] else l for l in y_sampled]

    if any(l in GLOBAL_DATA['selected_diseases'] for l in y_sampled):
        y_sampled = [str(GLOBAL_DATA['target_dictionary'][int(l)]) if l in GLOBAL_DATA['selected_diseases'] else l for l in y_sampled]

    # slice and save if training data
    new_data = {}
    raw_data = torch.from_numpy(np.array(signal_final_list_raw)).permute(1,0)

    max_seg_len_before_seiz_label = GLOBAL_DATA['max_bckg_before_slicelength'] * GLOBAL_DATA['feature_sample_rate']
    max_seg_len_before_seiz_raw = GLOBAL_DATA['max_bckg_before_slicelength'] * GLOBAL_DATA['sample_rate']
    max_seg_len_after_seiz_label = GLOBAL_DATA['max_bckg_after_seiz_length'] * GLOBAL_DATA['feature_sample_rate']
    max_seg_len_after_seiz_raw = GLOBAL_DATA['max_bckg_after_seiz_length'] * GLOBAL_DATA['sample_rate']

    min_seg_len_label = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['feature_sample_rate']
    min_seg_len_raw = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['sample_rate']
    max_seg_len_label = GLOBAL_DATA['max_binary_slicelength'] * GLOBAL_DATA['feature_sample_rate']
    max_seg_len_raw = GLOBAL_DATA['max_binary_slicelength'] * GLOBAL_DATA['sample_rate']

    label_order = [x[0] for x in groupby(y_sampled)]
    label_change_idxs = np.where(y_sampled_np[:-1] != y_sampled_np[1:])[0]

    start_raw_idx = 0
    start_label_idx = 0
    end_raw_idx = raw_data.size(0)
    end_label_idx = len(y_sampled)
    previous_bckg_len = 0
    
    sliced_raws = []
    sliced_labels = []
    pre_bckg_lens_label = []
    label_list_for_filename = []
    
    for idx, label in enumerate(label_order):
        # if last and the label is "bckg"
        if (len(label_order) == idx+1) and (label == "0"):
            sliced_raw_data = raw_data[start_raw_idx:].permute(1,0)
            sliced_y1 = torch.Tensor(list(map(int,y_sampled[start_label_idx:]))).byte()

            if sliced_y1.size(0) < min_seg_len_label:
                continue
            sliced_raws.append(sliced_raw_data)
            sliced_labels.append(sliced_y1)
            pre_bckg_lens_label.append(0)
            label_list_for_filename.append(label)

        # if not last and the label is "bckg"
        elif (len(label_order) != idx+1) and (label == "0"):
            end_raw_idx = (label_change_idxs[idx]+1) * GLOBAL_DATA['fsr_sr_ratio']
            end_label_idx = label_change_idxs[idx]+1
            
            sliced_raw_data = raw_data[start_raw_idx:end_raw_idx].permute(1,0)
            sliced_y1 = torch.Tensor(list(map(int,y_sampled[start_label_idx:end_label_idx]))).byte()
            previous_bckg_len = end_label_idx - start_label_idx

            start_raw_idx = end_raw_idx
            start_label_idx = end_label_idx
            if sliced_y1.size(0) < min_seg_len_label:
                continue

            sliced_raws.append(sliced_raw_data)
            sliced_labels.append(sliced_y1)
            pre_bckg_lens_label.append(0)
            label_list_for_filename.append(label)

        # if the first and the label is "seiz" 1 ~ 8       
        elif (idx == 0) and (label != "0"):
            end_raw_idx = (label_change_idxs[idx]+1) * GLOBAL_DATA['fsr_sr_ratio']
            end_label_idx = label_change_idxs[idx]+1
            
            if len(y_sampled)-end_label_idx > max_seg_len_after_seiz_label:
                post_len_label = max_seg_len_after_seiz_label
                post_len_raw = max_seg_len_after_seiz_raw
            else:
                post_len_label = len(y_sampled)-end_label_idx
                post_len_raw = ((len(y_sampled)-end_label_idx) * GLOBAL_DATA['fsr_sr_ratio'])
            post_ictal_end_label = end_label_idx + post_len_label
            post_ictal_end_raw = end_raw_idx + post_len_raw
            
            start_raw_idx = end_raw_idx
            start_label_idx = end_label_idx
            if len(y_sampled) < min_seg_len_label:
                continue

            sliced_raw_data = raw_data[:post_ictal_end_raw].permute(1,0)
            sliced_y1 = torch.Tensor(list(map(int,y_sampled[:post_ictal_end_label]))).byte()

            if sliced_y1.size(0) > max_seg_len_label:
                sliced_y2 = sliced_y1[:max_seg_len_label]
                sliced_raw_data2 = sliced_raw_data.permute(1,0)[:max_seg_len_raw].permute(1,0)
                sliced_raws.append(sliced_raw_data2)
                sliced_labels.append(sliced_y2)
                pre_bckg_lens_label.append(0)
                label_list_for_filename.append(label)
            elif sliced_y1.size(0) >= min_seg_len_label:
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y1)
                pre_bckg_lens_label.append(0)
                label_list_for_filename.append(label)
            else:
                sliced_y2 = torch.Tensor(list(map(int,y_sampled[:min_seg_len_label]))).byte()
                sliced_raw_data2 = raw_data[:min_seg_len_raw].permute(1,0)
                sliced_raws.append(sliced_raw_data2)
                sliced_labels.append(sliced_y2)
                pre_bckg_lens_label.append(0)
                label_list_for_filename.append(label)

        # the label is "seiz" 1 ~ 8
        elif label != "0":
            end_raw_idx = (label_change_idxs[idx]+1) * GLOBAL_DATA['fsr_sr_ratio']
            end_label_idx = label_change_idxs[idx]+1
            
            if len(y_sampled)-end_label_idx > max_seg_len_after_seiz_label:
                post_len_label = max_seg_len_after_seiz_label
                post_len_raw = max_seg_len_after_seiz_raw
            else:
                post_len_label = len(y_sampled)-end_label_idx
                post_len_raw = ((len(y_sampled)-end_label_idx) * GLOBAL_DATA['fsr_sr_ratio'])
            post_ictal_end_label = end_label_idx + post_len_label
            post_ictal_end_raw = end_raw_idx + post_len_raw

            if previous_bckg_len > max_seg_len_before_seiz_label:
                pre_seiz_label_len = max_seg_len_before_seiz_label
            else:
                pre_seiz_label_len = previous_bckg_len
            pre_seiz_raw_len = pre_seiz_label_len * GLOBAL_DATA['fsr_sr_ratio']

            sample_len = post_ictal_end_label - (start_label_idx-pre_seiz_label_len)
            if sample_len < min_seg_len_label:
                post_ictal_end_label = start_label_idx - pre_seiz_label_len + min_seg_len_label
                post_ictal_end_raw = start_raw_idx - pre_seiz_raw_len + min_seg_len_raw
            if len(y_sampled) < post_ictal_end_label:
                start_raw_idx = end_raw_idx
                start_label_idx = end_label_idx
                continue

            sliced_raw_data = raw_data[start_raw_idx-pre_seiz_raw_len:post_ictal_end_raw].permute(1,0)
            sliced_y1 = torch.Tensor(list(map(int,y_sampled[start_label_idx-pre_seiz_label_len:post_ictal_end_label]))).byte()

            if sliced_y1.size(0) > max_seg_len_label:
                sliced_y2 = sliced_y1[:max_seg_len_label]
                sliced_raw_data2 = sliced_raw_data.permute(1,0)[:max_seg_len_raw].permute(1,0)
                sliced_raws.append(sliced_raw_data2)
                sliced_labels.append(sliced_y2)
                pre_bckg_lens_label.append(pre_seiz_label_len)
                label_list_for_filename.append(label)
            # elif sliced_y1.size(0) >= min_seg_len_label:
            else:
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y1)
                pre_bckg_lens_label.append(pre_seiz_label_len)
                label_list_for_filename.append(label)
            start_raw_idx = end_raw_idx
            start_label_idx = end_label_idx
                
        else:
            print("Error! Impossible!")
            exit(1)

    for data_idx in range(len(sliced_raws)):
        sliced_raw = sliced_raws[data_idx]
        sliced_y = sliced_labels[data_idx]
        sliced_y_map = list(map(int,sliced_y))

        if GLOBAL_DATA['binary_target1'] is not None:
            sliced_y2 = torch.Tensor([GLOBAL_DATA['binary_target1'][i] for i in sliced_y_map]).byte()
        else:
            sliced_y2 = None

        if GLOBAL_DATA['binary_target2'] is not None:
            sliced_y3 = torch.Tensor([GLOBAL_DATA['binary_target2'][i] for i in sliced_y_map]).byte()
        else:
            sliced_y3 = None

        # Convert tensors to numpy arrays for efficient storage
        new_data['RAW_DATA'] = [sliced_raw.cpu().numpy().astype(np.float16)]
        new_data['LABEL1'] = [sliced_y.cpu().numpy().astype(np.uint8)]
        new_data['LABEL2'] = [sliced_y2.cpu().numpy().astype(np.uint8) if sliced_y2 is not None else None]
        new_data['LABEL3'] = [sliced_y3.cpu().numpy().astype(np.uint8) if sliced_y3 is not None else None]

        prelabel_len = pre_bckg_lens_label[data_idx]
        label = label_list_for_filename[data_idx]
        
        with open(GLOBAL_DATA['data_file_directory'] + "/{}_c{}_pre{}_len{}_label_{}.pkl".format(data_file_name, str(data_idx), str(prelabel_len), str(len(sliced_y)), str(label)), 'wb') as _f:
            pickle.dump(new_data, _f, protocol=pickle.HIGHEST_PROTOCOL)      
        new_data = {}

def generate_training_data_leadwise_tuh_train_final(file):
    sample_rate = GLOBAL_DATA['sample_rate']    # EX) 200Hz
    file_name = ".".join(file.split(".")[:-1])  # EX) $PATH_TO_EEG/train/01_tcp_ar/072/00007235/s003_2010_11_20/00007235_s003_t000
    data_file_name = file_name.split("/")[-1]   # EX) 00007235_s003_t000
    signals, signal_headers, header = highlevel.read_edf(file)
    label_list_c = []
    for idx, signal in enumerate(signals):
        label_noref = signal_headers[idx]['label'].split("-")[0]    # EX) EEG FP1-ref or EEG FP1-LE --> EEG FP1
        label_list_c.append(label_noref)   

    ############################# part 1: labeling  ###############################
    y, y_labels = read_label_file(file_name, GLOBAL_DATA['label_type'])
    if y is None:
        print(f"Warning: Label file not found for {file_name}. Skipping this file.")
        return  # Skip this file if the label file doesn't exist
    signal_sample_rate = int(signal_headers[0]['sample_frequency'])
    if sample_rate > signal_sample_rate:
        return
    if not all(elem in label_list_c for elem in GLOBAL_DATA['label_list']): # if one or more of ['EEG FP1', 'EEG FP2', ... doesn't exist
        return
    # if not any(elem in y_labels for elem in GLOBAL_DATA['disease_type']): # if non-patient exist
    #     return
    y_sampled = label_sampling_tuh(y, GLOBAL_DATA['feature_sample_rate'])

    # check if seizure patient or non-seizure patient
    patient_wise_dir = "/".join(file_name.split("/")[:-2])
    patient_id = file_name.split("/")[-3]
    edf_list = search_walk({'path': patient_wise_dir, 'extension': ".tse_bi"})
    if not edf_list:  # Try csv_bi if tse_bi not found
        edf_list = search_walk({'path': patient_wise_dir, 'extension': ".csv_bi"})
    patient_bool = False
    for label_file_path in edf_list:
        y, _ = read_label_file(".".join(label_file_path.split(".")[:-1]), GLOBAL_DATA['label_type'])
        if y is None:
            continue
        for line in y:
            if len(line) > 5:
                parts = line.split(" ")
                if len(parts) > 2 and parts[2] != 'bckg':
                    patient_bool = True
                    break
        if patient_bool:
            break
    
    ############################# part 2: input data filtering #############################
    signal_list = []
    signal_label_list = []
    signal_final_list_raw = []

    for idx, signal in enumerate(signals):
        label = signal_headers[idx]['label'].split("-")[0]
        if label not in GLOBAL_DATA['label_list']:
            continue

        if int(signal_headers[idx]['sample_frequency']) > sample_rate:
            secs = len(signal)/float(signal_sample_rate)
            samps = int(secs*sample_rate)
            x = sci_sig.resample(signal, samps)
            # Apply denoising after resampling
            x = denoise_signal(x, sample_rate, 
                             apply_bandpass=GLOBAL_DATA.get('apply_bandpass', True),
                             apply_notch=GLOBAL_DATA.get('apply_notch', True),
                             use_deepseparator=GLOBAL_DATA.get('use_deepseparator', False),
                             deepseparator_checkpoint=GLOBAL_DATA.get('deepseparator_checkpoint', None))
            signal_list.append(x)
            signal_label_list.append(label)
        else:
            # Apply denoising to signals that don't need resampling
            x = denoise_signal(signal, sample_rate,
                             apply_bandpass=GLOBAL_DATA.get('apply_bandpass', True),
                             apply_notch=GLOBAL_DATA.get('apply_notch', True),
                             use_deepseparator=GLOBAL_DATA.get('use_deepseparator', False),
                             deepseparator_checkpoint=GLOBAL_DATA.get('deepseparator_checkpoint', None))
            signal_list.append(x)
            signal_label_list.append(label)

    if len(signal_label_list) != len(GLOBAL_DATA['label_list']):
        print("Not enough labels: ", signal_label_list)
        return 
    
    for lead_signal in GLOBAL_DATA['label_list']:
        signal_final_list_raw.append(signal_list[signal_label_list.index(lead_signal)])

    new_length = len(signal_final_list_raw[0]) * (float(GLOBAL_DATA['feature_sample_rate']) / GLOBAL_DATA['sample_rate'])
    
    new_length = int(new_length)  # Convert to int for indexing and string operations
    if len(y_sampled) > new_length:
        y_sampled = y_sampled[:new_length]
    elif len(y_sampled) < new_length:
        diff = int(new_length - len(y_sampled))
        if len(y_sampled) > 0:
            y_sampled += y_sampled[-1] * diff
        else:
            # If y_sampled is empty (all labels were skipped), pad with '0' (background)
            print(f"Warning: y_sampled is empty for {data_file_name}. Padding with background labels.")
            y_sampled = '0' * int(new_length)

    # Check if y_sampled is still empty after padding
    if len(y_sampled) == 0:
        print(f"Warning: No valid labels found for {data_file_name}. Skipping this file.")
        return

    y_sampled_np = np.array(list(map(int,y_sampled)))
    new_labels = []
    new_labels_idxs = []

    ############################# part 3: slicing for easy training  #############################
    y_sampled = ["0" if l not in GLOBAL_DATA['selected_diseases'] else l for l in y_sampled]

    if any(l in GLOBAL_DATA['selected_diseases'] for l in y_sampled):
        y_sampled = [str(GLOBAL_DATA['target_dictionary'][int(l)]) if l in GLOBAL_DATA['selected_diseases'] else l for l in y_sampled]

    # slice and save if training data
    new_data = {}
    raw_data = torch.from_numpy(np.array(signal_final_list_raw)).permute(1,0)
    raw_data = raw_data.type(torch.float16)
    
    min_seg_len_label = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['feature_sample_rate']
    min_seg_len_raw = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['sample_rate']
    min_binary_edge_seiz_label = GLOBAL_DATA['min_binary_edge_seiz'] * GLOBAL_DATA['feature_sample_rate']
    min_binary_edge_seiz_raw = GLOBAL_DATA['min_binary_edge_seiz'] * GLOBAL_DATA['sample_rate']

    label_order = [x[0] for x in groupby(y_sampled)]
    label_change_idxs = np.where(y_sampled_np[:-1] != y_sampled_np[1:])[0]
    label_change_idxs = np.append(label_change_idxs, np.array([len(y_sampled_np)-1]))

    sliced_raws = []
    sliced_labels = []
    label_list_for_filename = []
    if len(y_sampled) < min_seg_len_label:
        return
    else:
        label_count = {}
        y_sampled_2nd = list(y_sampled)
        raw_data_2nd = raw_data
        while len(y_sampled) >= min_seg_len_label:
            is_at_middle = False
            sliced_y = y_sampled[:min_seg_len_label]
            labels = [x[0] for x in groupby(sliced_y)]
                
            if len(labels) == 1 and "0" in labels:
                y_sampled = y_sampled[min_seg_len_label:]
                sliced_raw_data = raw_data[:min_seg_len_raw].permute(1,0)
                raw_data = raw_data[min_seg_len_raw:]
                if patient_bool:
                    label = "0_patT"
                else:
                    label = "0_patF"
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)
                label_list_for_filename.append(label)
                
            elif len(labels) != 1 and (sliced_y[0] == '0') and (sliced_y[-1] != '0'):
                temp_sliced_y = list(sliced_y)
                temp_sliced_y.reverse()
                boundary_seizlen = temp_sliced_y.index("0") + 1
                if boundary_seizlen < min_binary_edge_seiz_label:
                    if len(y_sampled) > (min_seg_len_label + min_binary_edge_seiz_label):
                        sliced_y = y_sampled[min_binary_edge_seiz_label:min_seg_len_label+min_binary_edge_seiz_label]
                        sliced_raw_data = raw_data[min_binary_edge_seiz_raw:min_seg_len_raw+min_binary_edge_seiz_raw].permute(1,0)
                    else:
                        sliced_raw_data = raw_data[:min_seg_len_raw].permute(1,0)
                else:
                    sliced_raw_data = raw_data[:min_seg_len_raw].permute(1,0)

                y_sampled = y_sampled[min_seg_len_label:]
                raw_data = raw_data[min_seg_len_raw:]
                
                label = str(max(list(map(int, labels))))
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)
                
                label = label + "_beg"
                label_list_for_filename.append(label)
                is_at_middle = True

            elif (len(labels) != 1) and (sliced_y[0] != '0') and (sliced_y[-1] != '0'):
                y_sampled = y_sampled[min_seg_len_label:]
                sliced_raw_data = raw_data[:min_seg_len_raw].permute(1,0)
                raw_data = raw_data[min_seg_len_raw:]

                label = str(max(list(map(int, labels))))
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)

                label = label + "_whole"
                label_list_for_filename.append(label)
                is_at_middle = True

            elif (len(labels) == 1) and (sliced_y[0] != '0') and (sliced_y[-1] != '0'):
                y_sampled = y_sampled[min_seg_len_label:]
                sliced_raw_data = raw_data[:min_seg_len_raw].permute(1,0)
                raw_data = raw_data[min_seg_len_raw:]

                label = str(max(list(map(int, labels))))
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)

                label = label + "_middle"
                label_list_for_filename.append(label)
                is_at_middle = True
            
            elif len(labels) != 1 and (sliced_y[0] != '0') and (sliced_y[-1] == '0'):
                y_sampled = y_sampled[min_seg_len_label:]
                sliced_raw_data = raw_data[:min_seg_len_raw].permute(1,0)
                raw_data = raw_data[min_seg_len_raw:]
                
                label = str(max(list(map(int, labels))))
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)
                
                label = label + "_end"
                label_list_for_filename.append(label)
            
            elif len(labels) != 1 and (sliced_y[0] == '0') and (sliced_y[-1] == '0'):
                y_sampled = y_sampled[min_seg_len_label:]
                sliced_raw_data = raw_data[:min_seg_len_raw].permute(1,0)
                raw_data = raw_data[min_seg_len_raw:]
                
                label = str(max(list(map(int, labels))))
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)

                label = label + "_whole"
                label_list_for_filename.append(label)
            
            else:
                print("unexpected case")
                exit(1)
        if is_at_middle == True:
            sliced_y = y_sampled_2nd[-min_seg_len_label:]
            sliced_raw_data = raw_data_2nd[-min_seg_len_raw:].permute(1,0)
            
            if sliced_y[-1] == '0':
                label = str(max(list(map(int, labels))))
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)
                
                label = label + "_end"
                label_list_for_filename.append(label)
            else:
                pass
            
    for data_idx in range(len(sliced_raws)):
        sliced_raw = sliced_raws[data_idx]
        sliced_y = sliced_labels[data_idx]
        sliced_y_map = list(map(int,sliced_y))
        sliced_y = torch.Tensor(sliced_y_map).byte()

        if GLOBAL_DATA['binary_target1'] is not None:
            sliced_y2 = torch.Tensor([GLOBAL_DATA['binary_target1'][i] for i in sliced_y_map]).byte()
        else:
            sliced_y2 = None

        if GLOBAL_DATA['binary_target2'] is not None:
            sliced_y3 = torch.Tensor([GLOBAL_DATA['binary_target2'][i] for i in sliced_y_map]).byte()
        else:
            sliced_y3 = None

        # Convert tensors to numpy arrays for efficient storage
        new_data['RAW_DATA'] = [sliced_raw.cpu().numpy().astype(np.float16)]
        new_data['LABEL1'] = [sliced_y.cpu().numpy().astype(np.uint8)]
        new_data['LABEL2'] = [sliced_y2.cpu().numpy().astype(np.uint8) if sliced_y2 is not None else None]
        new_data['LABEL3'] = [sliced_y3.cpu().numpy().astype(np.uint8) if sliced_y3 is not None else None]

        label = label_list_for_filename[data_idx]
        
        with open(GLOBAL_DATA['data_file_directory'] + "/{}_c{}_label_{}.pkl".format(data_file_name, str(data_idx), str(label)), 'wb') as _f:
            pickle.dump(new_data, _f, protocol=pickle.HIGHEST_PROTOCOL)      
        new_data = {}

def generate_training_data_leadwise_tuh_dev(file):
    sample_rate = GLOBAL_DATA['sample_rate']    # EX) 200Hz
    file_name = ".".join(file.split(".")[:-1])  # EX) $PATH_TO_EEG/train/01_tcp_ar/072/00007235/s003_2010_11_20/00007235_s003_t000
    data_file_name = file_name.split("/")[-1]   # EX) 00007235_s003_t000
    signals, signal_headers, header = highlevel.read_edf(file)
    label_list_c = []
    for idx, signal in enumerate(signals):
        label_noref = signal_headers[idx]['label'].split("-")[0]    # EX) EEG FP1-ref or EEG FP1-LE --> EEG FP1
        label_list_c.append(label_noref)   

    ############################# part 1: labeling  ###############################
    y, y_labels = read_label_file(file_name, GLOBAL_DATA['label_type'])
    if y is None:
        print(f"Warning: Label file not found for {file_name}. Skipping this file.")
        return  # Skip this file if the label file doesn't exist
    signal_sample_rate = int(signal_headers[0]['sample_frequency'])
    if sample_rate > signal_sample_rate:
        return
    if not all(elem in label_list_c for elem in GLOBAL_DATA['label_list']): # if one or more of ['EEG FP1', 'EEG FP2', ... doesn't exist
        return
    # if not any(elem in y_labels for elem in GLOBAL_DATA['disease_type']): # if non-patient exist
    #     return
    y_sampled = label_sampling_tuh(y, GLOBAL_DATA['feature_sample_rate'])
    
    # check if seizure patient or non-seizure patient
    patient_wise_dir = "/".join(file_name.split("/")[:-2])
    edf_list = search_walk({'path': patient_wise_dir, 'extension': ".tse_bi"})
    if not edf_list:  # Try csv_bi if tse_bi not found
        edf_list = search_walk({'path': patient_wise_dir, 'extension': ".csv_bi"})
    patient_bool = False
    for label_file_path in edf_list:
        y, _ = read_label_file(".".join(label_file_path.split(".")[:-1]), GLOBAL_DATA['label_type'])
        if y is None:
            continue
        for line in y:
            if len(line) > 5:
                parts = line.split(" ")
                if len(parts) > 2 and parts[2] != 'bckg':
                    patient_bool = True
                    break
        if patient_bool:
            break

    ############################# part 2: input data filtering #############################
    signal_list = []
    signal_label_list = []
    signal_final_list_raw = []

    for idx, signal in enumerate(signals):
        label = signal_headers[idx]['label'].split("-")[0]
        if label not in GLOBAL_DATA['label_list']:
            continue

        if int(signal_headers[idx]['sample_frequency']) > sample_rate:
            secs = len(signal)/float(signal_sample_rate)
            samps = int(secs*sample_rate)
            x = sci_sig.resample(signal, samps)
            # Apply denoising after resampling
            x = denoise_signal(x, sample_rate, 
                             apply_bandpass=GLOBAL_DATA.get('apply_bandpass', True),
                             apply_notch=GLOBAL_DATA.get('apply_notch', True),
                             use_deepseparator=GLOBAL_DATA.get('use_deepseparator', False),
                             deepseparator_checkpoint=GLOBAL_DATA.get('deepseparator_checkpoint', None))
            signal_list.append(x)
            signal_label_list.append(label)
        else:
            # Apply denoising to signals that don't need resampling
            x = denoise_signal(signal, sample_rate,
                             apply_bandpass=GLOBAL_DATA.get('apply_bandpass', True),
                             apply_notch=GLOBAL_DATA.get('apply_notch', True),
                             use_deepseparator=GLOBAL_DATA.get('use_deepseparator', False),
                             deepseparator_checkpoint=GLOBAL_DATA.get('deepseparator_checkpoint', None))
            signal_list.append(x)
            signal_label_list.append(label)

    if len(signal_label_list) != len(GLOBAL_DATA['label_list']):
        print("Not enough labels: ", signal_label_list)
        return 
    
    for lead_signal in GLOBAL_DATA['label_list']:
        signal_final_list_raw.append(signal_list[signal_label_list.index(lead_signal)])

    new_length = len(signal_final_list_raw[0]) * (float(GLOBAL_DATA['feature_sample_rate']) / GLOBAL_DATA['sample_rate'])
    
    new_length = int(new_length)  # Convert to int for indexing and string operations
    if len(y_sampled) > new_length:
        y_sampled = y_sampled[:new_length]
    elif len(y_sampled) < new_length:
        diff = int(new_length - len(y_sampled))
        if len(y_sampled) > 0:
            y_sampled += y_sampled[-1] * diff
        else:
            # If y_sampled is empty (all labels were skipped), pad with '0' (background)
            print(f"Warning: y_sampled is empty for {data_file_name}. Padding with background labels.")
            y_sampled = '0' * int(new_length)

    # Check if y_sampled is still empty after padding
    if len(y_sampled) == 0:
        print(f"Warning: No valid labels found for {data_file_name}. Skipping this file.")
        return

    y_sampled_np = np.array(list(map(int,y_sampled)))
    new_labels = []
    new_labels_idxs = []

    ############################# part 3: slicing for easy training  #############################
    y_sampled = ["0" if l not in GLOBAL_DATA['selected_diseases'] else l for l in y_sampled]

    if any(l in GLOBAL_DATA['selected_diseases'] for l in y_sampled):
        y_sampled = [str(GLOBAL_DATA['target_dictionary'][int(l)]) if l in GLOBAL_DATA['selected_diseases'] else l for l in y_sampled]

    # slice and save if training data
    new_data = {}
    raw_data = torch.from_numpy(np.array(signal_final_list_raw)).permute(1,0)
    raw_data = raw_data.type(torch.float16)
    
    # max_seg_len_before_seiz_label = GLOBAL_DATA['max_bckg_before_slicelength'] * GLOBAL_DATA['feature_sample_rate']
    # max_seg_len_before_seiz_raw = GLOBAL_DATA['max_bckg_before_slicelength'] * GLOBAL_DATA['sample_rate']
    min_end_margin_label = GLOBAL_DATA['slice_end_margin_length'] * GLOBAL_DATA['feature_sample_rate']
    # min_end_margin_raw = GLOBAL_DATA['slice_end_margin_length'] * GLOBAL_DATA['sample_rate']

    min_seg_len_label = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['feature_sample_rate']
    min_seg_len_raw = GLOBAL_DATA['min_binary_slicelength'] * GLOBAL_DATA['sample_rate']
    # max_seg_len_label = GLOBAL_DATA['max_binary_slicelength'] * GLOBAL_DATA['feature_sample_rate']
    # max_seg_len_raw = GLOBAL_DATA['max_binary_slicelength'] * GLOBAL_DATA['sample_rate']
    
    sliced_raws = []
    sliced_labels = []
    label_list_for_filename = []

    if len(y_sampled) < min_seg_len_label:
        return
    else:
        label_count = {}
        while len(y_sampled) >= min_seg_len_label:
            one_left_slice = False
            sliced_y = y_sampled[:min_seg_len_label]
                
            if (sliced_y[-1] == '0'):
                sliced_raw_data = raw_data[:min_seg_len_raw].permute(1,0)
                raw_data = raw_data[min_seg_len_raw:]
                y_sampled = y_sampled[min_seg_len_label:]

                labels = [x[0] for x in groupby(sliced_y)]
                if (len(labels) == 1) and (labels[0] == '0'):
                    label = "0"
                else:
                    label = ("".join(labels)).replace("0", "")[0]
                sliced_raws.append(sliced_raw_data)
                sliced_labels.append(sliced_y)
                label_list_for_filename.append(label)

            else:
                if '0' in y_sampled[min_seg_len_label:]:
                    end_1 = y_sampled[min_seg_len_label:].index('0')
                    temp_y_sampled = list(y_sampled[min_seg_len_label+end_1:])
                    temp_y_sampled_order = [x[0] for x in groupby(temp_y_sampled)]

                    if len(list(set(temp_y_sampled))) == 1:
                        end_2 = len(temp_y_sampled)
                        one_left_slice = True
                    else:
                        end_2 = temp_y_sampled.index(temp_y_sampled_order[1])

                    if end_2 >= min_end_margin_label:
                        temp_sec = random.randint(1, GLOBAL_DATA['slice_end_margin_length'])
                        temp_seg_len_label = int(min_seg_len_label + (temp_sec * GLOBAL_DATA['feature_sample_rate_arg']) + end_1)
                        temp_seg_len_raw = int(min_seg_len_raw + (temp_sec * GLOBAL_DATA['samplerate']) + (end_1 * GLOBAL_DATA['fsr_sr_ratio']))
                    else:
                        if one_left_slice:
                            temp_label = end_2
                        else:
                            temp_label = end_2 // 2

                        temp_seg_len_label = int(min_seg_len_label + temp_label + end_1)
                        temp_seg_len_raw = int(min_seg_len_raw + (temp_label * GLOBAL_DATA['fsr_sr_ratio']) + (end_1 * GLOBAL_DATA['fsr_sr_ratio']))

                    sliced_y = y_sampled[:temp_seg_len_label]
                    sliced_raw_data = raw_data[:temp_seg_len_raw].permute(1,0)
                    raw_data = raw_data[temp_seg_len_raw:]
                    y_sampled = y_sampled[temp_seg_len_label:]

                    labels = [x[0] for x in groupby(sliced_y)]
                    if (len(labels) == 1) and (labels[0] == '0'):
                        label = "0"
                    else:
                        label = ("".join(labels)).replace("0", "")[0]
                    sliced_raws.append(sliced_raw_data)
                    sliced_labels.append(sliced_y)
                    label_list_for_filename.append(label)
                else:
                    sliced_y = y_sampled[:]
                    sliced_raw_data = raw_data[:].permute(1,0)
                    raw_data = []
                    y_sampled = []

                    labels = [x[0] for x in groupby(sliced_y)]
                    if (len(labels) == 1) and (labels[0] == '0'):
                        label = "0"
                    else:
                        label = ("".join(labels)).replace("0", "")[0]
                    sliced_raws.append(sliced_raw_data)
                    sliced_labels.append(sliced_y)
                    label_list_for_filename.append(label)
            
    for data_idx in range(len(sliced_raws)):
        sliced_raw = sliced_raws[data_idx]
        sliced_y = sliced_labels[data_idx]
        sliced_y_map = list(map(int,sliced_y))
        sliced_y = torch.Tensor(sliced_y_map).byte()

        if GLOBAL_DATA['binary_target1'] is not None:
            sliced_y2 = torch.Tensor([GLOBAL_DATA['binary_target1'][i] for i in sliced_y_map]).byte()
        else:
            sliced_y2 = None

        if GLOBAL_DATA['binary_target2'] is not None:
            sliced_y3 = torch.Tensor([GLOBAL_DATA['binary_target2'][i] for i in sliced_y_map]).byte()
        else:
            sliced_y3 = None

        # Convert tensors to numpy arrays for efficient storage
        new_data['RAW_DATA'] = [sliced_raw.cpu().numpy().astype(np.float16)]
        new_data['LABEL1'] = [sliced_y.cpu().numpy().astype(np.uint8)]
        new_data['LABEL2'] = [sliced_y2.cpu().numpy().astype(np.uint8) if sliced_y2 is not None else None]
        new_data['LABEL3'] = [sliced_y3.cpu().numpy().astype(np.uint8) if sliced_y3 is not None else None]

        label = label_list_for_filename[data_idx]
        
        with open(GLOBAL_DATA['data_file_directory'] + "/{}_c{}_len{}_label_{}.pkl".format(data_file_name, str(data_idx), str(len(sliced_y)), str(label)), 'wb') as _f:
            pickle.dump(new_data, _f, protocol=pickle.HIGHEST_PROTOCOL)      
        new_data = {}


def main(args):
    save_directory = args.save_directory
    if save_directory is None:
        raise ValueError("--save_directory (-sp) is required. Please provide a path to save the processed data.")
    
    # Normalize save_directory to remove trailing slash
    save_directory = save_directory.rstrip('/')
    
    data_type = args.data_type
    dataset = args.dataset
    label_type = args.label_type
    sample_rate = args.samplerate
    cpu_num = args.cpu_num
    feature_type = args.feature_type
    feature_sample_rate = args.feature_sample_rate
    task_type = args.task_type
    # Modified to add _denoised suffix to output directory
    if args.use_deepseparator:
        data_file_directory = save_directory + "/dataset-{}_task-{}_datatype-{}_v6_deepseparator".format(dataset, task_type, data_type)
    else:
        data_file_directory = save_directory + "/dataset-{}_task-{}_datatype-{}_v6_denoised".format(dataset, task_type, data_type)
    
    
    labels = ['EEG FP1', 'EEG FP2', 'EEG F3', 'EEG F4', 'EEG F7', 'EEG F8',  
                    'EEG C3', 'EEG C4', 'EEG CZ', 'EEG T3', 'EEG T4', 
                    'EEG P3', 'EEG P4', 'EEG O1', 'EEG O2', 'EEG T5', 'EEG T6', 'EEG PZ', 'EEG FZ']

    # eeg_data_directory = "$PATH_TO_EEG/{}".format(data_type)
    # eeg_data_directory = "/mnt/aitrics_ext/ext01/shared/edf/tuh_final/{}".format(data_type)
    eeg_data_directory = "/srv/data/biosignals/isip.piconepress.com/{}".format(data_type)
    
    if label_type == "tse":
        disease_labels =  {'bckg': 0, 'cpsz': 1, 'mysz': 2, 'gnsz': 3, 'fnsz': 4, 'tnsz': 5, 'tcsz': 6, 'spsz': 7, 'absz': 8}
    elif label_type == "tse_bi":
        disease_labels =  {'bckg': 0, 'seiz': 1}
        args.disease_type = ['seiz']
    disease_labels_inv = {v: k for k, v in disease_labels.items()}
    
    edf_list1 = search_walk({'path': eeg_data_directory, 'extension': ".edf"})
    edf_list2 = search_walk({'path': eeg_data_directory, 'extension': ".EDF"})
    if edf_list2:
        edf_list = edf_list1 + edf_list2
    else:
        edf_list = edf_list1

    if os.path.isdir(data_file_directory):
        os.system("rm -rf {}".format(data_file_directory))
    os.system("mkdir {}".format(data_file_directory))

    GLOBAL_DATA['label_list'] = labels # 'EEG FP1', 'EEG FP2', 'EEG F3', ...
    GLOBAL_DATA['disease_labels'] = disease_labels #  {'bckg': 0, 'cpsz': 1, 'mysz': 2, ...
    GLOBAL_DATA['disease_labels_inv'] = disease_labels_inv #  {0:'bckg', 1:'cpsz', 2:'mysz', ...
    GLOBAL_DATA['data_file_directory'] = data_file_directory
    GLOBAL_DATA['label_type'] = label_type # "tse_bi" ...
    GLOBAL_DATA['feature_type'] = feature_type
    GLOBAL_DATA['feature_sample_rate'] = feature_sample_rate
    GLOBAL_DATA['sample_rate'] = sample_rate
    GLOBAL_DATA['fsr_sr_ratio'] = (sample_rate // feature_sample_rate)
    GLOBAL_DATA['min_binary_slicelength'] = args.min_binary_slicelength
    GLOBAL_DATA['min_binary_edge_seiz'] = args.min_binary_edge_seiz
    GLOBAL_DATA['slice_end_margin_length'] = args.slice_end_margin_length
    GLOBAL_DATA['samplerate'] = args.samplerate
    GLOBAL_DATA['feature_sample_rate_arg'] = args.feature_sample_rate

    target_dictionary = {0:0}
    selected_diseases = []
    for idx, i in enumerate(args.disease_type):
        selected_diseases.append(str(disease_labels[i]))
        target_dictionary[disease_labels[i]] = idx + 1
    
    GLOBAL_DATA['disease_type'] = args.disease_type # args.disease_type == ['gnsz', 'fnsz', 'spsz', 'cpsz', 'absz', 'tnsz', 'tcsz', 'mysz']
    GLOBAL_DATA['target_dictionary'] = target_dictionary # {0: 0, 4: 1, 5: 2, 8: 3, 2: 4, 9: 5, 6: 6, 7: 7, 3: 8}
    GLOBAL_DATA['selected_diseases'] = selected_diseases # ['4', '5', '8', '2', '9', '6', '7', '3']
    GLOBAL_DATA['binary_target1'] = args.binary_target1
    GLOBAL_DATA['binary_target2'] = args.binary_target2
    
    # DeepSeparator settings
    GLOBAL_DATA['use_deepseparator'] = args.use_deepseparator
    GLOBAL_DATA['deepseparator_checkpoint'] = args.deepseparator_checkpoint
    GLOBAL_DATA['apply_bandpass'] = args.apply_bandpass
    GLOBAL_DATA['apply_notch'] = args.apply_notch
    # By default, apply traditional filters first when using DeepSeparator
    # Set to False only if --skip_traditional_filters is used
    GLOBAL_DATA['apply_traditional_filters_first'] = not args.skip_traditional_filters

    print("########## Preprocessor Setting Information (DENOISED) ##########")
    print("Number of EDF files: ", len(edf_list))
    print("Output directory: ", data_file_directory)
    for i in GLOBAL_DATA:
        print("{}: {}".format(i, GLOBAL_DATA[i]))
    with open(data_file_directory + '/preprocess_info.infopkl', 'wb') as pkl:
        pickle.dump(GLOBAL_DATA, pkl, protocol=pickle.HIGHEST_PROTOCOL)
    print("################ Preprocess with DENOISING begins... ################\n")
    
    if (task_type == "binary") and (args.data_type == "train"):
        run_multi_process(generate_training_data_leadwise_tuh_train_final, edf_list, n_processes=cpu_num)
    elif (task_type == "binary") and (args.data_type == "dev"):
        run_multi_process(generate_training_data_leadwise_tuh_dev, edf_list, n_processes=cpu_num)
        
if __name__ == '__main__':
    # make sure all edf file name different!!! if not, additional coding is necessary
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-sd', type=int, default=1004,
                        help='Random seed number')
    parser.add_argument('--samplerate', '-sr', type=int, default=200,
                        help='Sample Rate')
    parser.add_argument('--save_directory', '-sp', type=str, required=True,
                        help='Path to save data')
    parser.add_argument('--label_type', '-lt', type=str,
                        default='tse',
                        help='tse_bi = global with binary label, tse = global with various labels, cae = severance CAE seizure label.')                      
    parser.add_argument('--cpu_num', '-cn', type=int,
                        default=32,
                        help='select number of available cpus')   
    parser.add_argument('--feature_type', '-ft', type=str,
                        default=['rawsignal'])   
    parser.add_argument('--feature_sample_rate', '-fsr', type=int,
                        default=50,
                        help='select features sample rate')   
    parser.add_argument('--dataset', '-st', type=str,
                        default='tuh',
                        choices=['tuh'])                   
    parser.add_argument('--data_type', '-dt', type=str,
                        default='train',
                        choices=['train', 'dev'])                   
    parser.add_argument('--task_type', '-tt', type=str,
                        default='binary',
                        choices=['anomaly', 'multiclassification', 'binary'])                   

    ##### Target Grouping #####
    parser.add_argument('--disease_type', type=list, default=['gnsz', 'fnsz', 'spsz', 'cpsz', 'absz', 'tnsz', 'tcsz', 'mysz'], choices=['gnsz', 'fnsz', 'spsz', 'cpsz', 'absz', 'tnsz', 'tcsz', 'mysz'])

    ### for binary detector ###
    # key numbers represent index of --disease_type + 1  ### -1 is "not being used"
    parser.add_argument('--binary_target1', type=dict, default={0:0, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1})
    parser.add_argument('--binary_target2', type=dict, default={0:0, 1:1, 2:2, 3:2, 4:2, 5:1, 6:3, 7:4, 8:5})
    parser.add_argument('--min_binary_slicelength', type=int, default=30)           
    parser.add_argument('--min_binary_edge_seiz', type=int, default=3)
    parser.add_argument('--slice_end_margin_length', type=int, default=5,
                        help='Slice end margin length for dev data processing')
    
    ### DeepSeparator denoising options ###
    parser.add_argument('--use_deepseparator', action='store_true',
                        help='Use DeepSeparator model for denoising instead of traditional filters')
    parser.add_argument('--deepseparator_checkpoint', type=str, default=None,
                        help='Path to DeepSeparator checkpoint file (e.g., DeepSeparator/code/checkpoint/DeepSeparator.pkl)')
    parser.add_argument('--apply_bandpass', action='store_true', default=True,
                        help='Apply bandpass filter (0.5-70 Hz). Default: True. Ignored if --use_deepseparator and not --apply_traditional_filters_first')
    parser.add_argument('--no_apply_bandpass', dest='apply_bandpass', action='store_false',
                        help='Disable bandpass filter')
    parser.add_argument('--apply_notch', action='store_true', default=True,
                        help='Apply notch filter for line noise. Default: True. Ignored if --use_deepseparator and not --apply_traditional_filters_first')
    parser.add_argument('--no_apply_notch', dest='apply_notch', action='store_false',
                        help='Disable notch filter')
    parser.add_argument('--skip_traditional_filters', action='store_true', default=False,
                        help='Skip traditional filters (bandpass/notch) when using DeepSeparator. By default, traditional filters are applied first, then DeepSeparator')
    
    args = parser.parse_args()
    
    # Validate DeepSeparator arguments
    if args.use_deepseparator and args.deepseparator_checkpoint is None:
        # Try to find default checkpoint path
        default_checkpoint = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                         'DeepSeparator', 'code', 'checkpoint', 'DeepSeparator.pkl')
        if os.path.exists(default_checkpoint):
            args.deepseparator_checkpoint = default_checkpoint
            print(f"Using default DeepSeparator checkpoint: {default_checkpoint}")
        else:
            raise ValueError("--deepseparator_checkpoint must be provided when --use_deepseparator is enabled")
    
    main(args)

