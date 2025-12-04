# Copyright (c) 2022, Kwanhyung Lee. All rights reserved.
#
# Licensed under the MIT License; 
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt, iirnotch
import os

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("Warning: PyWavelets (pywt) not installed. SWT denoising will not be available.")
    print("Install with: pip install PyWavelets")


def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=200, order=4):
    """
    Apply bandpass filter to EEG signal.
    
    Args:
        signal: Input signal as numpy array or torch.Tensor
                Shape: (time_steps,) or (channels, time_steps) or (batch, time_steps, channels)
        lowcut: Low cutoff frequency in Hz (default: 0.5)
        highcut: High cutoff frequency in Hz (default: 40.0)
        fs: Sampling frequency in Hz (default: 200)
        order: Filter order (default: 4)
    
    Returns:
        Filtered signal with same shape and type as input
    """
    # Convert to numpy for processing
    is_tensor = isinstance(signal, torch.Tensor)
    if is_tensor:
        device = signal.device
        signal_np = signal.detach().cpu().numpy()
        original_shape = signal_np.shape
    else:
        signal_np = np.asarray(signal)
        original_shape = signal_np.shape
    
    # Handle different input shapes
    if len(original_shape) == 1:  # (time_steps,)
        signal_reshaped = signal_np.reshape(-1, 1)
        needs_reshape = True
        n_channels = 1
    elif len(original_shape) == 2:
        if original_shape[0] < original_shape[1]:  # (channels, time_steps)
            signal_reshaped = signal_np.T  # Transpose to (time_steps, channels)
            n_channels = original_shape[0]
        else:  # (time_steps, channels)
            signal_reshaped = signal_np
            n_channels = original_shape[1]
        needs_reshape = False
    elif len(original_shape) == 3:  # (batch, time_steps, channels)
        batch_size, time_steps, n_channels = original_shape
        signal_reshaped = signal_np.reshape(-1, n_channels)
        needs_reshape = True
        batch_dim = batch_size
    else:
        raise ValueError(f"Unsupported signal shape: {original_shape}")
    
    # Design bandpass filter
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    
    if low >= 1.0 or high >= 1.0:
        raise ValueError(f"Cutoff frequencies must be less than Nyquist frequency ({nyquist} Hz)")
    
    b, a = butter(order, [low, high], btype='band')
    
    # Apply filter to each channel
    filtered = np.zeros_like(signal_reshaped)
    for ch in range(n_channels):
        filtered[:, ch] = filtfilt(b, a, signal_reshaped[:, ch])
    
    # Reshape back to original
    if needs_reshape:
        if len(original_shape) == 1:
            filtered = filtered[:, 0]
        elif len(original_shape) == 3:
            filtered = filtered.reshape(batch_dim, -1, n_channels)
    
    # Convert back to tensor if needed
    if is_tensor:
        filtered = torch.from_numpy(filtered).float().to(device)
    
    return filtered


def notch_filter(signal, freq=50.0, fs=200, Q=30.0):
    """
    Apply notch filter to remove line noise (50 Hz or 60 Hz).
    
    Args:
        signal: Input signal as numpy array or torch.Tensor
                Shape: (time_steps,) or (channels, time_steps) or (batch, time_steps, channels)
        freq: Notch frequency in Hz (default: 50.0, use 60.0 for US power lines)
        fs: Sampling frequency in Hz (default: 200)
        Q: Quality factor - higher Q means narrower notch (default: 30.0)
    
    Returns:
        Filtered signal with same shape and type as input
    """
    # Convert to numpy for processing
    is_tensor = isinstance(signal, torch.Tensor)
    if is_tensor:
        device = signal.device
        signal_np = signal.detach().cpu().numpy()
        original_shape = signal_np.shape
    else:
        signal_np = np.asarray(signal)
        original_shape = signal_np.shape
    
    # Handle different input shapes
    if len(original_shape) == 1:  # (time_steps,)
        signal_reshaped = signal_np.reshape(-1, 1)
        needs_reshape = True
        n_channels = 1
    elif len(original_shape) == 2:
        if original_shape[0] < original_shape[1]:  # (channels, time_steps)
            signal_reshaped = signal_np.T  # Transpose to (time_steps, channels)
            n_channels = original_shape[0]
        else:  # (time_steps, channels)
            signal_reshaped = signal_np
            n_channels = original_shape[1]
        needs_reshape = False
    elif len(original_shape) == 3:  # (batch, time_steps, channels)
        batch_size, time_steps, n_channels = original_shape
        signal_reshaped = signal_np.reshape(-1, n_channels)
        needs_reshape = True
        batch_dim = batch_size
    else:
        raise ValueError(f"Unsupported signal shape: {original_shape}")
    
    # Design notch filter
    b, a = iirnotch(freq, Q, fs)
    
    # Apply filter to each channel
    filtered = np.zeros_like(signal_reshaped)
    for ch in range(n_channels):
        filtered[:, ch] = filtfilt(b, a, signal_reshaped[:, ch])
    
    # Reshape back to original
    if needs_reshape:
        if len(original_shape) == 1:
            filtered = filtered[:, 0]
        elif len(original_shape) == 3:
            filtered = filtered.reshape(batch_dim, -1, n_channels)
    
    # Convert back to tensor if needed
    if is_tensor:
        filtered = torch.from_numpy(filtered).float().to(device)
    
    return filtered


def swt_denoise(signal, wavelet='db4', mode='symmetric', threshold_mode='soft', 
                threshold=None, levels=None, sample_rate=200):
    """
    Apply Stationary Wavelet Transform (SWT) denoising to EEG signal.
    
    SWT is shift-invariant and preserves signal length, making it ideal for real-time
    EEG denoising. Uses Daubechies 4 (db4) wavelet by default.
    
    Args:
        signal: Input signal as numpy array or torch.Tensor
                Shape: (time_steps,) or (channels, time_steps) or (batch, time_steps, channels)
        wavelet: Wavelet type (default: 'db4' for Daubechies 4)
        mode: Signal extension mode - 'symmetric', 'periodic', 'reflect', etc. (default: 'symmetric')
        threshold_mode: 'soft' or 'hard' thresholding (default: 'soft')
        threshold: Threshold value. If None, uses VisuShrink (default: None)
        levels: Number of decomposition levels. If None, auto-calculates based on signal length (default: None)
        sample_rate: Sampling frequency in Hz, used for auto-calculating levels (default: 200)
    
    Returns:
        Denoised signal with same shape and type as input
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets (pywt) is required for SWT denoising. Install with: pip install PyWavelets")
    
    # Convert to numpy for processing
    is_tensor = isinstance(signal, torch.Tensor)
    if is_tensor:
        device = signal.device
        signal_np = signal.detach().cpu().numpy()
        original_shape = signal_np.shape
    else:
        signal_np = np.asarray(signal)
        original_shape = signal_np.shape
    
    # Handle different input shapes
    if len(original_shape) == 1:  # (time_steps,)
        signal_reshaped = signal_np.reshape(-1, 1)
        needs_reshape = True
        n_channels = 1
    elif len(original_shape) == 2:
        if original_shape[0] < original_shape[1]:  # (channels, time_steps)
            signal_reshaped = signal_np.T  # Transpose to (time_steps, channels)
            n_channels = original_shape[0]
        else:  # (time_steps, channels)
            signal_reshaped = signal_np
            n_channels = original_shape[1]
        needs_reshape = False
    elif len(original_shape) == 3:  # (batch, time_steps, channels)
        batch_size, time_steps, n_channels = original_shape
        signal_reshaped = signal_np.reshape(-1, n_channels)
        needs_reshape = True
        batch_dim = batch_size
    else:
        raise ValueError(f"Unsupported signal shape: {original_shape}")
    
    time_steps, n_channels = signal_reshaped.shape
    
    # Auto-calculate decomposition levels if not provided
    if levels is None:
        # For 200 Hz sampling rate, 4-5 levels captures 0-50 Hz range well
        # Maximum levels is limited by signal length: 2^levels <= signal_length
        max_levels = pywt.swt_max_level(time_steps)
        # Use 4-5 levels for EEG, but not more than max
        levels = min(4, max_levels) if max_levels > 0 else 1
    
    # Ensure signal length is even (required for SWT)
    if time_steps % 2 != 0:
        # Pad with last value to make even
        signal_reshaped = np.pad(signal_reshaped, ((0, 1), (0, 0)), mode='edge')
        needs_padding = True
        padded_time_steps = time_steps + 1
    else:
        needs_padding = False
        padded_time_steps = time_steps
    
    # Apply SWT denoising to each channel
    denoised = np.zeros_like(signal_reshaped)
    
    for ch in range(n_channels):
        channel_signal = signal_reshaped[:, ch]
        
        # Perform SWT decomposition
        # pywt.swt returns list of tuples: [(cA_n, cD_n), (cA_n-1, cD_n-1), ..., (cA_1, cD_1)]
        # where each tuple contains (approximation, detail) coefficients for that level
        # Note: coeffs[0] is the finest level (highest frequency details)
        #       coeffs[-1] is the coarsest level (lowest frequency approximation)
        coeffs = pywt.swt(channel_signal, wavelet, level=levels)
        
        # Calculate threshold if not provided (VisuShrink)
        if threshold is None:
            # Use median absolute deviation (MAD) for robust threshold estimation
            # Estimate noise from finest detail coefficients (highest frequency)
            detail_coeffs = coeffs[0][1]  # Finest detail coefficients
            sigma = np.median(np.abs(detail_coeffs)) / 0.6745  # MAD estimate
            threshold = sigma * np.sqrt(2 * np.log(len(channel_signal)))  # VisuShrink
        
        # Threshold the coefficients
        thresholded_coeffs = []
        for i, (cA, cD) in enumerate(coeffs):
            # Keep approximation coefficients (low frequency) as-is
            # Threshold detail coefficients (high frequency - noise)
            if threshold_mode == 'soft':
                cD_thresh = pywt.threshold(cD, threshold, mode='soft')
            elif threshold_mode == 'hard':
                cD_thresh = pywt.threshold(cD, threshold, mode='hard')
            else:
                raise ValueError(f"Unknown threshold_mode: {threshold_mode}. Use 'soft' or 'hard'")
            
            thresholded_coeffs.append((cA, cD_thresh))
        
        # Reconstruct signal using inverse SWT
        denoised_channel = pywt.iswt(thresholded_coeffs, wavelet)
        denoised[:, ch] = denoised_channel
    
    # Remove padding if added
    if needs_padding:
        denoised = denoised[:-1, :]
    
    # Reshape back to original
    if needs_reshape:
        if len(original_shape) == 1:
            denoised = denoised[:, 0]
        elif len(original_shape) == 3:
            denoised = denoised.reshape(batch_dim, -1, n_channels)
    
    # Convert back to tensor if needed
    if is_tensor:
        denoised = torch.from_numpy(denoised).float().to(device)
    
    return denoised


def modwt_denoise(signal, wavelet='db4', threshold_mode='soft', 
                  threshold=None, levels=None, sample_rate=200):
    """
    Apply Maximal Overlap Discrete Wavelet Transform (MODWT) denoising to EEG signal.
    
    MODWT is shift-invariant and preserves signal length, making it ideal for real-time
    EEG denoising. Unlike SWT, MODWT doesn't require even-length signals. Uses Daubechies 4 (db4) wavelet by default.
    
    MODWT is similar to SWT but uses a different algorithm (maximal overlap) that provides
    better energy conservation and doesn't require signal length to be a multiple of 2^levels.
    
    Args:
        signal: Input signal as numpy array or torch.Tensor
                Shape: (time_steps,) or (channels, time_steps) or (batch, time_steps, channels)
        wavelet: Wavelet type (default: 'db4' for Daubechies 4)
        threshold_mode: 'soft' or 'hard' thresholding (default: 'soft')
        threshold: Threshold value. If None, uses VisuShrink (default: None)
        levels: Number of decomposition levels. If None, auto-calculates based on signal length (default: None)
        sample_rate: Sampling frequency in Hz, used for auto-calculating levels (default: 200)
    
    Returns:
        Denoised signal with same shape and type as input
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets (pywt) is required for MODWT denoising. Install with: pip install PyWavelets")
    
    # Convert to numpy for processing
    is_tensor = isinstance(signal, torch.Tensor)
    if is_tensor:
        device = signal.device
        signal_np = signal.detach().cpu().numpy()
        original_shape = signal_np.shape
    else:
        signal_np = np.asarray(signal)
        original_shape = signal_np.shape
    
    # Handle different input shapes
    if len(original_shape) == 1:  # (time_steps,)
        signal_reshaped = signal_np.reshape(-1, 1)
        needs_reshape = True
        n_channels = 1
    elif len(original_shape) == 2:
        if original_shape[0] < original_shape[1]:  # (channels, time_steps)
            signal_reshaped = signal_np.T  # Transpose to (time_steps, channels)
            n_channels = original_shape[0]
        else:  # (time_steps, channels)
            signal_reshaped = signal_np
            n_channels = original_shape[1]
        needs_reshape = False
    elif len(original_shape) == 3:  # (batch, time_steps, channels)
        batch_size, time_steps, n_channels = original_shape
        signal_reshaped = signal_np.reshape(-1, n_channels)
        needs_reshape = True
        batch_dim = batch_size
    else:
        raise ValueError(f"Unsupported signal shape: {original_shape}")
    
    time_steps, n_channels = signal_reshaped.shape
    
    # Auto-calculate decomposition levels if not provided
    if levels is None:
        # For 200 Hz sampling rate, 4-5 levels captures 0-50 Hz range well
        # MODWT doesn't have the same length restrictions as SWT
        # Maximum levels is limited by signal length: level <= log2(signal_length)
        max_levels = int(np.log2(time_steps)) if time_steps > 1 else 1
        # Use 4-5 levels for EEG, but not more than max
        levels = min(4, max_levels) if max_levels > 0 else 1
    
    # Apply MODWT denoising to each channel
    denoised = np.zeros_like(signal_reshaped)
    
    for ch in range(n_channels):
        channel_signal = signal_reshaped[:, ch]
        
        # Perform MODWT decomposition
        # pywt.modwt returns list: [cA_n, cD_n, cD_n-1, ..., cD_1]
        # where cA_n is the approximation at level n, and cD_i are detail coefficients
        # Note: coeffs[0] is the approximation (lowest frequency)
        #       coeffs[1] is the finest detail (highest frequency)
        #       coeffs[-1] is the coarsest detail
        coeffs = pywt.modwt(channel_signal, wavelet, level=levels)
        
        # Calculate threshold if not provided (VisuShrink)
        if threshold is None:
            # Use median absolute deviation (MAD) for robust threshold estimation
            # Estimate noise from finest detail coefficients (highest frequency)
            # In MODWT, coeffs[1] is the finest detail
            detail_coeffs = coeffs[1]  # Finest detail coefficients
            sigma = np.median(np.abs(detail_coeffs)) / 0.6745  # MAD estimate
            threshold = sigma * np.sqrt(2 * np.log(len(channel_signal)))  # VisuShrink
        
        # Threshold the coefficients
        thresholded_coeffs = [coeffs[0]]  # Keep approximation coefficients as-is
        
        # Threshold detail coefficients (high frequency - noise)
        for i in range(1, len(coeffs)):
            cD = coeffs[i]
            if threshold_mode == 'soft':
                cD_thresh = pywt.threshold(cD, threshold, mode='soft')
            elif threshold_mode == 'hard':
                cD_thresh = pywt.threshold(cD, threshold, mode='hard')
            else:
                raise ValueError(f"Unknown threshold_mode: {threshold_mode}. Use 'soft' or 'hard'")
            
            thresholded_coeffs.append(cD_thresh)
        
        # Reconstruct signal using inverse MODWT
        denoised_channel = pywt.imodwt(thresholded_coeffs, wavelet)
        denoised[:, ch] = denoised_channel
    
    # Reshape back to original
    if needs_reshape:
        if len(original_shape) == 1:
            denoised = denoised[:, 0]
        elif len(original_shape) == 3:
            denoised = denoised.reshape(batch_dim, -1, n_channels)
    
    # Convert back to tensor if needed
    if is_tensor:
        denoised = torch.from_numpy(denoised).float().to(device)
    
    return denoised


def apply_denoising_pipeline(signal, sample_rate=200, use_bandpass=True, use_notch=True, 
                            use_swt=False, use_modwt=False, device=None,
                            bandpass_low=0.5, bandpass_high=40.0, notch_freq=50.0,
                            swt_wavelet='db4', swt_threshold_mode='soft', swt_levels=None,
                            modwt_wavelet='db4', modwt_threshold_mode='soft', modwt_levels=None):
    """
    Apply a complete denoising pipeline (bandpass + notch + optional SWT/MODWT).
    
    Recommended order: Bandpass → Notch → SWT/MODWT
    - Bandpass removes frequencies outside EEG range
    - Notch removes line noise (50/60 Hz)
    - SWT/MODWT removes remaining noise while preserving signal structure
    
    Note: Use either SWT or MODWT, not both. MODWT is preferred for signals of arbitrary length.
    
    Args:
        signal: Input signal (numpy array or torch.Tensor)
        sample_rate: Sampling frequency in Hz
        use_bandpass: Whether to apply bandpass filter
        use_notch: Whether to apply notch filter
        use_swt: Whether to apply SWT denoising
        use_modwt: Whether to apply MODWT denoising (use either SWT or MODWT, not both)
        device: torch device (not used for wavelet transforms, kept for compatibility)
        bandpass_low: Low cutoff for bandpass (Hz)
        bandpass_high: High cutoff for bandpass (Hz)
        notch_freq: Notch frequency (Hz)
        swt_wavelet: Wavelet type for SWT (default: 'db4')
        swt_threshold_mode: 'soft' or 'hard' thresholding for SWT (default: 'soft')
        swt_levels: Number of SWT decomposition levels (None = auto)
        modwt_wavelet: Wavelet type for MODWT (default: 'db4')
        modwt_threshold_mode: 'soft' or 'hard' thresholding for MODWT (default: 'soft')
        modwt_levels: Number of MODWT decomposition levels (None = auto)
    
    Returns:
        Denoised signal
    """
    denoised = signal
    
    # Step 1: Bandpass filter (removes frequencies outside EEG range)
    if use_bandpass:
        denoised = bandpass_filter(denoised, lowcut=bandpass_low, highcut=bandpass_high, fs=sample_rate)
    
    # Step 2: Notch filter (removes line noise)
    if use_notch:
        denoised = notch_filter(denoised, freq=notch_freq, fs=sample_rate)
    
    # Step 3: Wavelet denoising (removes remaining noise, preserves signal structure)
    # Use either SWT or MODWT, not both
    if use_swt and use_modwt:
        raise ValueError("Cannot use both SWT and MODWT. Choose one.")
    
    if use_swt:
        denoised = swt_denoise(denoised, wavelet=swt_wavelet, threshold_mode=swt_threshold_mode, 
                              levels=swt_levels, sample_rate=sample_rate)
    elif use_modwt:
        denoised = modwt_denoise(denoised, wavelet=modwt_wavelet, threshold_mode=modwt_threshold_mode, 
                                levels=modwt_levels, sample_rate=sample_rate)
    
    return denoised

