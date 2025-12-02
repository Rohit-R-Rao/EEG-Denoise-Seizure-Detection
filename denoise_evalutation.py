# -*- coding: utf-8 -*-
"""
Denoising Evaluation Script
Compares original preprocessed data with denoised preprocessed data
Computes 7 metrics: APR, pSNR, PSD Cosine Similarity, KL Divergence, BPE, Kurtosis Change, Skewness Change
"""

import numpy as np
import pickle
import os
import glob
import re
from scipy import signal as sci_sig
from scipy.stats import kurtosis, skew
from scipy.spatial.distance import jensenshannon
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Hardcoded parameters
DATA_TYPE = 'dev'  # 'train' or 'dev'
SAVE_DIRECTORY = 'out/'
SAMPLE_RATE = 200  # Hz
FEATURE_SAMPLE_RATE = 50  # Hz
OUTPUT_FILE = 'denoising_evaluation_results.csv'
MAX_FILES = None  # Set to None to process all files, or a number to limit

# EEG frequency bands
EEG_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100)
}

# Artifact bands
ARTIFACT_BANDS = {
    'eye_blinks': (0, 4),
    'ecg': (1, 3),
    'emg': (20, 80)
}

# PSD computation parameters
PSD_NPERSEG = 256  # Segment length for Welch's method
PSD_OVERLAP = 128  # Overlap between segments


def extract_base_filename(filename):
    """Extract base filename (before _c{idx}) from pickle filename"""
    basename = os.path.basename(filename)
    # Filename format: {base}_c{idx}_... or {base}_c{idx}_len{len}_label_{label}.pkl
    if '_c' in basename:
        base = basename.split('_c')[0]
        return base
    return basename.replace('.pkl', '')

def extract_chunk_info(filename):
    """Extract chunk index, length, and label from filename"""
    basename = os.path.basename(filename)
    chunk_idx = None
    length = None
    label = None
    
    # Try to extract chunk index (after _c)
    if '_c' in basename:
        parts = basename.split('_c')
        if len(parts) > 1:
            chunk_part = parts[1]
            # Extract chunk index (first number after _c)
            match = re.search(r'^(\d+)', chunk_part)
            if match:
                chunk_idx = int(match.group(1))
            
            # Extract length (after _len)
            if '_len' in chunk_part:
                len_match = re.search(r'_len(\d+)', chunk_part)
                if len_match:
                    length = int(len_match.group(1))
            
            # Extract label (after _label_)
            if '_label_' in chunk_part:
                label_match = re.search(r'_label_([^\.]+)', chunk_part)
                if label_match:
                    label = label_match.group(1)
    
    return chunk_idx, length, label

def load_data_files(original_dir, denoised_dir):
    """Load and match pickle files from both directories"""
    print(f"Loading files from:\n  Original: {original_dir}\n  Denoised: {denoised_dir}")
    
    # Get all pickle files
    original_files = glob.glob(os.path.join(original_dir, "*.pkl"))
    denoised_files = glob.glob(os.path.join(denoised_dir, "*.pkl"))
    
    # Filter out preprocess_info.infopkl
    original_files = [f for f in original_files if 'preprocess_info' not in f]
    denoised_files = [f for f in denoised_files if 'preprocess_info' not in f]
    
    print(f"Found {len(original_files)} original files and {len(denoised_files)} denoised files")
    
    # Group files by base filename
    original_by_base = {}
    for f in original_files:
        base = extract_base_filename(f)
        if base not in original_by_base:
            original_by_base[base] = []
        original_by_base[base].append(f)
    
    denoised_by_base = {}
    for f in denoised_files:
        base = extract_base_filename(f)
        if base not in denoised_by_base:
            denoised_by_base[base] = []
        denoised_by_base[base].append(f)
    
    # Match files: first try exact filename match, then match by base + chunk index
    matched_pairs = []
    exact_matches = 0
    chunk_matches = 0
    
    # First pass: exact filename matches
    original_basenames = {os.path.basename(f): f for f in original_files}
    denoised_basenames = {os.path.basename(f): f for f in denoised_files}
    matched_denoised = set()
    
    for basename in original_basenames:
        if basename in denoised_basenames:
            matched_pairs.append((original_basenames[basename], denoised_basenames[basename]))
            matched_denoised.add(denoised_basenames[basename])
            exact_matches += 1
    
    # Second pass: match by base filename and chunk index
    for base in original_by_base:
        if base not in denoised_by_base:
            continue
        
        # Create dictionaries keyed by chunk index
        orig_by_chunk = {}
        for f in original_by_base[base]:
            chunk_idx, _, _ = extract_chunk_info(f)
            if chunk_idx is not None:
                if chunk_idx not in orig_by_chunk:
                    orig_by_chunk[chunk_idx] = []
                orig_by_chunk[chunk_idx].append(f)
        
        denoised_by_chunk = {}
        for f in denoised_by_base[base]:
            if f in matched_denoised:
                continue  # Already matched
            chunk_idx, _, _ = extract_chunk_info(f)
            if chunk_idx is not None:
                if chunk_idx not in denoised_by_chunk:
                    denoised_by_chunk[chunk_idx] = []
                denoised_by_chunk[chunk_idx].append(f)
        
        # Match by chunk index
        for chunk_idx in orig_by_chunk:
            if chunk_idx in denoised_by_chunk:
                # Match first available file from each list
                orig_files = orig_by_chunk[chunk_idx]
                denoised_files_list = denoised_by_chunk[chunk_idx]
                min_len = min(len(orig_files), len(denoised_files_list))
                for i in range(min_len):
                    if denoised_files_list[i] not in matched_denoised:
                        matched_pairs.append((orig_files[i], denoised_files_list[i]))
                        matched_denoised.add(denoised_files_list[i])
                        chunk_matches += 1
    
    print(f"Matched {len(matched_pairs)} file pairs ({exact_matches} exact matches, {chunk_matches} chunk-based matches)")
    
    if MAX_FILES is not None:
        matched_pairs = matched_pairs[:MAX_FILES]
        print(f"Processing first {len(matched_pairs)} pairs (limited by MAX_FILES)")
    
    return matched_pairs


def load_pickle_data(filepath):
    """Load data from pickle file"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['RAW_DATA'][0]  # Shape: (channels, time_samples)


def compute_psd(signal, sample_rate, nperseg=PSD_NPERSEG, noverlap=PSD_OVERLAP):
    """
    Compute Power Spectral Density using Welch's method
    signal: (channels, time_samples) or (time_samples,)
    Returns: (frequencies, psd) where psd is averaged across channels if multi-channel
    """
    if signal.ndim == 2:
        # Multi-channel: compute PSD for each channel and average
        psds = []
        for ch in range(signal.shape[0]):
            freqs, psd = sci_sig.welch(signal[ch], fs=sample_rate, nperseg=nperseg, 
                                      noverlap=noverlap, detrend='constant')
            psds.append(psd)
        psd = np.mean(psds, axis=0)
    else:
        # Single channel
        freqs, psd = sci_sig.welch(signal, fs=sample_rate, nperseg=nperseg, 
                                  noverlap=noverlap, detrend='constant')
    
    return freqs, psd


def get_band_power(psd, freqs, band_low, band_high):
    """Compute total power in a frequency band"""
    band_mask = (freqs >= band_low) & (freqs <= band_high)
    return np.sum(psd[band_mask])


def calculate_apr(psd_before, psd_after, freqs):
    """Calculate Artifact Power Reduction for each artifact band"""
    apr_results = {}
    
    for band_name, (low, high) in ARTIFACT_BANDS.items():
        P_before = get_band_power(psd_before, freqs, low, high)
        P_after = get_band_power(psd_after, freqs, low, high)
        
        if P_before > 0:
            apr = (P_before - P_after) / P_before
        else:
            apr = 0.0
        
        apr_results[band_name] = apr
    
    return apr_results


def calculate_psnr(psd_before, psd_after, freqs):
    """Calculate Pseudo-SNR and improvement"""
    # EEG power: 1-30 Hz
    P_EEG_before = get_band_power(psd_before, freqs, 1, 30)
    P_EEG_after = get_band_power(psd_after, freqs, 1, 30)
    
    # Noise power: 40-100 Hz
    P_noise_before = get_band_power(psd_before, freqs, 40, 100)
    P_noise_after = get_band_power(psd_after, freqs, 40, 100)
    
    # Calculate pSNR
    if P_noise_before > 0:
        psnr_before = 10 * np.log10(P_EEG_before / P_noise_before)
    else:
        psnr_before = np.inf
    
    if P_noise_after > 0:
        psnr_after = 10 * np.log10(P_EEG_after / P_noise_after)
    else:
        psnr_after = np.inf
    
    delta_psnr = psnr_after - psnr_before
    
    return {
        'psnr_before': psnr_before,
        'psnr_after': psnr_after,
        'delta_psnr': delta_psnr
    }


def calculate_psd_cosine_similarity(psd_before, psd_after, freqs, exclude_artifacts=True):
    """Calculate cosine similarity between PSDs"""
    if exclude_artifacts:
        # Exclude artifact frequencies: keep 4-20 Hz and 30-40 Hz
        mask = ((freqs >= 4) & (freqs <= 20)) | ((freqs >= 30) & (freqs <= 40))
        psd_before_filtered = psd_before[mask]
        psd_after_filtered = psd_after[mask]
    else:
        psd_before_filtered = psd_before
        psd_after_filtered = psd_after
    
    # Normalize
    norm_before = np.linalg.norm(psd_before_filtered)
    norm_after = np.linalg.norm(psd_after_filtered)
    
    if norm_before > 0 and norm_after > 0:
        cosine_sim = np.dot(psd_before_filtered, psd_after_filtered) / (norm_before * norm_after)
    else:
        cosine_sim = 0.0
    
    return cosine_sim


def calculate_kl_divergence(psd_before, psd_after, freqs):
    """Calculate KL divergence between normalized PSDs"""
    # Normalize PSDs to probabilities
    P = psd_before / (np.sum(psd_before) + 1e-10)
    Q = psd_after / (np.sum(psd_after) + 1e-10)
    
    # Avoid log(0)
    P = np.clip(P, 1e-10, None)
    Q = np.clip(Q, 1e-10, None)
    
    # KL divergence: sum(P * log(P / Q))
    kl_div = np.sum(P * np.log(P / Q))
    
    return kl_div


def calculate_bpe(psd_before, psd_after, freqs):
    """Calculate Bandpower Preservation Error for each EEG band"""
    bpe_results = {}
    
    for band_name, (low, high) in EEG_BANDS.items():
        BP_before = get_band_power(psd_before, freqs, low, high)
        BP_after = get_band_power(psd_after, freqs, low, high)
        
        if BP_before > 0:
            bpe = np.abs(BP_after - BP_before) / BP_before
        else:
            bpe = 0.0
        
        bpe_results[band_name] = bpe
    
    return bpe_results


def calculate_kurtosis_change(signal_before, signal_after):
    """Calculate change in kurtosis"""
    if signal_before.ndim == 2:
        # Multi-channel: compute for each channel and average
        k_before = np.mean([kurtosis(signal_before[ch]) for ch in range(signal_before.shape[0])])
        k_after = np.mean([kurtosis(signal_after[ch]) for ch in range(signal_after.shape[0])])
    else:
        k_before = kurtosis(signal_before)
        k_after = kurtosis(signal_after)
    
    delta_k = k_before - k_after
    
    return {
        'kurtosis_before': k_before,
        'kurtosis_after': k_after,
        'delta_kurtosis': delta_k
    }


def calculate_skewness_change(signal_before, signal_after):
    """Calculate change in skewness"""
    if signal_before.ndim == 2:
        # Multi-channel: compute for each channel and average
        s_before = np.mean([skew(signal_before[ch]) for ch in range(signal_before.shape[0])])
        s_after = np.mean([skew(signal_after[ch]) for ch in range(signal_after.shape[0])])
    else:
        s_before = skew(signal_before)
        s_after = skew(signal_after)
    
    delta_s = s_before - s_after
    
    return {
        'skewness_before': s_before,
        'skewness_after': s_after,
        'delta_skewness': delta_s
    }


def evaluate_original_only(original_file):
    """Evaluate original data only (baseline statistics)"""
    try:
        # Load data
        signal = load_pickle_data(original_file)
        
        # Compute PSD
        freqs, psd = compute_psd(signal, SAMPLE_RATE)
        
        # Calculate baseline metrics
        results = {
            'filename': os.path.basename(original_file)
        }
        
        # Artifact band powers
        for band_name, (low, high) in ARTIFACT_BANDS.items():
            power = get_band_power(psd, freqs, low, high)
            results[f'artifact_power_{band_name}'] = power
        
        # EEG and noise powers
        P_EEG = get_band_power(psd, freqs, 1, 30)
        P_noise = get_band_power(psd, freqs, 40, 100)
        results['eeg_power_1_30hz'] = P_EEG
        results['noise_power_40_100hz'] = P_noise
        
        if P_noise > 0:
            results['psnr'] = 10 * np.log10(P_EEG / P_noise)
        else:
            results['psnr'] = np.inf
        
        # Bandpower for each EEG band
        for band_name, (low, high) in EEG_BANDS.items():
            bp = get_band_power(psd, freqs, low, high)
            results[f'bandpower_{band_name}'] = bp
        
        # Kurtosis
        if signal.ndim == 2:
            k = np.mean([kurtosis(signal[ch]) for ch in range(signal.shape[0])])
        else:
            k = kurtosis(signal)
        results['kurtosis'] = k
        
        # Skewness
        if signal.ndim == 2:
            s = np.mean([skew(signal[ch]) for ch in range(signal.shape[0])])
        else:
            s = skew(signal)
        results['skewness'] = s
        
        # Total power
        results['total_power'] = np.sum(psd)
        
        return results
        
    except Exception as e:
        print(f"Error processing {os.path.basename(original_file)}: {str(e)}")
        return None


def evaluate_file_pair(original_file, denoised_file):
    """Evaluate a single file pair"""
    try:
        # Load data
        signal_before = load_pickle_data(original_file)
        signal_after = load_pickle_data(denoised_file)
        
        # Ensure same shape
        if signal_before.shape != signal_after.shape:
            print(f"Warning: Shape mismatch for {os.path.basename(original_file)}")
            return None
        
        # Compute PSD
        freqs, psd_before = compute_psd(signal_before, SAMPLE_RATE)
        freqs, psd_after = compute_psd(signal_after, SAMPLE_RATE)
        
        # Calculate all metrics
        results = {
            'filename': os.path.basename(original_file)
        }
        
        # 1. APR
        apr = calculate_apr(psd_before, psd_after, freqs)
        results.update({f'apr_{k}': v for k, v in apr.items()})
        
        # 2. pSNR
        psnr = calculate_psnr(psd_before, psd_after, freqs)
        results.update(psnr)
        
        # 3. PSD Cosine Similarity
        results['psd_cosine_similarity'] = calculate_psd_cosine_similarity(psd_before, psd_after, freqs)
        
        # 4. KL Divergence
        results['kl_divergence'] = calculate_kl_divergence(psd_before, psd_after, freqs)
        
        # 5. BPE
        bpe = calculate_bpe(psd_before, psd_after, freqs)
        results.update({f'bpe_{k}': v for k, v in bpe.items()})
        
        # 6. Kurtosis Change
        kurt_results = calculate_kurtosis_change(signal_before, signal_after)
        results.update(kurt_results)
        
        # 7. Skewness Change
        skew_results = calculate_skewness_change(signal_before, signal_after)
        results.update(skew_results)
        
        return results
        
    except Exception as e:
        print(f"Error processing {os.path.basename(original_file)}: {str(e)}")
        return None


def aggregate_results(all_results):
    """Aggregate results across all files"""
    if not all_results:
        return None
    
    df = pd.DataFrame(all_results)
    
    # Calculate statistics
    summary = {
        'metric': [],
        'mean': [],
        'std': [],
        'min': [],
        'max': [],
        'median': []
    }
    
    # Exclude filename column
    metric_columns = [col for col in df.columns if col != 'filename']
    
    for col in metric_columns:
        summary['metric'].append(col)
        summary['mean'].append(df[col].mean())
        summary['std'].append(df[col].std())
        summary['min'].append(df[col].min())
        summary['max'].append(df[col].max())
        summary['median'].append(df[col].median())
    
    summary_df = pd.DataFrame(summary)
    
    return df, summary_df


def print_summary(summary_df):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("DENOISING EVALUATION SUMMARY")
    print("="*80)
    print(f"\n{'Metric':<30} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Median':<12}")
    print("-"*80)
    
    for _, row in summary_df.iterrows():
        print(f"{row['metric']:<30} {row['mean']:>12.4f} {row['std']:>12.4f} "
              f"{row['min']:>12.4f} {row['max']:>12.4f} {row['median']:>12.4f}")
    
    print("="*80)


def main():
    """Main execution function"""
    print("="*80)
    print("DENOISING EVALUATION SCRIPT")
    print("="*80)
    print(f"Data Type: {DATA_TYPE}")
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Save Directory: {SAVE_DIRECTORY}")
    print("="*80)
    
    # Set up directories
    original_dir = os.path.join(SAVE_DIRECTORY, f"dataset-tuh_task-binary_datatype-{DATA_TYPE}_v6")
    denoised_dir = os.path.join(SAVE_DIRECTORY, f"dataset-tuh_task-binary_datatype-{DATA_TYPE}_v6_denoised")
    
    # Check if directories exist
    if not os.path.exists(original_dir):
        print(f"Error: Original directory not found: {original_dir}")
        return
    
    denoised_exists = os.path.exists(denoised_dir)
    
    if not denoised_exists:
        print(f"Warning: Denoised directory not found: {denoised_dir}")
        print("Processing original data only (baseline statistics)...")
        
        # Process original files only
        original_files = glob.glob(os.path.join(original_dir, "*.pkl"))
        original_files = [f for f in original_files if 'preprocess_info' not in f]
        
        if MAX_FILES is not None:
            original_files = original_files[:MAX_FILES]
        
        print(f"Found {len(original_files)} original files")
        
        # Evaluate original files only
        print("\nEvaluating original files...")
        all_results = []
        
        for original_file in tqdm(original_files, desc="Processing files"):
            results = evaluate_original_only(original_file)
            if results is not None:
                all_results.append(results)
        
        print(f"\nSuccessfully evaluated {len(all_results)} original files")
    else:
        # Load matched file pairs
        matched_pairs = load_data_files(original_dir, denoised_dir)
        
        if len(matched_pairs) == 0:
            print("No matched file pairs found!")
            return
        
        # Evaluate each pair
        print("\nEvaluating file pairs...")
        all_results = []
        
        for original_file, denoised_file in tqdm(matched_pairs, desc="Processing files"):
            results = evaluate_file_pair(original_file, denoised_file)
            if results is not None:
                all_results.append(results)
        
        print(f"\nSuccessfully evaluated {len(all_results)} file pairs")
    
    if len(all_results) == 0:
        print("No valid results to aggregate!")
        return
    
    # Aggregate results
    print("\nAggregating results...")
    results_df, summary_df = aggregate_results(all_results)
    
    # Print summary
    print_summary(summary_df)
    
    # Save results
    if not denoised_exists:
        # Use different filename for baseline-only results
        output_file = OUTPUT_FILE.replace('.csv', '_baseline_only.csv')
    else:
        output_file = OUTPUT_FILE
    
    output_path = os.path.join(SAVE_DIRECTORY, output_file)
    results_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")
    
    summary_path = os.path.join(SAVE_DIRECTORY, output_file.replace('.csv', '_summary.csv'))
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to: {summary_path}")
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()

