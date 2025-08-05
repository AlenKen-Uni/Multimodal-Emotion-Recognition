import numpy as np
import subprocess
import librosa
import io

from spafe.features.mfcc import mfcc
from spafe.utils import preprocessing

def truncate_waveform(
    waveform   : np.ndarray, 
    sr         : int=16000, 
    win_len    : float=0.04, 
    overlap_len: float=0.02
    ) -> np.ndarray:
    """ Eliminate the last redundant frame to ensure valid framing """
    
    total_samples = len(waveform)
    frame_size, overlap = int(sr*win_len), int(sr*overlap_len)
    step_size = frame_size - overlap  # hop length

    # compute the maximum number of valid frames
    num_valid_frames = (total_samples - overlap) // step_size

    # compute the valid length
    valid_length = num_valid_frames * step_size + overlap

    # truncate waveform
    truncated_waveform = waveform[:valid_length]
    
    return truncated_waveform

def handcrafted_feature_extractor(waveform: np.ndarray) -> np.ndarray:
    """ Extracts Mel-Frequency Cepstral Coefficients (MFCC) from the waveform """
    # compute mfcc
    mfccs = mfcc(
        sig=waveform,
        fs=16000,
        num_ceps=40,
        pre_emph=True,
        pre_emph_coeff=0.97,
        window=preprocessing.SlidingWindow(0.04, 0.02, "hamming"),
        nfilts=80,
        low_freq=250,
        high_freq=8000,
        lifter=22,
        normalize="mvn", # z-normalization
    )
    return mfccs

def split_audio_from_video(video_path: str, sr: int=16000) -> np.ndarray:
    """
    Uses ffmpeg to extract the audio track directly into memory (no file saved),
    then loads it with librosa.
    """
    cmd = [
        'ffmpeg',
        '-i', video_path,   # input
        '-vn',              # drop video
        '-ar', str(sr),     # resample to sr
        '-ac', '1',         # mono
        '-f', 'wav',        # WAV PCM
        '-'                 # send to stdout
    ]
    # capture stdout (audio wav bytes), discard stderr
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True
    )
    wav_bytes = proc.stdout
    
    # load from in‑memory bytes buffer
    waveform, _ = librosa.load(io.BytesIO(wav_bytes), sr=sr)
    
    return waveform