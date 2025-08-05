import numpy as np
import cv2

def pearson_correlation(A: np.ndarray, B: np.ndarray) -> float:
    # flatten
    a = A.ravel().astype(np.float64)
    b = B.ravel().astype(np.float64)
    # subtract means
    a -= a.mean()
    b -= b.mean()
    # compute numerator & denominator
    num = (a * b).sum()
    den = np.sqrt((a*a).sum() * (b*b).sum())
    return num/den

def rv_coefficient(A: np.ndarray, B: np.ndarray) -> float:
    if len(A.shape) != 2:
        A = cv2.cvtColor(A, cv2.COLOR_RGB2GRAY)
    if len(B.shape) != 2:
        B = cv2.cvtColor(B, cv2.COLOR_RGB2GRAY)
    a = A.astype(np.float64)
    b = B.astype(np.float64)
    num = np.trace((a@a.T) @ (b@b.T))
    den = np.sqrt(np.trace((a@a.T) @ (a@a.T)) * np.trace((b@b.T) @ (b@b.T)))
    return num / den

