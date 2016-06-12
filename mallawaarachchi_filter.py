from pylab import *
from scipy.ndimage.filters import correlate         # equivalent to matlab's imfilter

def mallawaarachchi_filter(I,alpha=1,beta=1,M=3):
    """
    Input:
        I       input image (spectrogram)
        alpha   high alpha preserves more of original detail
        beta    high beta increases amount of attenuation of vertical patterns
        M       filter size MxM, default 3 (probably should be larger than contour thickness)

    Use on spectrogram to remove the influence on clicks
    Paper: Mallawaarachchi et a. (2008) Spectrogram denoising and automated extraction of the
            fundamental frequency variation in dolphin whistles
    Link: http://arl.nus.edu.sg/twiki6/pub/ARL/BibEntries/Mallawaarachchi2008a.pdf

    Earlier Sturtivant and Datta method was only vertical edge suppression (for clicks). This includes
    horizontal smoothing also.

    Code converted from Arik Kershenbaum's IPRiT
    Converted Feb 6, 2015 by Jeremy Karnowski
    """
    (p,q) = np.mgrid[1:M+1,1:M+1]
    p = p - float(M)/2
    q = q - float(M)/2

    a = float(M)/10

    v1 = exp(-0.5*((p/a/6)**2+(q/a)**2))
    v2 = exp(-0.5*((p/a)**2+(q/a/6)**2))
    v3 = exp(-(((q-p)/a)**2+((q+p)/a/6)**2))
    v4 = exp(-(((q-p)/a/6)**2+((q+p)/a)**2))

    ih = correlate(I,v1)
    iv = correlate(I,v2)
    id1 = correlate(I,v3)
    id2 = correlate(I,v4)

    R = np.max(dstack([ih,id1,id2]),2)
    V = iv

    return (alpha*I + beta*(R-V))/(alpha+beta)
