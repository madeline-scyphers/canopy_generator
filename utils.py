import numpy as np
from scipy.fft import fft2, ifft2
from scipy.stats import norm


def calculate_autocorrelation_function(nx, ny, Dx, Dy, L):
    x = np.arange(-(nx - 1) / 2, (nx - 1) / 2 + 1) * Dx
    y = np.arange(-(ny - 1) / 2, (ny - 1) / 2 + 1) * Dy
    [X, Y] = np.meshgrid(x, y)

    AcF = np.exp(-(1 / L) * (X ** 2 + Y ** 2) ** 0.5)
    return AcF


def Find_StemDensityGridPT(StemDensityTPHa, Dx, nxy, Dy):
    StemDensityGridPT = 0
    i = 1
    while i < nxy:
        sizeHa = Dx * Dy * StemDensityTPHa / 10000
        if sizeHa * (i * i + (i + 1) * (i + 1)) / 2 >= 1:
            StemDensityGridPT = i
            i = nxy
        i = i + 1

    if StemDensityGridPT == 0:
        StemDensityGridPT = nxy

    return StemDensityGridPT


def get_stem_diam_and_breast_height(patch, HDBHpar, Height, nx, Dx, ny, Dy, StandDenc, numpatch):
    DBH = np.zeros((ny, nx))
    for i in range(numpatch):

        cjxmax = 1
        cjymax = 1
        StemDensityGridPT = Find_StemDensityGridPT(StandDenc[i], Dx, min(ny, nx), Dy)
        for ix in range(nx // StemDensityGridPT + 1):
            for iy in range(ny // StemDensityGridPT + 1):
                maxH = 0
                emptyind = 1
                for jx in range(StemDensityGridPT):
                    cjx = (ix - 1) * StemDensityGridPT + jx
                    if cjx <= len(Height[0, :]):
                        for jy in range(StemDensityGridPT):
                            cjy = (iy - 1) * StemDensityGridPT + jy
                            if cjy <= len(Height[:, 0]) and patch[cjy, cjx] == i:
                                emptyind = 0
                                if Height[cjy, cjx] > maxH:
                                    maxH = Height[cjy, cjx]
                                    cjxmax = cjx
                                    cjymax = cjy
                if emptyind == 0:
                    # scale DBH with Height, based on Naidu et al 98 can j for res - this function should be replaced
                    # if the user has another observed allometric indication of stem diameter
                    DBH[cjymax, cjxmax] = (
                        (StandDenc[i] / 10000 * StemDensityGridPT ** 2)
                        * HDBHpar[i, 2]
                        * (10 ** ((np.log10(Height[cjymax, cjxmax] * 100) - HDBHpar[i, 1]) / HDBHpar[i, 0]) / 100)
                    )

    return DBH


def Generate_PatchMap(patchtype, lambda_r, ny, nx, PatchCutOff, numpatch):

    patch = np.zeros(lambda_r.shape, dtype=np.int8) - 1
    LAIvec = lambda_r.reshape((nx * ny, 1))
    values, bins = np.histogram(LAIvec, bins=100)
    N, Lx = values, bins  # TODO remove

    cumN = np.zeros(len(N) + 1)

    for j in range(1, len(N) + 1):
        cumN[j] = cumN[j - 1] + N[j - 1]

    norm_cdf = norm.cdf(values)

    cumN = cumN / cumN[len(N)]
    # figure(500); plot(Lx,(N-min(N))/(max(N)-min(N)),'-k',[0, Lx],cumN,'--k'),axis('tight')

    px = np.zeros(numpatch + 1)
    cumulat = 0
    h = 0
    dj = 1
    while dj < numpatch:
        cumulat = cumulat + N[h]
        if cumulat > (PatchCutOff[dj - 1] * nx * ny):
            px[dj] = Lx[h]
            dj = dj + 1
        h = h + 1

    px[numpatch] = max(LAIvec)

    for i in range(numpatch):
        patchi = (lambda_r <= px[i + 1]) & (lambda_r > px[i])
        patch[patchi] = i

    return patch


def Make_VCaGe_rand_field(nx, ny, AcF):
    # Fourier trasnform the autocorrelation function
    ZF = fft2(AcF)
    # convert to real
    ZF2 = abs(ZF)

    # generate random phase matrix from uniform [0,1] to uniform [-pi,pi]
    R = 2 * np.pi * (np.random.rand(ny, nx) - 0.5)
    # compute complex amplitudes from real with random phase
    ZF3 = ZF2 * np.exp(1j * R)
    # inverse transform
    Z2 = ifft2(ZF3)
    # make real
    lambda_ = abs(Z2)
    return lambda_
