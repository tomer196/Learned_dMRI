import numpy as np
import os, glob
from collections import OrderedDict
# from scipy.ndimage import map_coordinates
# import nibabel as nib
# from nibabel.streamlines import ArraySequence
# from nibabel import streamlines
# from dipy.data import get_sphere
# from dipy.core.sphere import Sphere, HemiSphere
# from dipy.core.geometry import sphere_distance
# from dipy.reconst.shm import sph_harm_lookup, smooth_pinv
from scipy.special import gammaln
import torch

def map_coordinates_3d_4d(input_array, indices, affine=None, order=1):
    """ Evaluate the input_array data at the given indices
    using trilinear interpolation
    Parameters
    ----------
    input_array : ndarray,
        3D or 4D array
    indices : ndarray
    Returns
    -------
    output : ndarray
        1D or 2D array
    Notes
    -----
    At some point this will be merged in Dipy. See PR #587.
    """
    if affine is not None:
        inv_affine = np.linalg.inv(affine)
        indices = (np.dot(indices, inv_affine[:3, :3]) + inv_affine[:3, 3])

    if input_array.ndim <= 2 or input_array.ndim >= 5:
        raise ValueError("Input array can only be 3d or 4d")

    if input_array.ndim == 3:
        return map_coordinates(input_array, indices.T, order=order)

    if input_array.ndim == 4:
        values_4d = []
        for i in range(input_array.shape[-1]):
            values_tmp = map_coordinates(input_array[..., i],
                                         indices.T, order=order)
            values_4d.append(values_tmp)
        return np.ascontiguousarray(np.array(values_4d).T)


def eval_volume_at_3d_coordinates(volume, coords):
    """ Evaluates the volume data at the given coordinates using trilinear interpolation.
    Parameters
    ----------
    volume : 3D array or 4D array
        Data volume.
    coords : ndarray of shape (N, 3)
        3D coordinates where to evaluate the volume data.
    Returns
    -------
    output : 2D array
        Values from volume.
    """

    if volume.ndim <= 2 or volume.ndim >= 5:
        raise ValueError("Volume must be 3D or 4D!")

    if volume.ndim == 3:
        return map_coordinates(volume, coords.T, order=1, mode="nearest")

    if volume.ndim == 4:
        values_4d = []
        for i in range(volume.shape[-1]):
            values_tmp = map_coordinates(volume[..., i],
                                         coords.T, order=1, mode="nearest")
            values_4d.append(values_tmp)
        return np.ascontiguousarray(np.array(values_4d).T)


def denoise_dwi(weights, gtab):
    """ denoise dwi bataset using local PCA
    -----------
    weights : ndarray of shape (X, Y, Z, #gradients)
    Returns
    -------
    ndarray
        Diffusion weights volume denoised using local PCA.
    """
    sigma = pca_noise_estimate(weights, gtab, correct_bias=True, smooth=3)
    denoised_arr = localpca(weights, sigma=sigma, patch_radius=2)

    return denoised_arr


def normalize_dwi(weights, b0):
    """ Normalize dwi by the first b0.
    Parameters:
    -----------
    weights : ndarray of shape (X, Y, Z, #gradients)
        Diffusion weighted images.
    b0 : ndarray of shape (X, Y, Z)
        B0 image.
    Returns
    -------
    ndarray
        Diffusion weights normalized by the B0.
    """
    b0 = b0[..., None]  # Easier to work if it is a 4D array.

    # Make sure in every voxels weights are lower than ones from the b0.
    # Should not happen, but with the noise we never know!
    # nb_erroneous_voxels = np.sum(weights > b0)
    # if nb_erroneous_voxels != 0:
    #     print("Nb. erroneous voxels: {}".format(nb_erroneous_voxels))
    #     weights = np.minimum(weights, b0)

    # Normalize dwi using the b0.
    weights_normed = weights / b0
    weights_normed[weights_normed=="Inf"] = 0.

    return weights_normed


def get_spherical_harmonics_coefficients(dwi, bvals, bvecs,normalized,device, sh_order=8, smooth=0.006, first=False):
    """ Compute coefficients of the spherical harmonics basis.
    Parameters
    -----------
    dwi : `nibabel.NiftiImage` object
        Diffusion signal as weighted images (4D).
    bvals : ndarray shape (N,)
        B-values used with each direction.
    bvecs : ndarray shape (N, 3)
        Directions of the diffusion signal. Directions are
        assumed to be only on the hemisphere.
    sh_order : int, optional
        SH order. Default: 8
    smooth : float, optional
        Lambda-regularization in the SH fit. Default: 0.006.
    Returns
    -------
    sh_coeffs : ndarray of shape (X, Y, Z, #coeffs)
        Spherical harmonics coefficients at every voxel. The actual number of
        coeffs depends on `sh_order`.
    """
    # bvals = np.asarray(bvals)
    # bvecs = np.asarray(bvecs)
    #     dwi_weights = dwi.get_data().astype("float32")
    dwi_weights = dwi

    # Exract the averaged b0.
    if not normalized:
        b0_idx = torch.lt(bvals,100)
        b0 = dwi_weights[..., b0_idx].mean(dim=3) + 0.0000000000000001

        # Extract diffusion weights and normalize by the b0.
        not_b0_idx = torch.gt(bvals,100)
        bvecs = bvecs[not_b0_idx]
        weights = dwi_weights[..., not_b0_idx]
        weights = normalize_dwi(weights, b0)
    else:
        weights=dwi

    # Assuming all directions are on the hemisphere.
    theta,phi = HemiSphere_torch(xyz=bvecs)

    # Fit SH to signal
    # sph_harm_basis = sph_harm_lookup.get('tournier07')
    Ba, m, n = sph_harm_basis_torch(sh_order, theta, phi,device)
    L = -n * (n + 1)
    invB = smooth_pinv(Ba, np.sqrt(smooth) * L.float())
    data_sh = torch.matmul(weights, invB.t())
    return data_sh

def smooth_pinv(B, L):
    """Regularized pseudo-inverse

    Computes a regularized least square inverse of B

    Parameters
    ----------
    B : array_like (n, m)
        Matrix to be inverted
    L : array_like (n,)

    Returns
    -------
    inv : ndarray (m, n)
        regularized least square inverse of B

    Notes
    -----
    In the literature this inverse is often written $(B^{T}B+L^{2})^{-1}B^{T}$.
    However here this inverse is implemented using the pseudo-inverse because
    it is more numerically stable than the direct implementation of the matrix
    product.

    """
    L = torch.diag(L)
    inv = torch.pinverse(torch.cat((B, L)))
    return inv[:, :len(B)]

def sph_harm_basis_torch(sh_order, theta, phi,device):
    """
    Compute real spherical harmonics as in Tournier 2007 [2]_, where the real
    harmonic $Y^m_n$ is defined to be::

        Real($Y^m_n$)       if m > 0
        $Y^0_n$             if m = 0
        Imag($Y^|m|_n$)     if m < 0

    This may take scalar or array arguments. The inputs will be broadcasted
    against each other.

    Parameters
    -----------
    sh_order : int
        The maximum degree or the spherical harmonic basis.
    theta : float [0, pi]
        The polar (colatitudinal) coordinate.
    phi : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.

    Returns
    --------
    y_mn : real float
        The real harmonic $Y^m_n$ sampled at `theta` and `phi` as
        implemented in mrtrix. Warning: the basis is Tournier et al.
        2007 [2]_; 2004 [1]_ is slightly different.
    m : array
        The order of the harmonics.
    n : array
        The degree of the harmonics.

    References
    ----------
    .. [1] Tournier J.D., Calamante F., Gadian D.G. and Connelly A.
           Direct estimation of the fibre orientation density function from
           diffusion-weighted MRI data using spherical deconvolution.
           NeuroImage. 2004;23:1176-1185.
    .. [2] Tournier J.D., Calamante F. and Connelly A. Robust determination
           of the fibre orientation distribution in diffusion MRI:
           Non-negativity constrained super-resolved spherical deconvolution.
           NeuroImage. 2007;35(4):1459-1472.

    """
    m, n = sph_harm_ind_list_torch(sh_order,device)
    phi = torch.reshape(phi, (-1, 1))
    theta = torch.reshape(theta, (-1, 1))

    m = -m
    real_sh = real_sph_harm_torch(m, n, theta, phi)
    # real_sh /= np.where(m == 0, 1., np.sqrt(2))
    return real_sh, m, n

def real_sph_harm_torch(m, n, theta, phi):
    r""" Compute real spherical harmonics.

    Where the real harmonic $Y^m_n$ is defined to be:

        Imag($Y^m_n$) * sqrt(2)     if m > 0
        $Y^0_n$                     if m = 0
        Real($Y^|m|_n$) * sqrt(2)   if m < 0

    This may take scalar or array arguments. The inputs will be broadcasted
    against each other.

    Parameters
    ----------
    m : int ``|m| <= n``
        The order of the harmonic.
    n : int ``>= 0``
        The degree of the harmonic.
    theta : float [0, 2*pi]
        The azimuthal (longitudinal) coordinate.
    phi : float [0, pi]
        The polar (colatitudinal) coordinate.

    Returns
    --------
    y_mn : real float
        The real harmonic $Y^m_n$ sampled at `theta` and `phi`.

    See Also
    --------
    scipy.special.sph_harm
    """
    # dipy uses a convention for theta and phi that is reversed with respect to
    # function signature of scipy.special.sph_harm
    sh = spherical_harmonics_torch(torch.abs(m), n, phi, theta)

    real_sh = torch.where(m > 0, sh[:,:,1], sh[:,:,0])
    # real_sh = real_sh * torch.where(m == 0, torch.tensor(1.,device=phi.device), torch.tensor(np.sqrt(2),device=phi.device))
    return real_sh

def spherical_harmonics_torch(m, n, theta, phi):
    x = torch.cos(phi)
    val = legendre_associated(m, n, x)
    val = val * torch.sqrt((2 * n.float() + 1) / 4.0 / np.pi)
    val = val * torch.tensor(np.exp(0.5 * (gammaln(n.cpu().numpy() - m.cpu().numpy() + 1) - gammaln(n.cpu().numpy() + m.cpu().numpy() + 1))),device=phi.device).float()
    val = val.unsqueeze(-1) * torch.stack([torch.cos(m.float() * theta),torch.sin(m.float() * theta)],dim=-1)
    return val

def legendre_associated(m, n, x):
    x=x.squeeze(1)
    ans=torch.zeros(x.shape[0],m.shape[0],device=x.device)
    somx2 = torch.sqrt(1.0 - x * x+1e-8)
    for j in range(m.shape[0]):
        # cx = torch.zeros(x.shape[0],n[j] + 1,device=x.device)
        #
        # cx[:,m[j]] = 1.0
        cx_list=[torch.ones(x.shape[0],device=x.device)]
        fact = 1.0
        for i in range(0, m[j]):
            # cx[:,m[j]] = - cx[:,m[j]] * fact * somx2
            cx_list[0] = -cx_list[0] * fact * somx2
            fact = fact + 2.0

        # cx_list = [cx[:,m[j]]]
        if (m[j] != n[j]):
            cx_list.append(x * float(2 * m[j] + 1) * cx_list[0])
            # cx[:,m[j] + 1] = x * float(2 * m[j] + 1) * cx[:,m[j]]

            for i in range(m[j] + 2, n[j] + 1):
                cx_list.append((float(2 * i - 1) * x * cx_list[i - 1-m[j]] + float(- i - m[j] + 1) * cx_list[i - 2-m[j]]) / float(i - m[j]))
                # cx[:,i] = (float(2 * i - 1) * x * cx[:,i - 1] + float(- i - m[j] + 1) * cx[:,i - 2]) / float(i - m[j])
        ans[:,j]=cx_list[-1]
    return ans

def sph_harm_ind_list_torch(sh_order,device):
    """
    Returns the degree (n) and order (m) of all the symmetric spherical
    harmonics of degree less then or equal to `sh_order`. The results, `m_list`
    and `n_list` are kx1 arrays, where k depends on sh_order. They can be
    passed to :func:`real_sph_harm`.

    Parameters
    ----------
    sh_order : int
        even int > 0, max degree to return

    Returns
    -------
    m_list : array
        orders of even spherical harmonics
    n_list : array
        degrees of even spherical harmonics

    See also
    --------
    real_sph_harm
    """
    if sh_order % 2 != 0:
        raise ValueError('sh_order must be an even integer >= 0')

    n_range = torch.arange(0, sh_order + 1, 2, device=device)
    n_list = n_range.repeat_interleave(n_range * 2 + 1)

    ncoef = int((sh_order + 2) * (sh_order + 1) // 2)
    offset = 0
    m_list = torch.empty(ncoef, dtype=n_list.dtype,device=device)
    for ii in n_range:
        m_list[offset:offset + 2 * ii + 1] = torch.arange(-ii, ii + 1)
        offset = offset + 2 * ii + 1

    # makes the arrays ncoef by 1, allows for easy broadcasting later in code
    return (m_list, n_list)

def HemiSphere_torch(xyz):
    xyz = xyz * ( 1 - 2 * torch.lt(xyz[:, -1:],0).float()) # to remove if we can assume xyz on HemSphere
    theta, phi = cart2sphere(xyz)
    return theta, phi

def cart2sphere(xyz):
    r""" Return angles for Cartesian 3D coordinates `x`, `y`, and `z`

    See doc for ``sphere2cart`` for angle conventions and derivation
    of the formulae.

    $0\le\theta\mathrm{(theta)}\le\pi$ and $-\pi\le\phi\mathrm{(phi)}\le\pi$

    Parameters
    ------------
    x : array_like
       x coordinate in Cartesian space
    y : array_like
       y coordinate in Cartesian space
    z : array_like
       z coordinate

    Returns
    ---------
    r : array
       radius
    theta : array
       inclination (polar) angle
    phi : array
       azimuth angle
    """
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    r = torch.sqrt(x * x + y * y + z * z)
    theta = torch.acos(z/r)
    # theta = torch.where(r > 0, theta, 0.)
    phi = torch.atan2(y, x)
    return theta, phi

def dwi2sh (dwi, theta, phi, sh_order=8, smooth=0.006):
    Ba, m, n = sph_harm_basis_torch(sh_order, theta, phi, device=dwi.device)
    L = -n * (n + 1)
    invB = smooth_pinv(Ba, np.sqrt(smooth) * L.float())
    data_sh = torch.matmul(dwi, invB.t())
    return data_sh

def sh2dwi (dwi_sh, theta, phi, sh_order=8, smooth=0.006):
    Ba, m, n = sph_harm_basis_torch(sh_order, theta, phi, dwi_sh.device)
    dwi = torch.matmul(dwi_sh, Ba.t())
    return dwi

def resample_dwi(dwi, bvals, bvecs,device, directions=None, sh_order=8, smooth=0.006,normalized=False):
    """ Resamples a diffusion signal according to a set of directions using spherical harmonics.
    Parameters
    -----------
    dwi : `nibabel.NiftiImage` object
        Diffusion signal as weighted images (4D).
    bvals : ndarray shape (N,)
        B-values used with each direction.
    bvecs : ndarray shape (N, 3)
        Directions of the diffusion signal. Directions are
        assumed to be only on the hemisphere.
    directions : `dipy.core.sphere.Sphere` object, optional
        Directions the diffusion signal will be resampled to. Directions are
        assumed to be on the whole sphere, not the hemisphere like bvecs.
        If omitted, 100 directions evenly distributed on the sphere will be used.
    sh_order : int, optional
        SH order. Default: 8
    smooth : float, optional
        Lambda-regularization in the SH fit. Default: 0.006.
    Returns
    -------
    ndarray
        Diffusion weights resampled according to `sphere`.
    """
    data_sh = get_spherical_harmonics_coefficients(dwi, bvals, bvecs,normalized,device, sh_order=sh_order, smooth=smooth)

    theta, phi = HemiSphere_torch(xyz=directions)

    Ba, m, n = sph_harm_basis_torch(sh_order, theta, phi, device)

    data_resampled = torch.matmul(data_sh, Ba.t())

    return data_resampled


def mask_dwi(dwi_vol, mask_vol):
    if mask_vol.ndim == 3:
        masked_dwi = dwi_vol * np.tile(mask_vol[..., None], (1, 1, 1, dwi_vol.shape[-1]))
    else:
        masked_dwi = dwi_vol * mask_vol

    return masked_dwi


def get_file_path(curr_path, target_dir, extension):
    os.chdir(target_dir)
    for file in glob.glob(extension):
        file_path = os.path.join(target_dir, file)
    os.chdir(curr_path)

    return file_path


def align_streamlines_to_grid(tract, dwi):
    dwi_vec = np.diag(dwi.affine)[0:3]
    tract_vec = np.diag(tract.affine)[0:3]
    ratio_vec = tract_vec / dwi_vec

    if not all(ratio_vec == 1.0):
        ratio_mat = np.diag(ratio_vec)
        all_streamlines = []
        for i in range(len(tract.streamlines)):
            all_streamlines.append(np.matmul(tract.streamlines[i], ratio_mat))

        return all_streamlines

    else:
        return tract.streamlines


def get_streamlines_lengths(streamlines_list):
    """
INPUT: fibers_list - python list of size N, where each element is a fiber represented by a Ln x 3 np array
OUTPUT: lengths - a np array (vector) of length N, holding the #points in each fiber
"""
    lengths = np.zeros(len(streamlines_list))

    for i in range(len(streamlines_list)):
        lengths[i] = streamlines_list[i].shape[0]

    return lengths