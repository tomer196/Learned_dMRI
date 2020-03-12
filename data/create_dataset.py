# pre-process tha dataset:
#   use only the first shell (b=1000)
#   filter out subjects with less then 90 diffusion directions on that shell
#   resample to predefine set of directions
#   save as h5 files per slice
import h5py
import scipy.io as sio
from util.sphere_interp import *
from dipy.io import read_bvals_bvecs
import nibabel as nib

# load the set of diffusion direction
mat = sio.loadmat('dir90.mat')
phi = mat['phi']
theta = mat['theta']
dirs90=HemiSphere(theta=theta.squeeze(), phi=phi.squeeze()).vertices

raw_dir = '/mnt/walkure_pub/Datasets/tomer/h5_1000_new2/raw/'
out_dir = '/home/tomerweiss/Datasets/dMRI/h5_1000_new3/'

dirlist = os.listdir(raw_dir)
i = 0
for dir_name in dirlist:
    if dir!='zip':
        # check if the subject have 90 dir on the b=1000 shell
        with open(raw_dir + dir_name + '/T1w/Diffusion/bvals') as f:
            lineList = f.readlines()
        bvals = np.array(lineList[0].split(), dtype='int16')
        ind = (bvals > 900) * (bvals < 1100)
        temp = bvals[ind]
        if temp.shape[0] == 90:
            dwi_file = nib.load(raw_dir + dir_name + '/T1w/Diffusion/data.nii.gz')
            dwi = dwi_file.get_fdata().astype('float32')
            mask = nib.load(raw_dir + dir_name + + '/T1w/Diffusion/nodif_brain_mask.nii.gz').get_data()
            bvals, bvecs = read_bvals_bvecs(raw_dir + dir_name + '/T1w/Diffusion/bvals',
                                            raw_dir + dir_name + '/T1w/Diffusion/bvecs')
            # choose only b=0,1000 shells
            ind = (bvals > 900) * (bvals < 1100) + (bvals<100)
            bvecs = bvecs[ind,:]
            dwi = dwi[:, :, :, ind]
            bvals = bvals[ind]

            # remove empty slices
            slice = np.sum(dwi, axis=(0, 1, 3))
            dwi = dwi[:, :, slice != 0, :]
            t1 = t1[:, :, slice != 0]
            mask = mask[:, :, slice != 0]
            t1 = t1 * mask

            # resample DWI to the 90 pre defined directions
            resampled_dwi = resample_dwi(mask_dwi(dwi, mask), bvals, bvecs, directions=dirs90, sh_order=12, smooth=0.006)
            resampled_dwi = mask_dwi(resampled_dwi, mask)
            resampled_dwi = resampled_dwi.astype('float32')

            # save each slice
            for j in range(resampled_dwi.shape[2]):
                h5f = h5py.File(out_dir + dir + '_s' + str(j) + '.h5', 'w')
                h5f.create_dataset('data', data=resampled_dwi[:,:,j,:])
                h5f.close()
            i += 1
            print(f'{i}, name:{dir}, dir={resampled_dwi.shape}')
