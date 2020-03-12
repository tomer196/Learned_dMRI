import pathlib
import random
import h5py
from torch.utils.data import Dataset
class SliceData(Dataset):
    def __init__(self, root, transform, sample_rate=1):
        self.transform = transform

        files = list(pathlib.Path(root).iterdir())
        if sample_rate < 1:
            # random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        self.files=files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fname = self.files[i]
        with h5py.File(fname, 'r') as data:
            img = data['data'][()]
        return self.transform(img, fname.name, 0)
