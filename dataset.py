import os, glob
import cv2
import pickle
import numpy as np
from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# (original) prepare_data를 통해 lmdb key값을 어느 정도 맞춰놓음. -> 해당 부분 수정
class MultiResolutionDataset(Dataset):
    """
    dataset for training alpha networks. It handles lmdb data formats.
    """
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        
        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
            
            if not os.path.isfile(f"{path}/key_list.pickle"):
                self.key_list = list(txn.cursor().iternext(values=False))
                
                print("Successfully Generated a Key list!")

                with open(f"{path}/key_list.pickle", "wb") as f:
                    pickle.dump(self.key_list, f)

            else:
                with open(f"{path}/key_list.pickle", "rb") as f:
                    self.key_list = pickle.load(f)

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = self.key_list[index]
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        
        if self.transform:
            img = self.transform(img)

        return img

class TestDataset(Dataset):
    """
    Dataset for a real image projection.
    """
    def __init__(self, root, transform):
        super().__init__()

        self.file_list = sorted(glob.glob(root+"/*.jpg")+glob.glob(root+"/*.jfif")+glob.glob(root+"/*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index]).convert("RGB")

        if self.transform:
            img = self.transform(img)
        
        return img

class CifarVehicleDataset(Dataset):
    """
    Cifar-10 Dataset (only vehicle images)
    """
    def __init__(self, path="/home/data/Labels4Free/cifar-10-batches-py", is_train=True, n_downsample=-1, transforms=None):
        
        if is_train:
            self.path = sorted(glob.glob(path+"/data_*"))

        else:
            self.path = os.path.join(path, "/test_batch")

        self.n_downsample = n_downsample
        self.transforms = transforms
        file_list = [self.unpickle(p) for p in self.path]
        
        self.imgs = []
        self.labels = []
        self.file_list = []

        for f in file_list:
            imgs, labels, files = self.filter_labels(f)
            self.imgs.append(imgs) # 이렇게 들어가면 안되는 게 인덱싱이 안됨.
            self.labels.append(labels)
            self.file_list.extend(files)
        
        self.imgs = np.vstack(self.imgs).reshape(-1,3,32,32) # make it a 4d-tensor
        self.imgs = self.imgs.transpose(0,2,3,1) # convert to (H,W,C)

        self.labels = np.vstack(self.labels).squeeze()
        
        if self.n_downsample != -1:
            indices = np.random.randint(0, len(self.imgs), size=n_downsample)
            self.imgs = self.imgs[indices]
            self.labels = self.labels[indices]
            self.file_list = [fn for i, fn in enumerate(self.file_list) if i in indices]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img, label = self.imgs[idx], self.labels[idx]

        # convert a numpy array to a PIL Image (to consistent with torch transforms)
        img = Image.fromarray(img)

        if self.transforms:
            img = self.transforms(img)

        return img, label

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict
    
    def filter_labels(self, batch):
        """
        batch 딕셔너리를 받아서 automobile, truck 라벨 값을 갖는 데이터만 필터링

        - Args
            batch: a dictionary that contains data and metadata (batch_labels, labels, data, filename)
        """
        indices = [i for i,f in enumerate(batch['labels']) if f in [1,9]]
        labels = np.array(batch['labels'])[indices][:, None]
        imgs = batch['data'][indices]
        files = [fn for i, fn in enumerate(batch['filenames']) if i in indices]
        
        return imgs, labels, files

    def get_img_with_filename(self, filename):
        img = self.imgs[self.file_list.index(filename)]
        print(self.file_list.index(filename))
        
        if self.transforms:
            img = self.transforms(img)
        
        return img


class PadTransform(object):
    def __init__(self, resize):
        self.resize = resize

    def __call__(self, img): # img: PIL Image
        if isinstance(img, np.ndarray):
            cv2.resize(img, (256,256), cv2.INTER_LANCZOS4)
            img = Image.fromarray(img)
        
        w,h = img.width, img.height

        if h > w:
            resize = (self.resize, round(self.resize*w/h))
            padding = (round(self.resize*(1-w/h)/2), 0) # rounding error 때문에 발생하는 padding 값 에러 확인할 것.
        
        else:
            resize = (round(self.resize*h/w), self.resize)
            padding = (0,round(self.resize*(1-h/w)/2))
        

        transform = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.Pad(padding),
                transforms.Resize((self.resize, self.resize)), # to ensure that transformed image has a given shape.
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        return transform(img)

if __name__ == "__main__":
    dataset = CifarVehicleDataset(n_downsample=100)
    print(len(dataset))