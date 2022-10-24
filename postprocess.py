# DBSCAN 이용해서 postprocess 수행
from sklearn.cluster import DBSCAN
import os, glob
import cv2
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm


class PostProcessor(object):
    """
    Segmentation 후처리를 통해, 최대한 자동차만 남기도록 함.

    동작 범위: 하나의 디렉토리가 주어지면 그 디렉토리 내부의 mask, img 리스트를 가져와서 후처리 진행

    후처리는 DBSCAN 알고리즘을 사용하여 이미지 내 mask가 여러 영역으로 분리 가능한지? 분리 가능하다면 outlier segmentation mask로 간주하여, 삭제.
    """
    def __init__(self, root, eps, min_samples, masking_val):
        self.root = root
        self.mask_dir = sorted(glob.glob(root+"/*_mask.png")+glob.glob(root+"/*_mask.jpg"))
        self.img_dir = sorted(glob.glob(root+"/*_origin.png")+glob.glob(root+"/*_origin.jpg"))

        assert len(self.mask_dir) == len(self.img_dir), "Number of masks and images must be same."
        assert len(self.mask_dir) > 0, f"Nothing detected in the given root '{root}'."

        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.masking_val = masking_val

    def get_cluster(self, mask):
        """
        하나의 mask에 대한 clustering 수행

        - Args
            mask(pd.DataFrame): masking이 되어 있지 않은 부분에 대한 좌표 y,x를 각 열로 하는 데이터 프레임 mask
        
        - Returns
            clusters(np.ndarray): 클러스터 라벨값으로 구성된 1d-array
        """
        clusters = self.dbscan.fit_predict(mask)

        return clusters        
    
    def _postprocess(self, idx):
        """
        single image 처리
        """
        mask = cv2.imread(self.mask_dir[idx])[:,:,0]
        img = cv2.imread(self.img_dir[idx])
        
        # convert a mask array into a dataframe which contains coords of non-masking area.
        mask_df = pd.DataFrame(zip(*np.where(mask==self.masking_val)))
        clusters = self.get_cluster(mask_df)
        n_clusters = len(set(clusters))

        df_new = pd.concat([mask_df, pd.Series(clusters)], axis=1)
        df_new.columns = ['y','x','c']

        # 발견된 non-masking 영역이 2개 이상이면, major region 제외하고 삭제
        if n_clusters > 1:
            coords_del = df_new[df_new['c'] != df_new['c'].value_counts().sort_values(ascending=False).index[0]].iloc[:, :2].values.tolist()
            
            for y,x in coords_del:
                mask[y][x] = 0

        # 이미지 재합성
        new_comp = cv2.bitwise_and(img, np.repeat(mask[:,:,np.newaxis], 3, -1))

        return new_comp

    def postprocess(self):
        """
        root 디렉토리 내 모든 이미지 처리
        """
        print(f">>>>> Processing for {self.root}...")

        for i in tqdm(range(len(self.mask_dir))):
            img = self._postprocess(i)
            
            
            new_path = self.mask_dir[i].replace("mask", "composite")
            cv2.imwrite(new_path, img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", required=True, type=str, help="A data root to postprocess.")
    parser.add_argument("--eps", type=float, default=20.0, help="A value of epsilon for a DBSCAN.")
    parser.add_argument("--min_samples", type=int, default=1200, help="Number of minimum samples for a DBSCAN.")
    parser.add_argument("--masking_val", type=int, default=255)

    opt = parser.parse_args()

    post_processor = PostProcessor(opt.root, opt.eps, opt.min_samples, opt.masking_val)
    post_processor.postprocess()

