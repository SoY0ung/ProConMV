import os
import numpy as np
import torch
import cv2


class Multiview_Concept_Mask_ImgDataset(torch.utils.data.Dataset):

    def __init__(self, imgs_dir, masks_dir, data_list, scale_aug=False, rot_aug=False, num_views=4,
                 transform=None, no_mask=False):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.data_list = data_list
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.num_views = num_views
        self.transform = transform
        self.no_mask = no_mask

    def __len__(self):
        return len(self.data_list) // self.num_views

    def __getitem__(self, idx):
        base_idx = idx * self.num_views
        class_id = self.data_list.iloc[base_idx, -1].astype(np.int64)

        imgs = []
        lesion_labels = []
        for view in range(self.num_views):
            row = self.data_list.iloc[base_idx + view]
            img_name = f"{row['ID']}.jpg"
            mask_name = f"{row['ID']}_mask.png"
            if row['RATE'] != class_id:
                print(f'[ERROR]: Except RATE {class_id}, but got RATE {row["RATE"]} at {row["ID"]}')
                exit(1)
            img_path = os.path.join(self.imgs_dir, img_name)
            mask_path = os.path.join(self.masks_dir, mask_name)


            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Image error: {img_path} is not found!")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Mask error: {mask_path} is not found!")
            mask = mask.reshape((mask.shape[0], mask.shape[1], 1))  # HWC -> HWC1

            if self.no_mask:
                mask[:] = 255


            image_with_mask = np.concatenate([image, mask], axis=2)  # HWC+1

            if self.transform:
                image_with_mask = self.transform(image_with_mask)

            imgs.append(image_with_mask)

            lesion = row[['EX', 'HE', 'MA', 'SE', 'MHE', 'BRD']].astype(np.float32).values
            lesion_labels.append(torch.tensor(lesion))

        return torch.stack(imgs), torch.stack(lesion_labels), class_id
