import os
import torch
import logging
import numpy as np
from PIL import Image


class PennFudanDatatset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        # load all image files, sorting them to ensure that they are aligned
        self.imgPath = os.path.join(root, "PNGImages")
        self.maskPath = os.path.join(root, "PedMasks")
        self.imgs = list(sorted(os.listdir(self.imgPath)))
        self.masks = list(sorted(os.listdir(self.maskPath)))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.imgPath, self.imgs[idx])
        mask_path = os.path.join(self.maskPath, self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask = np.array(mask)
        obj_ids = np.unique(mask)
        try:
            # delete `0` from obj_idx (background)
            zidx = obj_ids.index(0)
            obj_idsj = np.delete(obj_ids, zidx)
        except:
            pass

        # split the color-encoded maks inot a set of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a toch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crows
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def test():
    from pyobjdetect.transform import base as transform
    from pyobjdetect.utils import logutils, viz, helpers

    logutils.setupLogging("DEBUG")
    root = os.path.join(os.environ["ODT_DATA_DIR"], "PennFudanPed")
    dataset = PennFudanDatatset(root, transforms=transform.get_example_transform(train=True))

    assert len(dataset) > 0, f"Length of dataset must be larger than 0"
    logging.info(f"loaded dataset with {len(dataset)} examples")
    idx = np.random.randint(0, len(dataset))
    img, target = dataset.__getitem__(idx)
    img = helpers.torch2numpy(img)

    matToShow = [helpers.torch2numpy(target["masks"][i]) for i in range(target["masks"].shape[0])]
    matToShow.append(np.array(img))
    viz.quickmatshow(matToShow, title=f"example {idx}")
    viz.show()


if __name__ == "__main__":
    test()
