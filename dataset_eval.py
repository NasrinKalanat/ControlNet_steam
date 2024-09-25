import os
import glob
from PIL import Image
from torchvision import transforms
import json
class DatasetEval(object):
    def __init__(self, imgs_gt_path, imgs_gen_path, image_size=256):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.imgs_gt_path = imgs_gt_path
        self.imgs_gen_path = imgs_gen_path
        self.imgs_gt = {}
        self.imgs_gen = {}
        self.labels = {}
        self.ids = []
        for file in glob.glob(f"{imgs_gt_path}/*.*"):
            file = os.path.basename(file)
            if file.startswith("control"):
                _, idx = file.split(".png")[0].split("_gs-")
                self.imgs_gt[idx]=file
            if file.startswith("label"):
                _, idx = file.split(".json")[0].split("_gs-")
                with open(f"{imgs_gen_path}/{file}", "r") as f:
                    self.labels[idx]=json.load(f)
                for i in range(len(self.labels[idx])):
                    self.ids.append((idx,len(self.labels[idx]), i))
            if file.startswith("samples"):
                _, idx = file.split(".png")[0].split("_gs-")
                self.imgs_gen[idx]=file

    def split_img(self, img_gt_path, l, i):
        img = Image.open(img_gt_path)
        w, h = img.size
        w_per_img = w //4
        return img.crop((i*w_per_img,0,(i+1)*w_per_img,h))
    def __getitem__(self, id):
        idx, l, i = self.ids[id]
        img_gt_path = os.path.join(self.imgs_gt_path, self.imgs_gt[idx])
        img_gen_path = os.path.join(self.imgs_gen_path, self.imgs_gen[idx])
        lbl = self.labels[idx]
        print("*********")
        for i in range(4):
            img_gt = self.split_img(img_gt_path, l, i)
            img_gt.save(f"/data/nak168/spatial_temporal/stream_img/ControlNet_steam/ckpt_test/image_log/test_{idx}_{i}.png")
        print("&&&&&&&&&&&")
        img_gt = self.transform(img_gt)
        img_gen = self.split_img(img_gen_path, l, i)
        img_gen = self.transform(img_gen)
        return img_gt, img_gen, lbl

    def __len__(self):
        return len(self.ids)
