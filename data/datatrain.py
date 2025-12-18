import os
import torch
import random
from PIL import Image
from torch.utils import data
from torchvision import transforms

class TrainData(data.Dataset):
    def __init__(self, exocentric_root, egocentric_root, resize_size=256, crop_size=224, divide="Seen"):

        self.exocentric_root = exocentric_root
        self.egocentric_root = egocentric_root

        
        self.depth_exocentric_root = self.exocentric_root.replace("AGD20K", "AGD20K-Depth")  
        self.depth_egocentric_root = self.egocentric_root.replace("AGD20K", "AGD20K-Depth")

        self.image_list = []
        #self.exo_image_list = []

        self.depth_image_list = []
        #self.depth_exo_image_list = []

        self.resize_size = resize_size
        self.crop_size = crop_size
        if divide == "Seen":
            self.aff_list = ['beat', "boxing", "brush_with", "carry", "catch", "cut", "cut_with", "drag", 'drink_with',
                             "eat", "hit", "hold", "jump", "kick", "lie_on", "lift", "look_out", "open", "pack", "peel",
                             "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick", "stir", "swing", "take_photo",
                             "talk_on", "text_on", "throw", "type_on", "wash", "write"]
            self.obj_list = ['apple', 'axe', 'badminton_racket', 'banana', 'baseball', 'baseball_bat',
                             'basketball', 'bed', 'bench', 'bicycle', 'binoculars', 'book', 'bottle',
                             'bowl', 'broccoli', 'camera', 'carrot', 'cell_phone', 'chair', 'couch',
                             'cup', 'discus', 'drum', 'fork', 'frisbee', 'golf_clubs', 'hammer', 'hot_dog',
                             'javelin', 'keyboard', 'knife', 'laptop', 'microwave', 'motorcycle', 'orange',
                             'oven', 'pen', 'punching_bag', 'refrigerator', 'rugby_ball', 'scissors',
                             'skateboard', 'skis', 'snowboard', 'soccer_ball', 'suitcase', 'surfboard',
                             'tennis_racket', 'toothbrush', 'wine_glass']
        else:
            self.aff_list = ["carry", "catch", "cut", "cut_with", 'drink_with',
                             "eat", "hit", "hold", "jump", "kick", "lie_on", "open", "peel",
                             "pick_up", "pour", "push", "ride", "sip", "sit_on", "stick",
                             "swing", "take_photo", "throw", "type_on", "wash"]
            self.obj_list = ['apple', 'axe', 'badminton_racket', 'banana', 'baseball', 'baseball_bat',
                             'basketball', 'bed', 'bench', 'bicycle', 'binoculars', 'book', 'bottle',
                             'bowl', 'broccoli', 'camera', 'carrot', 'cell_phone', 'chair', 'couch',
                             'cup', 'discus', 'drum', 'fork', 'frisbee', 'golf_clubs', 'hammer', 'hot_dog',
                             'javelin', 'keyboard', 'knife', 'laptop', 'microwave', 'motorcycle', 'orange',
                             'oven', 'pen', 'punching_bag', 'refrigerator', 'rugby_ball', 'scissors',
                             'skateboard', 'skis', 'snowboard', 'soccer_ball', 'suitcase', 'surfboard',
                             'tennis_racket', 'toothbrush', 'wine_glass']

        self.transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))])
        
        self.depth_transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))  
        ])

        self.all_transform = PairedTransforms(resize_size, crop_size)
        
        # image list for egocentric images
        files = os.listdir(self.exocentric_root)
        for file in files:
            file_path = os.path.join(self.exocentric_root, file)
            obj_files = os.listdir(file_path)
            for obj_file in obj_files:
                obj_file_path = os.path.join(file_path, obj_file)
                images = os.listdir(obj_file_path)

                depth_obj_file_path = obj_file_path.replace("AGD20K", "AGD20K-Depth")
                depth_images = os.listdir(depth_obj_file_path)

                for img in images:
                    img_path = os.path.join(obj_file_path, img)
                    self.image_list.append(img_path)

                for img in depth_images:
                    depth_img_path = os.path.join(depth_obj_file_path, img)
                    self.depth_image_list.append(depth_img_path)

        # multiple affordance labels for exo-centric samples

    def __getitem__(self, item):

        # load egocentric image
        exocentric_image_path = self.image_list[item]
        names = exocentric_image_path.split("/")
        aff_name, object = names[-3], names[-2]
        ######exocentric_image = self.load_img(exocentric_image_path)
        aff_label = self.aff_list.index(aff_name)
        ego_label_name = self.aff_list

        filename, ext = exocentric_image_path.rsplit('.', 1)
        depth_exocentric_image_path = filename + '_depth.' + 'png'
        depth_exocentric_image_path = depth_exocentric_image_path.replace("AGD20K", "AGD20K-Depth")
        #######depth_exocentric_image = self.load_depth_img(depth_exocentric_image_path)
        exocentric_image, depth_exocentric_image = self.load_all_img(exocentric_image_path, depth_exocentric_image_path)
        # #加载exo深度
        # depth_exocentric_image_path = self.depth_image_list[item]
        # depth_exocentric_image = self.load_depth_img(depth_exocentric_image_path)

        ego_path = os.path.join(self.egocentric_root, aff_name, object)
        obj_images = os.listdir(ego_path)
        idx = random.randint(0, len(obj_images) - 1)
        egocentric_image_path = os.path.join(ego_path, obj_images[idx])
        #######egocentric_image = self.load_img(egocentric_image_path)
        

        filename, ext = egocentric_image_path.rsplit('.', 1)
        depth_egocentric_image_path = filename + '_depth.' + 'png'
        depth_egocentric_image_path = depth_egocentric_image_path.replace("AGD20K", "AGD20K-Depth")
        #######depth_egocentric_image = self.load_depth_img(depth_egocentric_image_path)
        egocentric_image, depth_egocentric_image = self.load_all_img(egocentric_image_path, depth_egocentric_image_path)

        # pick one available affordance, and then choose & load exo-centric images
        num_exo = 3
        exo_dir = os.path.dirname(exocentric_image_path)
        exocentrics = os.listdir(exo_dir)
        exo_img_name = [os.path.basename(exocentric_image_path)]
        exocentric_images = [exocentric_image]

        depth_exo_dir = os.path.dirname(exocentric_image_path.replace("AGD20K", "AGD20K-Depth"))
        depth_exocentric_images = [depth_exocentric_image]
        # exocentric_labels = []

        if len(exocentrics) > num_exo:
            for i in range(num_exo - 1):
                exo_img_ = random.choice(exocentrics)
                while exo_img_ in exo_img_name:
                    exo_img_ = random.choice(exocentrics)
                exo_img_name.append(exo_img_)
                ######tmp_exo = self.load_img(os.path.join(exo_dir, exo_img_))
                

                filename, ext = exo_img_.rsplit('.', 1)
                depth_img_ = filename + '_depth.' + 'png'
                #######depth_tmp_exo = self.load_depth_img(os.path.join(depth_exo_dir, depth_img_)) 
                tmp_exo, depth_tmp_exo = self.load_all_img(os.path.join(exo_dir, exo_img_), os.path.join(depth_exo_dir, depth_img_))
                exocentric_images.append(tmp_exo)
                depth_exocentric_images.append(depth_tmp_exo)

        else:
            for i in range(num_exo - 1):
                exo_img_ = random.choice(exocentrics)
                # while exo_img_ in exo_img_name:
                #     exo_img_ = random.choice(exocentrics)
                exo_img_name.append(exo_img_)
                ######tmp_exo = self.load_img(os.path.join(exo_dir, exo_img_))
                

                filename, ext = exo_img_.rsplit('.', 1)
                depth_img_ = filename + '_depth.' + 'png'
                ######depth_tmp_exo = self.load_depth_img(os.path.join(depth_exo_dir, depth_img_))
                tmp_exo, depth_tmp_exo = self.load_all_img(os.path.join(exo_dir, exo_img_), os.path.join(depth_exo_dir, depth_img_))
                exocentric_images.append(tmp_exo)
                depth_exocentric_images.append(depth_tmp_exo)

        exocentric_images = torch.stack(exocentric_images, dim=0)  # n x 3 x 224 x 224
        depth_exocentric_images = torch.stack(depth_exocentric_images, dim=0)  # n x 3 x 224 x 224

        return exocentric_images, egocentric_image, depth_exocentric_images, depth_egocentric_image, aff_label, ego_label_name

    def load_img(self, path):
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img
    
    def load_depth_img(self, path):
        img = Image.open(path).convert('L') 
        img = self.depth_transform(img)
        return img

    def load_all_img(self, rgb_path, dedpth_path):
        rgb_img = Image.open(rgb_path).convert('RGB')
        depth_img = Image.open(dedpth_path).convert('L')
        rgb, depth = self.all_transform(rgb_img, depth_img)
        return rgb, depth
    
    def __len__(self):

        return len(self.image_list)

class PairedTransforms:
    def __init__(self, resize_size, crop_size):
        self.resize = transforms.Resize(resize_size)
        self.random_crop = transforms.RandomCrop(crop_size)
        self.random_flip = transforms.RandomHorizontalFlip()
        
        self.to_tensor_rgb = transforms.ToTensor()
        self.to_tensor_depth = transforms.ToTensor()
        
        self.normalize_rgb = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                  std=(0.229, 0.224, 0.225))
        self.normalize_depth = transforms.Normalize(mean=(0.5,), std=(0.5,))
    
    def __call__(self, rgb_img, depth_img):
        # Resize both images
        rgb_img = self.resize(rgb_img)
        depth_img = self.resize(depth_img)
        
        # Random Crop (using the same parameters)
        i, j, h, w = transforms.RandomCrop.get_params(rgb_img, output_size=(224, 224))
        rgb_img = transforms.functional.crop(rgb_img, i, j, h, w)
        depth_img = transforms.functional.crop(depth_img, i, j, h, w)

        # Random Horizontal Flip (using the same parameters)
        if random.random() > 0.5:
            rgb_img = transforms.functional.hflip(rgb_img)
            depth_img = transforms.functional.hflip(depth_img)

        # Convert to tensor
        rgb_img = self.to_tensor_rgb(rgb_img)
        depth_img = self.to_tensor_depth(depth_img)
        
        # Normalize
        rgb_img = self.normalize_rgb(rgb_img)
        depth_img = self.normalize_depth(depth_img)

        return rgb_img, depth_img
