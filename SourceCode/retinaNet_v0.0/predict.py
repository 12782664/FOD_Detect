import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import time
import json

import torch
from PIL import Image
import matplotlib.pyplot as plt


from torchvision import transforms
from network_files import RetinaNet
from backbone import resnet50_fpn_backbone, LastLevelP6P7
from draw_box_utils import draw_objs
import cv2

def create_model(num_classes):
    # resNet50+fpn+retinanet
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d,
                                     returned_layers=[2, 3, 4],
                                     extra_blocks=LastLevelP6P7(256, 256))
    model = RetinaNet(backbone, num_classes)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device._2".format(device))

    # create model
    # 注意：不包含背景
    model = create_model(num_classes=31)

    # load train weights
    weights_path = "./save_weights/230321-fod-19.pth"
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')["model"],strict=False)
    model.to(device)

    # read class_indict
    # label_json_path = './pascal_voc_classes.json'
    label_json_path = './pascal_voc_classes_FOD.json'
    #label_json_path = './pascal_voc_classes_Fatigue.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    for i in os.listdir("E:\FOD--\FODPascalVOCFormat-V.2.1\VOCdevkit\VOC2007\JPEGImages"):
        if int(i.split(".")[0]) < 45:
            continue
        print(i)
        # load image
        #original_img = Image.open("E:\\FOD--\\FODPascalVOCFormat-V.2.1\\VOCdevkit\\VOC2007\\JPEGImages\\013117.jpg")
        # original_img = Image.open("E:\\Dataset\\Fatigue\\VOCdevkit\\VOC2007\\JPEGImages\\176.jpg")
        original_img = Image.open("E:\FOD--\FODPascalVOCFormat-V.2.1\VOCdevkit\VOC2007\JPEGImages\\{}".format(i))
        #original_img = Image.open("E:\FOD--\FODPascalVOCFormat-V.2.1\VOCdevkit\VOC2007\Test0913\\{}".format(i))
        #  # 'C:\\Users\\xwf\\Pictures\\Camera Roll\\2.jpg'
        #"E:\Dataset\\64_CASIA-FaceV5\\009\\009_1.bmp"

        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            t_end = time_synchronized()
            print("22  inference+NMS time: {}".format(t_end - t_start))

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")

            plot_img = draw_objs(original_img,
                                predict_boxes,
                                predict_classes,
                                predict_scores,
                                category_index=category_index,
                                box_thresh=0.4,
                                line_thickness=1,
                                font='arial.ttf',
                                font_size=20)
            plt.imshow(plot_img)
            plt.show()
        
        # 保存预测的图片结果
        #plot_img.save("test_result.jpg")
        plot_img.save("E:\FOD--\FODPascalVOCFormat-V.2.1\VOCdevkit\VOC2007\Test0922\\{}".format(i))

def test_cam(cam=0):
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device._1".format(device))

    model = create_model(num_classes=31)

    weights_path = "./save_weights/230321-fod-0.pth"
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')["model"],strict=False)
    model.to(device)

    label_json_path = './pascal_voc_classes_Fatigue.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    cap=cv2.VideoCapture(cam)

    while cap.isOpened():
        ret,img=cap.read()#读取图片
        if ret != True:
            print("cv read failed")
            break
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            t_end = time_synchronized()
            print("11 inference+NMS time: {}".format(t_end - t_start))

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")

            plot_img = draw_objs(img,
                                predict_boxes,
                                predict_classes,
                                predict_scores,
                                category_index=category_index,
                                box_thresh=0.4,
                                line_thickness=1,
                                font='arial.ttf',
                                font_size=20)
            cv2.imshow("test",plot_img)
            plt.show()
            # 保存预测的图片结果
            plot_img.save("test_result.jpg")

if __name__ == '__main__':
    main()
    # test_cam()

