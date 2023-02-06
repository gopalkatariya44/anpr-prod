# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img1.jpg                         # image
                                                     vid.mp4                         # video
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import easyocr
import numpy as np
import torch

# import pytesseract
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import time
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
import re
import joblib

from datetime import datetime

currentDateAndTime = datetime.now()
number_plate_color = joblib.load('number_plate.pkl')


# def locate_license_plate_candidates(gray, keep=5):
# 		# perform a blackhat morphological operation that will allow
# 		# us to reveal dark regions (i.e., text) on light backgrounds
# 		# (i.e., the license plate itself)
# 		rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
# 		blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
# 		self.debug_imshow("Blackhat", blackhat)
def pre_process_imgs(img):
    # img1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32))
    return img


@smart_inference_mode()
def noise_remove(image):
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image


def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image


def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format

                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        x1 = xyxy[0].detach().cpu().numpy()
                        y1 = xyxy[1].detach().cpu().numpy()
                        x2 = xyxy[2].detach().cpu().numpy()
                        y2 = xyxy[3].detach().cpu().numpy()
                        crop_img = im0[int(y1):int(y2), int(x1):int(x2)]
                        print(crop_img.shape)
                        # crop_img = cv2.resize(crop_img, (190,52),
                        #                           interpolation=cv2.INTER_LINEAR)
                        new_crop_img = pre_process_imgs(crop_img)
                        flattened_arr1 = np.insert(new_crop_img[:, :, -1].flatten(), 0, 0)
                        pred = number_plate_color.predict([flattened_arr1[1:]])[0]
                        reader = easyocr.Reader(['en'])
                        ocr_list = []
                        print("-----------pred------->", pred)
                        # try:
                        if pred == 0:
                            ocr = reader.readtext(crop_img, detail=0)
                            ocr_str = "".join(ocr)
                            ocr_str = ocr_str.upper()
                            ocr_str = re.sub('[^a-zA-Z0-9]+', '', ocr_str)
                            ocr_list = list(ocr_str)
                        elif pred == 1:
                            kernel = np.ones((3, 3), np.uint8)
                            img_dilation = cv2.dilate(crop_img, kernel, iterations=1)
                            ocr = reader.readtext(img_dilation, detail=0)
                            ocr_str = "".join(ocr)
                            ocr_str = ocr_str.upper()
                            ocr_str = re.sub('[^a-zA-Z0-9]+', '', ocr_str)
                            ocr_list = list(ocr_str)
                        print(ocr_list)
                        # img_name = str(int(time.time())) + ".png"
                        # reader = easyocr.Reader(['en'])
                        # ocr = reader.readtext(crop_img, detail=0)
                        # ocr_str = "".join(ocr)
                        # print("before con",ocr_str)
                        # ocr_str = ocr_str.upper()
                        # ocr_str = re.sub('[^a-zA-Z0-9]+', '', ocr_str)
                        # print("after OCR string",ocr_str)
                        # ocr_list = list(ocr_str)
                        if len(ocr_list) == 10:
                            if ocr_list[0].isdigit() or ocr_list[1].isdigit() or ocr_list[4].isdigit() or ocr_list[
                                5].isdigit():
                                if ocr_list[0] == '6' or ocr_list[0] == '0' or ocr_list[0] == '1':
                                    ocr_list[0] = 'G'
                                if ocr_list[1] == 'U' or ocr_list[1] == '3':
                                    ocr_list[1] = 'J'
                                if ocr_list[4] == '0' or ocr_list[4] == '9':
                                    ocr_list[4] = 'B'
                                if ocr_list[4] == '1':
                                    ocr_list[4] = 'D'
                                if ocr_list[5] == '7' or ocr_list[5] == '1':
                                    ocr_list[5] = 'T'
                                if ocr_list[5] == '4':
                                    ocr_list[5] = 'A'
                            if ocr_list[0] == 'N':
                                ocr_list[0] = 'M'
                        # if (not ocr_list[2].isdigit()
                        #     or not ocr_list[3].isdigit()
                        #     or not ocr_list[6].isdigit()
                        #     or not ocr_list[7].isdigit()
                        #     or not ocr_list[8].isdigit()
                        #     or not ocr_list[9].isdigit()):
                        #     if ocr_list[2] == "o" or "O":
                        #         ocr_list[2] = 0
                        #     if ocr_list[3] == "o" or "O":
                        #         ocr_list[3] = 0
                        #     if ocr_list[6] == "o" or "O":
                        #         ocr_list[6] = 0
                        #     if ocr_list[7] == "o" or "O":
                        #         ocr_list[7] = 0
                        #     if ocr_list[8] == "o" or "O":
                        #         ocr_list[8] = 0
                        #     if ocr_list[9] == "o" or "O":
                        #         ocr_list[9] = 0
                        #     if ocr_list[2] == "I":
                        #         ocr_list[2] = "1"
                        #     if ocr_list[3] == "B":
                        #         ocr_list[3] = "8"

                        # if not ocr_list[0]
                        print(ocr_list)
                        # except:
                        #     print("except")
                        # root_dir = "/home/mihir/computer_vision/yolo/number_plate_img/"
                        # # stretch_near = cv2.resize(crop_img, (128, 128),
                        # #                 interpolation = cv2.INTER_LINEAR)
                        # # locate_license_plate_candidates(crop_img)
                        # a1 = time.time()
                        # img_name = str(int(a1)) + ".png"
                        # print(root_dir+img_name)
                        # cv2.imwrite(root_dir+img_name, crop_img)
                        # inverted_image = cv2.bitwise_not(crop_img)
                        # median = np.median(crop_img)
                        # lower = int(max(0, (2 - 0.1) * median))
                        # upper = int(min(255, (2 + 0.1) * median))
                        # edge_image= cv2.Canny(crop_img, lower, upper)
                        # no_noise = noise_remove(edge_image)
                        # thin_image = thin_font(no_noise)
                        # pret, thresh_image_with_grey = cv2.threshold(no_noise, 100, 150, cv2.THRESH_BINARY)
                        # cv2.imwrite(f"{root_dir}canny.png",no_noise)
                        # crop_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
                        # reader = easyocr.Reader(['en'])
                        # ocr = reader.readtext(no_noise, detail = 0)
                        # print("OCR",ocr)
                        # ocr = "".join(ocr)
                        # ocr_list = list(ocr)
                        ocr_list = [str(i) for i in ocr_list]
                        ocr_label = "".join(ocr_list)
                        # if len(ocr_list) != 10:
                        #     ocr_list.pop(-1)
                        # if ocr_list[0] == '6' or ocr_list[0] == 'g' or ocr_list[2] == "o":
                        #     ocr_list[0] = "G"
                        #     ocr_list[2] = "0"
                        # print("OCR",ocr_list)
                        # predicted_res = pytesseract.image_to_string(im0, lang ='eng')
                        # print("answer ------------------- ",predicted_res)
                        # predicted_NP=[]
                        # filter_predicted_res = "".join(predicted_res.split()).replace(":", "").replace("-", "")
                        # predicted_NP.append(filter_predicted_res)
                        # print(predicted_NP)
                        # ocr_label = "GJ18AH6000"
                        annotator.box_label(xyxy, ocr_label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                currentTime = currentDateAndTime.strftime("%H:%M:%S")
                print(currentTime)
                if dataset.mode == 'image':
                    os.makedirs('/home/dev1034/Documents/yolov5/output/plate_output', exist_ok=True)
                    cv2.imwrite(save_path, im0)
                    cv2.imwrite(f'/home/dev1034/Documents/yolov5/output/plate_output/{currentTime}.png', im0)


                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'number_plate.pt',
                        help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img1', '--img1-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.65, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img1', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
