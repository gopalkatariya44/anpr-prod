import datetime
from utils.datasets import *
from utils.general import *
from utils.torch_utils import *
# import pymongo
import urllib
from collections import defaultdict
import config as cfg
import joblib
import numpy as np
import easyocr
import re

# mongo_client = pymongo.MongoClient(host=cfg.MONGO_HOST,
#                                    port=cfg.MONGO_PORT,
#                                    username=cfg.MONGO_USER,
#                                    password=cfg.MONGO_PASS,
#                                    authSource=cfg.MONGO_AUTH_DB_NAME)
# db = mongo_client[cfg.MONGO_DB_NAME]
# collection = db[cfg.MONGO_COLL_NAME]
number_plate_color = joblib.load('number_plate.pkl')


def store_results(user_id, camera_id, image_path, image_url, result):
    def_dict = defaultdict(int)
    for data in result.get("detection"):
        def_dict[data['label']] += 1

    my_dict = {
        "user_id": str(user_id),
        "camera_id": str(camera_id),
        "image_name": image_path,
        "image_url": image_url,
        "result": result,
        "is_hide": False,
        "created_date": datetime.datetime.now(datetime.timezone.utc),
        "updated_date": datetime.datetime.now(datetime.timezone.utc),
        "status": True,
        "counts": dict(def_dict)
    }
    # cfg.logger.info("Data store in MMONGO ::",my_dict)
    x = collection.insert_one(my_dict)
    if x:
        # cfg.logger.info("result data inserted")
        print("result data inserted")
    else:
        # cfg.logger.info("result data not inserted")
        print("result data not inserted")


def get_device():
    return select_device("cpu")


def load_model(weight_path, map_location):
    # cfg.logger.info("Load Model")
    return (
        torch.load(weight_path, map_location=map_location)["model"]
        .float()
        .fuse()
        .eval()
    )


def load_image_from_url(image_path, img_size=640):
    if image_path:
        req = urllib.request.urlopen(image_path)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img0 = cv2.imdecode(arr, -1)
        img = letterbox(img0, new_shape=img_size, scaleup=False)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # to 3x416x416
        img = np.ascontiguousarray(img)
        return img, img0, image_path
    else:
        print("ERROR: image not found from URL")
        return False


def load_image_from_disk(image_path, img_size=640):
    if os.path.exists(image_path):
        img0 = cv2.imread(image_path)
        img = letterbox(img0, new_shape=img_size, scaleup=False)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # to 3x416x416
        img = np.ascontiguousarray(img)
        return img, img0, image_path
    else:
        print("ERROR: image not found in local")
        return False


pre_number_plate_str = ''


def predict(model, img, im0s, device, conf_thres, iou_thres):
    global pre_number_plate_str

    # cfg.logger.info("In Prediction")
    st = time.time()
    names = model.module.names if hasattr(model, "module") else model.names
    print("names", names)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # Inference
    t1 = time_synchronized()
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(
        pred, conf_thres, iou_thres, classes=None, agnostic=False
    )

    result_dict = {}
    result_list = []
    im_result = ''
    number_plate_str = ''
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            # cfg.logger.info("{} detections found.".format(len(det)))
            print("{} detections found.".format(len(det)))
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            for *xyxy, conf, cls in det:
                tmp_dict = {}
                # x1 = int(xyxy[0])
                # y1 = int(xyxy[1])
                # x2 = int(xyxy[2])
                # y2 = int(xyxy[3])
                x1 = xyxy[0].detach().cpu().numpy()
                y1 = xyxy[1].detach().cpu().numpy()
                x2 = xyxy[2].detach().cpu().numpy()
                y2 = xyxy[3].detach().cpu().numpy()
                tmp_dict["label"] = names[int(cls)]
                tmp_dict["location"] = [x1, y1, x2, y2]
                result_list.append(tmp_dict)
                number_plate_crop_img = im0s[int(y1):int(y2), int(x1):int(x2)]
                # print("**************************",number_plate_crop_img)
                color_bool = color_detection(crop_img=number_plate_crop_img)
                number_plate_str = ocr_char(crop_img=number_plate_crop_img, color_id=color_bool)

                # if pre number plate is same ---------------
                if number_plate_str[-1:-4:-1] == pre_number_plate_str[-1:-4:-1]:
                    continue
                else:
                    pre_number_plate_str = number_plate_str
                # -------------------------------------------

                tmp_dict["number_plate_recognization"] = number_plate_str
                print("last_out: ", tmp_dict["number_plate_recognization"])

                # draw_boxes and write image code ----------------------
                im_result = draw_boxes(img=im0s, bbox=[[int(x1), int(y1), int(x2), int(y2)]], names=number_plate_str)
                os.makedirs('run/img', exist_ok=True)
                cv2.imwrite(f"run/img/{tmp_dict['number_plate_recognization']}_{time.time()}.png", im_result)
                # ------------------------------------------------------

    result_dict["detection"] = result_list
    t2 = time_synchronized()
    result_dict["inference_time"] = t2 - t1
    print("sec: ", result_dict["inference_time"], end="  ")
    return result_dict, im_result


def pre_process_imgs(img):
    # img1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    # print(">>>>>>>>>>>>>>>>>",img1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (32, 32))
    return img


def color_detection(crop_img=None):
    process_crop_img = pre_process_imgs(crop_img)
    flattened_arr1 = np.insert(process_crop_img[:, :, -1].flatten(), 0, 0)
    pred = number_plate_color.predict([flattened_arr1[1:]])[0]
    return pred


def draw_boxes(img, bbox, identities=None, categories=None, names="", save_with_object_id=False, path=None,
               offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        data = (int((box[0] + box[2]) / 2), (int((box[1] + box[3]) / 2)))
        # label = str(id) + ":"+ names[cat]
        # label = names
        (w, h), _ = cv2.getTextSize(names, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 20), 2)
        # cv2.rectangle(img1, (x1, y1 - 20), (x1 + w, y1), (255,144,30), -1)
        cv2.putText(img, names, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, [0, 0, 255], 2)
        # cv2.circle(img1, data, 6, color,-1)   #centroid of box
        txt_str = ""
        if save_with_object_id:
            txt_str += "%i %i %f %f %f %f %f %f" % (
                id, cat, int(box[0]) / img.shape[1], int(box[1]) / img.shape[0], int(box[2]) / img.shape[1],
                int(box[3]) / img.shape[0], int(box[0] + (box[2] * 0.5)) / img.shape[1],
                int(box[1] + (
                        box[3] * 0.5)) / img.shape[0])
            txt_str += "\n"
            with open(path + '.txt', 'a') as f:
                f.write(txt_str)
    return img


def compare_numberplate(pre_plate, current_pate, con):
    if current_pate[-1:-4:-1] == pre_plate[-1:-4:-1]:
        con
    else:
        pre_plate = current_pate


def ocr_char(crop_img=None, color_id=None):
    # try:
    # gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    # roi = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # fun ----------------
    def str_opr(string):
        string = "".join(string)
        string = string.upper()
        string = re.sub('[^a-zA-Z0-9]+', '', string)
        string = string.replace("IND", "", 1)
        string = string.replace("INC", "", 1)
        string = string.replace("IN", "", 1)
        string = string.replace("ND", "", 1)
        return string

    # --------------------

    reader = easyocr.Reader(['en'])
    ocr_list = []

    # ocr_roi = reader.readtext(roi, detail=0)
    # ocr_roi_str = str_opr(ocr_roi)
    # print("------ocr_roi-------->", ocr_roi)
    # print("------ocr_roi_str-------->", ocr_roi_str)
    if color_id == 0:
        # print("color: White")
        # ocr = reader.readtext(crop_img, detail=0)
        kernel = np.ones((3, 3), np.uint8)
        img_dilation = cv2.dilate(crop_img, kernel, iterations=1)
        ocr = reader.readtext(img_dilation, detail=0)
        ocr_str = str_opr(ocr)

        print(f"White_out1: {ocr_str}")
        ocr_list = list(ocr_str)
        if len(ocr_list) != 10:
            ocr = reader.readtext(crop_img, detail=0)
            ocr_str = str_opr(ocr)
            print(f"White_out2: {ocr_str}")
            ocr_list = list(ocr_str)
    elif color_id == 1:
        # print("color: yeallow")
        kernel = np.ones((3, 3), np.uint8)
        img_dilation = cv2.dilate(crop_img, kernel, iterations=1)
        ocr = reader.readtext(img_dilation, detail=0)
        ocr_str = str_opr(ocr)
        print(f"yeallow_out: {ocr_str}")
        ocr_list = list(ocr_str)

    # fun ----------------
    def is_not_digit(i: int):
        return not ocr_list[i].isdigit()

    def is_digit(i: int):
        return ocr_list[i].isdigit()

    def check(check: str, change: str, i: int):
        if ocr_list[i] == check:
            ocr_list[i] = change
            print(check, change)

    # --------------------
    def master_change_digit(char_list: list):
        """
        function for change the errors where digit
        """
        for i in char_list:
            check('6', 'G', i)
            check('0', 'D', i)
            check('4', 'A', i)
            check('2', 'Z', i)
            check('1', 'T', i)
            check('7', 'T', i)
            check('2', 'Z', i)
            check('8', 'B', i)

            # extra
            check('3', 'K', i)
            # check('3', 'K', i)

    def master_change_not_digit(char_list: list):
        """
        function for change the errors where not digit
        """
        for i in char_list:
            check('I', '1', i)
            check('Z', '2', i)
            check('D', '0', i)
            check('G', '0', i)
            check('O', '0', i)
            check('T', '1', i)

    if len(ocr_list) == 10:
        if is_digit(0) or is_digit(1) or is_digit(4) or is_digit(5):
            char_list = [0, 1, 4, 5]
            # change the errors
            master_change_digit(char_list)
        if (is_not_digit(2) or is_not_digit(3) or is_not_digit(6) or is_not_digit(7) or is_not_digit(
                8) or is_not_digit(9)):
            digit_list = [2, 3, 6, 7, 8, 9]
            # change the errors
            master_change_not_digit(digit_list)
    elif len(ocr_list) == 9:
        if is_digit(0) or is_digit(1) or is_digit(4):
            char_list = [0, 1, 4]
            # change the errors
            master_change_digit(char_list)
        if (is_not_digit(2) or is_not_digit(3) or is_not_digit(5) or is_not_digit(6) or is_not_digit(
                7) or is_not_digit(8)):
            digit_list = [2, 3, 5, 6, 7, 8]
            # change the errors
            master_change_not_digit(digit_list)

    if is_digit(-5) and is_digit(-4) and is_digit(-3) and is_digit(-2) and is_digit(-1):
        if ocr_list[-1] == "1":
            ocr_list[-1] = ""

    ocr_list = [str(i) for i in ocr_list]
    number_plate_str = "".join(ocr_list)
    print(">>>>>>>>>>>>>>>>>>>>", number_plate_str)
    return number_plate_str
    # except Exception as e:
    #     print(e)
