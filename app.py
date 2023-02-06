import os
import my_utils as mu
import config as cfg
from model_init import get_model_device, load_model_from_path
import cv2
from time import sleep
# from persistqueue.exceptions import Empty
import requests
import shutil

NUMBER_PLATE_PATH = "number_plate.pt"
URL = "https://4ee8-106-201-230-210.ngrok.io/image_copy"
# load models

NUMBER_PLATE_MODEL = load_model_from_path(NUMBER_PLATE_PATH)
DEVICE = get_model_device()

if __name__ == "__main__":
    list_files = os.listdir("../0")
    list_files.sort()
    # i=0
    # print(list_files)
    # while True:
    for i in range(len(list_files)):
    # while True:
        try:
            image_path = "../0/" + list_files[i]
            # image_path = cfg.frames_queue.get()
            # cfg.logger.info(f"Image get from QUEUE ::{image_path}")
            print(image_path)
            image_path_url = image_path.replace("http://192.168.0.204","/home/mihir/adani")
            image_path2 = image_path.replace("raw_frames","events")
            image_path3 = image_path2.rsplit(os.sep,1)
            print(image_path3)
            new_path = (
                image_path3[0]
                + os.sep
                + "frames"
            )
            # # TODO: fetch from path
            camera_id = image_path.split("/")[-2]
            #
            # print(f"camera_id: {camera_id} || image_path: {image_path2}")
            # # TODO: bring image to local hdd from network
            img, img0, image_path = mu.load_image_from_disk(
                image_path, int(cfg.IMAGE_SIZE)
            )
            print("")
            if image_path:
                result,output_img = mu.predict(
                    NUMBER_PLATE_MODEL,
                    img,
                    img0,
                    DEVICE,
                    0.5,
                    float(cfg.IOU_THRESHOLD),
                )
            
                # cfg.logger.info(f"Result of Fire Model::{result}")
                # store data to db
                update_path = new_path + os.sep + image_path.split('/')[-1]
                print("update_path>>>>>>>",update_path)
                # break
        #         image_path_update = update_path.replace(
        #             cfg.DETECTION_PATH,
        #             cfg.ROOT_URL)
                PARAMS = {'image_source_path': image_path_url, 'image_des_path': update_path}
                print(f"img: {PARAMS['image_source_path'].split('/')[-1]}")
                # json_data = json.dumps(PARAMS).encode('utf8')
                # break
                if len(result['detection']) > 0:
                    cv2.imwrite(update_path, output_img)
                #     # cfg.result_queue.put(image_path_url)
                #     print("------------------------------------------------------------------------------------")
                #     # r = requests.post(url=URL, params=PARAMS)
                #     shutil.copyfile(image_path_url, update_path)


                # mu.store_results(cfg.COMPANY_ADMIN_ID, camera_id,
                #      image_path_url,update_path , result)
        # except Empty as empty_ex:
        #     print(f"queue empty, sleeping for {cfg.SLEEP_TIME} seconds")
        #     sleep(cfg.SLEEP_TIME)
        except Exception as e:
            print(e)
