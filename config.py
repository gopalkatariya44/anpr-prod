# https://github.com/pytorch/pytorch/issues/46971#issuecomment-722775106
import logging
import logging.handlers as handlers
import os
# from persistqueue import MySQLQueue
ROI_API_ENDPOINT = ""
COMPANY_ADMIN_ID = 13

# model configs
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# model <> camera mapping
MODEL_CAM_MAP = {"fire": [0,1], "face": [1]}
# 192.168.0.24
# mongo config
MONGO_HOST = "localhost"
MONGO_PORT = 27017
MONGO_USER = ""
MONGO_PASS = ""
MONGO_AUTH_DB_NAME = ""
MONGO_DB_NAME = "adani_avtron"
MONGO_COLL_NAME = "results_1"

# queue settings
QUEUE_NAME = "database/frame_db_6"
RESULT_QUEUE_NAME = "database/frame_db_7"
# TODO: test mysql queue and change this to mysql based queue
db_conf = {
    "host": "localhost",
    "user": "admin",
    "passwd": "Admin#123",
    "db_name": "frame_db",
    "port": 3306,
}

# logging
FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(funcName)2s() : %(message)s")
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
# LOG_FILE_PATH = basedir.rsplit('/', 1)[0] + os.sep + 'logs'
# LOG_LEVEL = "DEBUG"
#
# logging.basicConfig(level=LOG_LEVEL)
# logger = logging.getLogger('logger_name')
#
# f_handler = handlers.TimedRotatingFileHandler(
#     'log'+os.sep+'adani_log_gen_model', when='midnight', interval=1)
# f_handler.setFormatter(FORMATTER)
# f_handler.setLevel(LOG_LEVEL)
# logger.addHandler(f_handler)

# frames_queue = MySQLQueue(name=QUEUE_NAME, **db_conf)
# result_queue = MySQLQueue(name=RESULT_QUEUE_NAME, **db_conf)
ROOT_URL = "http://192.168.0.204"
DETECTION_PATH = "/home/dev1034/rushik_workspace/tusker/adani/events"
SLEEP_TIME = 60
