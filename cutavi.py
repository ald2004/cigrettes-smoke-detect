import base64
import glob
import os
import ssl
import tqdm
import urllib
from urllib.request import Request, urlopen
import traceback
import uuid
import cv2


def getann(data_dir):
    imagefiles = glob.glob(data_dir + '/*')
    for _, image_path in tqdm.tqdm(enumerate(imagefiles)):
        filename, _ = os.path.splitext(image_path)
        if os.path.exists(filename + '.txt'): continue
        with open(image_path, "rb") as imageFile:
            base64_bytes = base64.b64encode(imageFile.read())
        base64_string = base64_bytes.decode('utf-8')

        host = 'https://smoking.market.alicloudapi.com'
        path = '/ai_image_detect/ai_smoking/v1'
        appcode = 'c9c41ef9f05a4bbda06c6c9550848e24'
        bodys = {}
        url = host + path

        bodys['IMAGE'] = '''data:image/jpeg;base64,''' + base64_string

        post_data = urllib.parse.urlencode(bodys).encode("utf-8")
        request = Request(url, post_data)
        request.add_header('Authorization', 'APPCODE ' + appcode)
        request.add_header('Content-Type', 'application/x-www-form-urlencoded; charset=UTF-8')
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        try:
            response = urlopen(request, context=ctx)
            content = response.read()
        except Exception as exc:
            print(f"{image_path} has error")
            print(traceback.format_exc())
            continue

        with open(filename + '.txt', 'wb') as fid:
            fid.write(content)

        # if (content):
        # print(content.decode('utf-8'))
        # ret = json.loads(content.decode('utf-8'))
        # xx=open('','')


def batchrename():
    path = "./"
    for filename in os.listdir(path):
        a, b = os.path.splitext(filename)
        my_dest = uuid.uuid4().hex + b
        my_source = path + filename
        my_dest = path + my_dest
        os.rename(my_source, my_dest)


def cutframes(path, outputpicpath='./'):
    FPS = 20
    avifiles = glob.glob(path + '/*.avi')
    for _, avifile in tqdm.tqdm(enumerate(avifiles)):
        count = 0
        vidObj = cv2.VideoCapture(avifile)
        totalframes = vidObj.get(cv2.CAP_PROP_FRAME_COUNT)
        while True:
            try:
                ret, frame_read = vidObj.read()
                if not ret or count >= totalframes:
                    break
                picfilename = os.path.join(outputpicpath, uuid.uuid4().hex) + '.jpg'
                cv2.imwrite(picfilename, frame_read)
            except:
                print(traceback.format_exc())
                pass
            count += FPS
            vidObj.set(1, count)

cutframes('/home/yuanpu/smoke/avis','/home/yuanpu/smoke/picturefromavis')
