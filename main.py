import os
import json
import torch
from urllib import request
import glob
import io
import pathlib
import tempfile
import argparse
from typing import List, Optional, Union
import soundfile

from google.cloud import storage
from google.cloud import speech

import numpy as np
import torch
import torchvision.ops.boxes as bops

import norfair
from norfair import Detection, Paths, Tracker, Video
from norfair.distances import frobenius, iou

from PIL import Image
import easyocr

import nltk
from jamo import h2j, j2hcj
import moviepy.editor as mp
from scipy.io.wavfile import write

import sys
sys.argv = ['']

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'estnuxlear.json'
DISTANCE_THRESHOLD_BBOX: float = 0.7
DISTANCE_THRESHOLD_CENTROID: int = 30
MAX_DISTANCE: int = 10000

project_id = "estnuxlear"
bucket_name = "estnuxlear"


def matching_text(candidates: list, target: str):
    "input과 list of text를 입력으로 받아서 input과 가장 유사한 text의 index를 반환하는 함수"
    distances = []
    for candidate in candidates:
        distances.append(nltk.edit_distance(j2hcj(h2j(candidate[0])), j2hcj(h2j(target))) * (1 + abs(len(candidate[0]) - len(target)) / len(candidate[0])))
    return candidates[distances.index(min(distances))]

def distance(candidates, target):
    distances = []
    for candidate in candidates:
        distances.append((candidate[1][0] - target[0])**2 + (candidate[1][1] - target[1])**2)
    return candidates[distances.index(min(distances))]

class YOLO:
    def __init__(self, model_path: str, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception(
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        # automatically set device if its None
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if not os.path.exists(model_path):
            os.system(
                f"wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/{os.path.basename(model_path)} -O {model_path}"
            )

        # load model
        try:
            #self.model = torch.hub.load("WongKinYiu/yolov7", "custom", model_path)
            self.model = torch.load("./yolov7_custom.pt")
        except:
            raise Exception("Failed to load model from {}".format(model_path))

    def __call__(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 720,
        classes: Optional[List[int]] = None,
    ) -> torch.tensor:

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=image_size)
        return detections


def center(points):
    return [np.mean(np.array(points), axis=0)]


def yolo_detections_to_norfair_detections(
    yolo_detections: torch.tensor, track_points: str = "centroid"  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections"""
    norfair_detections: List[Detection] = []

    if track_points == "centroid":
        detections_as_xywh = yolo_detections.xywh[0]
        for detection_as_xywh in detections_as_xywh:
            centroid = np.array(
                [detection_as_xywh[0].item(), detection_as_xywh[1].item()]
            )
            scores = np.array([detection_as_xywh[4].item()])
            norfair_detections.append(
                Detection(
                    points=centroid,
                    scores=scores,
                    label=int(detection_as_xywh[-1].item()),
                )
            )
    elif track_points == "bbox":
        detections_as_xyxy = yolo_detections.xyxy[0]
        for detection_as_xyxy in detections_as_xyxy:
            bbox = np.array(
                [
                    [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                    [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()],
                ]
            )
            scores = np.array(
                [detection_as_xyxy[4].item(), detection_as_xyxy[4].item()]
            )
            norfair_detections.append(
                Detection(
                    points=bbox, scores=scores, label=int(detection_as_xyxy[-1].item())
                )
            )

    return norfair_detections

parser = argparse.ArgumentParser(description="Track objects in a video.")

parser.add_argument(
    "--detector-path", type=str, default="./yolov7_custom.pt", help="YOLOv7 model path"
)
parser.add_argument("--files", type=str, default="dataset/샘플데이터셋_cam1_01.mp4", help="Video files to process")
parser.add_argument(
    "--img-size", type=int, default="720", help="YOLOv7 inference size (pixels)"
)
parser.add_argument(
    "--conf-threshold",
    type=float,
    default="0.25",
    help="YOLOv7 object confidence threshold",
)
parser.add_argument(
    "--iou-threshold", type=float, default="0.45", help="YOLOv7 IOU threshold for NMS"
)
parser.add_argument(
    "--classes",
    nargs="+",
    type=int,
    help="Filter by class: --classes 0, or --classes 0 2 3",
)
parser.add_argument(
    "--device", type=str, default='cuda', help="Inference device: 'cpu' or 'cuda'"
)
parser.add_argument(
    "--track-points",
    type=str,
    default="bbox",
    help="Track points: 'centroid' or 'bbox'",
)
args = parser.parse_args()

model = YOLO(args.detector_path, device=args.device)


def main():
    # load environment variable
    api_url = os.environ['REST_ANSWER_URL']
    data_path = '/home/agc2022/dataset/'
    #data_path = '/data1/dict/4th/dataset/'
    reader = easyocr.Reader(['ko'])
    pred_json = json.load(open(os.path.join(data_path, 'pre_defined.json'), 'r'))
    vid_list = glob.glob(os.path.join(data_path, '*.mp4'))
    ans_cnt = 0
    
    for vid in vid_list:
        template = {
            "team_id": "est_nuxlear",
            "secret": "2P07pqvyEfNZvYG5",
            "answer_sheet": {}
        }
        #cam_no = int(vid)
        cam_no = vid.split('.mp4')[0][-2:]
        #ms = vid.split('.mp4')[0][-1]

        video = Video(input_path=vid)

        target_obs = {}

        distance_function = iou if args.track_points == "bbox" else frobenius

        distance_threshold = (
            DISTANCE_THRESHOLD_BBOX
            if args.track_points == "bbox"
            else DISTANCE_THRESHOLD_CENTROID
        )

        tracker = Tracker(
            distance_function=distance_function,
            distance_threshold=distance_threshold,
        )

        frame_idx = 0
        for frame in video:
            yolo_detections = model(
                frame,
                conf_threshold=args.conf_threshold,
                iou_threshold=args.iou_threshold,
                image_size=args.img_size,
                classes=args.classes,
            )
            detections = yolo_detections_to_norfair_detections(
                yolo_detections, track_points=args.track_points
            )
            tracked_objects = tracker.update(detections=detections)
            #norfair.draw_boxes(frame, detections)
            #norfair.draw_tracked_boxes(frame, tracked_objects)
            for to in tracked_objects:
                if to.label == 0 and to.age > 50:
                    x_diff = to.estimate[1][0] - to.estimate[0][0]
                    y_diff = to.estimate[1][1] - to.estimate[0][1]
                    if x_diff > y_diff * 1.2:
                        if to.id not in target_obs:
                            target_obs[to.id] = {'to':to, 'age':0, 'base':[to.estimate[0][0], to.estimate[0][1]], 'frame':frame}
                        target_obs[to.id]['age'] += 1
                        if abs(sum(target_obs[to.id]['base']) - (to.estimate[0][0] + to.estimate[0][1])) > 20:
                            del target_obs[to.id]

            #video.write(frame)
            frame_idx += 1

        for obs in target_obs:
            if target_obs[obs]['age'] > 50:
                #im = Image.fromarray(target_obs[obs]['frame'])
                #fname = f"your_file_{obs}.jpg"
                #im.save(fname)
                result = reader.readtext(target_obs[obs]['frame'])
                place_dict = []
                for item in result:
                    place_dict.append([item[1], item[0][0]])
                matchings = []
                for place in pred_json['place']:
                    matchings.append([place, matching_text(place_dict, place)[1]])
                print(distance(matchings, target_obs[obs]['base'])[0])
                place = distance(matchings, target_obs[obs]['base'])[0]
                #im = Image.fromarray(target_obs[obs]['frame'][930:,1620:])
                #fname = f"time_{obs}.jpg"
                #im.save(fname)
                result = reader.readtext(target_obs[obs]['frame'][930:,1620:])
                if len(result) > 0:
                    print(':'.join(result[0][1].split('.')))
                    time = ':'.join(result[0][1].split('.'))
                    tmp_answer = {"cam_no":cam_no, "mission":"1", "answer":{"source":"신체적이상", "place" : place, "event":"기타쓰러짐", "person":"UNCLEAR", "time":time}}
                    template['answer_sheet'] = tmp_answer
                    if data_path == '/data1/dict/4th/dataset/':
                        json.dump(template, open(f'{cam_no}_{obs}_{ans_cnt}.json', 'w'))
                        ans_cnt += 1
                    else:
                        # apply unicode to str json data
                        data = json.dumps(template).encode('unicode-escape')

                        # request ready
                        req =  request.Request(api_url, data=data)

                        # POST to API server
                        resp = request.urlopen(req)

                        # # check POST result
                        status = eval(resp.read().decode('utf8'))
                        print("received message: "+status['msg'])
        #############################################################################################################
        try:
            clip = mp.VideoFileClip(vid)
            a = clip.audio.to_soundarray()


            destination_blob_name = "temp.wav"

            storage_client = storage.Client(project_id)
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(destination_blob_name)

            b = a[:,0] * 32768
            c = b.clip(min=-32768, max=32767).astype(np.int16)

            bytes_wav = bytes()
            byte_io = io.BytesIO(bytes_wav)
            write(byte_io, 44100, c)

            blob.upload_from_file(byte_io)

            config = {"sample_rate_hertz":44100,
              "language_code": "ko-KR",
              "encoding": speech.RecognitionConfig.AudioEncoding.LINEAR16,
              "audio_channel_count": 1,
              "enable_word_time_offsets": True,
              "enable_automatic_punctuation": True,
              "model":'default'}

            audio = {"uri": 'gs://estnuxlear/temp.wav'}
            client = speech.SpeechClient()
            operation = client.long_running_recognize(config=config, audio=audio)
            response = operation.result()

            v_dict = {
                "020811" : ["주임", "선임", "책임", "대리", "과장", "차장", "부장", "사원", "사장"],
                "020121" : ["생명", "목숨", "죽기", "죽을", "죽어", "죽고", "죽여", "다치"],
                "02051" : ["돈", "만원", "내놔", "은행"]
            }

            paragraphs = []
            lb = ["020811", "020121", "02051", "020819"]
            s = -1
            e = -1
            lc = [0,0,0,1]
            for result in response.results:
                for w in result.alternatives[0].words:
                    #print(w.start_time.seconds, w.word)

                    if e > 0 and w.start_time.seconds - e > 10:
                        paragraphs.append({'start':s, 'end':e, 'lc':lb[np.argmax(lc)]})
                        s = w.start_time.seconds
                        lc = [0,0,0,1]

                    for item in v_dict["020811"]:
                        if item in w.word:
                            lc[0] += 1

                    for item in v_dict["020121"]:
                        if item in w.word:
                            lc[1] += 1

                    for item in v_dict["02051"]:
                        if item in w.word:
                            lc[2] += 1

                    if s < 0:
                        s = w.start_time.seconds
                    e = w.end_time.seconds
            paragraphs.append({'start':s, 'end':e, 'lc':lb[np.argmax(lc)]})

            for p in paragraphs:
                if p['end'] - p['start'] > 5:
                    event = p['lc']
                    person = "UNCLEAR"
                    person_num = "UNCLEAR"
                    frame = clip.get_frame(p['start'])
                    #im = Image.fromarray(frame[930:,1620:])
                    #fname = "start_time.jpg"
                    #im.save(fname)
                    result = reader.readtext(frame[930:,1620:])
                    if len(result) > 0:
                        n = int(result[0][1].replace('.', ''))
                        res = f'{n:<06d}'
                        time_start = res[0] + res[1] + ':' + res[2] + res[3] + ':' + res[4] + res[5]

                    frame = clip.get_frame(p['end'])
                    #im = Image.fromarray(frame[930:,1620:])
                    #fname = "end_time.jpg"
                    #im.save(fname)
                    result = reader.readtext(frame[930:,1620:])
                    if len(result) > 0:
                        n = int(result[0][1].replace('.', ''))
                        res = f'{n:<06d}'
                        time_end = res[0] + res[1] + ':' + res[2] + res[3] + ':' + res[4] + res[5]
                        tmp_answer = {"cam_no":cam_no, "mission":"2", "answer":{"event":event, "person":person, "time_start":time_start, "time_end":time_end, "person_num":person_num}}
                        template['answer_sheet'] = tmp_answer
                        if data_path == '/data1/dict/4th/dataset/':
                            json.dump(template, open(f'{cam_no}_{obs}_{ans_cnt}.json', 'w'))
                            ans_cnt += 1
                        else:
                            # apply unicode to str json data
                            data = json.dumps(template).encode('unicode-escape')

                            # request ready
                            req =  request.Request(api_url, data=data)

                            # POST to API server
                            resp = request.urlopen(req)

                            # # check POST result
                            status = eval(resp.read().decode('utf8'))
                            print("received message: "+status['msg'])
        except:
            pass
        #############################################################################################################################

#        tmp_answer = {"cam_no":cam_no, "mission":"3", "answer":{"recycle":recycle, "person_color":person_color, "time":time}}
#        template['answer_sheet'] = tmp_answer

#        if data_path == 'dataset/':
#            json.dump(template, open(f'{cam_no}.json', 'w'))
#        else:
            # apply unicode to str json data
#            data = json.dumps(template).encode('unicode-escape')

            # request ready
#            req =  request.Request(api_url, data=data)

            # POST to API server
#            resp = request.urlopen(req)

            # # check POST result
#            status = eval(resp.read().decode('utf8'))
#            print("received message: "+status['msg'])

#             if "OK" == status['result']:
#                 print("data requests successful!!")
#             elif "ERROR" == status['result']:    
#                 raise ValueError("Receive ERROR status. Please check your source code.")

    
    # request end of mission message
    message_structure = {
    "team_id": "est_nuxlear",
    "secret": "2P07pqvyEfNZvYG5",
    "end_of_mission": "true"
    }
    if data_path == '/data1/dict/4th/dataset/':
        json.dump(message_structure, open('eom.json', 'w'))
    else:
        # json dump & encode unicode
        tmp_message = json.dumps(message_structure).encode('unicode-escape')
        request_message = request.Request(api_url, data=tmp_message)
        resp = request.urlopen(request_message) # POST

        status = eval(resp.read().decode('utf8'))
        print("received message: "+status['msg'])

#         if "OK" == status['result']:
#             print("data requests successful!!")
#         elif "ERROR" == status['result']:    
#             raise ValueError("Receive ERROR status. Please check your source code.")    
    
if __name__ == "__main__":
    main()