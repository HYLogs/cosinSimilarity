import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from numpy import dot
from numpy.linalg import norm
import PIL.Image as Image
import time

def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))

def compare(targetPath, imgPaths):
    # url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/feature_vector/5" #클라우드
    url = "imagenet_mobilenet_v2_100_96_feature_vector_5" #로컬
    IMAGE_RES = 96
    model = tf.keras.Sequential([
        hub.KerasLayer(url, input_shape=(IMAGE_RES, IMAGE_RES, 3))
    ])

    # target 이미지 특징 벡터를 가져옵니다.
    grace_hopper = Image.open(targetPath).resize((IMAGE_RES, IMAGE_RES))
    grace_hopper = np.array(grace_hopper)/255.0
    data = grace_hopper.reshape(1, 96, 96, 3)

    # compare 이미지 특징 벡터를 가져온 후 코사인 유사도 계산을 한 후 유사도 리스트를 생성
    compares = []
    for imgPath in imgPaths:
        grace_hopper = Image.open(imgPath).resize((IMAGE_RES, IMAGE_RES))
        grace_hopper = np.array(grace_hopper)/255.0
        data = np.append(data, grace_hopper.reshape(1, 96, 96, 3), axis=0)
    
    Vectors = model.predict(data)
    TargetVector = Vectors[0]
    CompareVectors = Vectors[1:]
    for CompareVector in CompareVectors:
        similarity = cos_sim(TargetVector, CompareVector)
        compares.append(similarity.item())
    return compares

start = time.time()
result = compare("1.png", [str(i)+".png" for i in range(2,6)])
print(result)
print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간