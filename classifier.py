import cv2, numpy as np, pandas as pd, seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as ac
import os, ssl, time
from PIL import Image
import PIL.ImageOps

x,y = fetch_openml('mnist_784', version=1, return_X_y=True)
xtrain,xtest,ytrain,ytest = tts(x,y,random_state=9, train_size = 7500, test_size=2500)
xps = xtrain/255.0
x_testscale = xtest/255.0
model=LogisticRegression(solver='saga', multi_class='multinomial').fit(xps,ytrain)

def getPrediction(image):
    iampil = Image.open(image)
    imbw = iampil.convert('L')
    iamresize = imbw.resize((28,28), Image.ANTIALIAS)
    pixelFilter = 20
    minPixel = np.percentile(iamresize, pixelFilter)
    iamresize_invert = np.clip(iamresize-minPixel, 0,255)
    maxPixel = np.max(iamresize)
    iamresize_invert = np.asarray(iamresize_invert)/maxPixel
    test_sample = np.array(iamresize_invert).reshape(1,784)
    test_predict = model.predict(test_sample)
    return test_predict[0] 