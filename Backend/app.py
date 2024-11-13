'''
Description: 
Author: Qing Shi
Date: 2022-11-20 19:14:42
LastEditTime: 2022-11-21 00:43:57
'''
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import make_response
import json
import os
from flask import render_template
from flask import Flask
from flask import request, jsonify
from flask_cors import CORS
import os
import zipfile
import csv
import shutil
import json
import numpy as np
import torch
from pkg_resources import packaging
import io
import PIL
import PIL.Image
import time
import random
import json
import requests
import cv2
from segment_anything import SamPredictor
import torch
from segment_anything import sam_model_registry

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH="D:/sam_vit_h_4b8939.pth"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
mask_predictor = SamPredictor(sam)
import re
from loguru import logger
# from attrdict import AttrDict
import openai
from sentence_transformers import SentenceTransformer, util
text_model = SentenceTransformer('all-MiniLM-L6-v2')
import csv
All_traits=[]
ct=0
with open('D:/NFTSearch/NFTSearch/Backend/traits_new.csv',encoding='utf-8') as f:
  #reader=csv.reader(f)
    reader = csv.reader(x.replace('\0', '') for x in f)
    for item in reader:
        if ct>0:
            tag=item[0]
            if len(tag)>30:
                tag=tag[:30]
            All_traits.append(tag)
        ct=ct+1
print("All_traits")
print(len(All_traits))

#### Generation code#######
# from diffusers import StableDiffusionImg2ImgPipeline
# from diffusers import StableDiffusionPipeline
# model_id = "runwayml/stable-diffusion-v1-5"
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
# pipe = pipe.to("cuda")
# pipe1=StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
# pipe1 = pipe1.to("cuda")
#####Generation code#####

from diffusers import StableDiffusionInstructPix2PixPipeline

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")


openai.api_key=""
# import tensorflow as tf
with open('D:/NFTSearch/NFTSearch/Backend/static/contract_address.json') as json_file:
    address_dic = json.load(json_file)
from PIL import Image
import clip
previous_intent_ids=[[]]
previous_paths=[[]]
TextQuery=[]
im_paths=[]
addresses=[]
c=0
im_paths0=[]
price_dic={}
genstate=[0]
instruct_state=[0]
mask_list=[]
mask_state=[0]
current_seg_path=[""]
new_mask_list=[]
new_mask_state=[0]
new_current_seg_path=[""]

# with open('static/Crypto.csv') as f:
#     csvreader = csv.reader(f)
#     for line in csvreader:
#         if c>0:
#             im_paths0.append(line[1])
#         c=c+1
# c=0
# im_paths1=[]
# with open('static/Doodle_A.csv') as f:
#     csvreader = csv.reader(f)
#     for line in csvreader:
#         if c>0:
#             im_paths1.append(line[1])
#         c=c+1
# c=0
# im_paths2=[]
# with open('static/CoolCat.csv') as f:
#     csvreader = csv.reader(f)
#     for line in csvreader:
#         if c>0:
#             im_paths2.append(line[1])
#         c=c+1
# all_paths=[im_paths0, im_paths1, im_paths2]
# all_ps=im_paths0+im_paths1+im_paths2
all_paths=[]
all_ps=[]

model, preprocess = clip.load("RN50")
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size
# model2, preprocess = clip.load("RN50")
# checkpoint = torch.load("static/model_2.pt")
# checkpoint = torch.load("static/model_5.pt")
# model2.load_state_dict(checkpoint['model_state_dict'])
# model2 = torch.load("D:/weights/model_6.pt")
# model2 = torch.load("D:/weights/model_7.pt")

w1=9
w2=1
# model2 = torch.load("D:/weights/model_8.pt")
# model2.cuda().eval()
# print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
# print("Input resolution:", input_resolution)
# print("Context length:", context_length)
# print("Vocab size:", vocab_size)
model2=model

trait_tokens0 = clip.tokenize(All_traits).cuda()
with torch.no_grad():
    trait_features00 = model.encode_text(trait_tokens0).float()
    trait_features01= model2.encode_text(trait_tokens0).float()
trait_features00 /= trait_features00.norm(dim=-1, keepdim=True)
trait_features00=trait_features00.cpu().numpy()
trait_features01 /= trait_features01.norm(dim=-1, keepdim=True)
trait_features01=trait_features01.cpu().numpy()
trait_features=(w1*trait_features00+w2*trait_features01)/(w1+w2)

print("trait_ems")
print(trait_features.shape)

import os
import skimage
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
# LLMconfig = AttrDict({
#     'host': 'ssz105.ust.hk',
#     # 'host': '8.210.18.245',
#     # 'host': 'localhost',
#     'port': '5001',
#     'api': '/get-chatbot-response'
# })
class ChatbotClient:
    def __init__(self, config:dict) -> None:
        self.host = config.host
        self.port = config.port
        self.api = config.api
        logger.debug('client inited')

    def query(self, text:str)-> str:
        url = f'http://{self.host}:{self.port}{self.api}'
        data = {'text': text}
        headers = {'content-type': 'application/json' }
        logger.debug(f'url:{url}, data:{data}')
        resp = requests.post(url, json=data, headers=headers)
        resp.raise_for_status()
        logger.debug(f'resp.text:{resp.text}')
        return json.loads(resp.text)
from collections import OrderedDict
va=[]
bestOffer=[]
hisOffer=[]
hisSold=[]
for i in range(5000):
    va.append(0)
    bestOffer.append(0)
    hisOffer.append(0)
    hisSold.append(0)
import csv
bestOfferDic={}
hisOfferDic={}
hisSoldDic={}
with open('CC_currentDataFrame.csv',encoding='utf-8') as f:
  #reader=csv.reader(f)
    reader = csv.reader(x.replace('\0', '') for x in f)
    for item in reader:
        bestOfferDic["CoolCat"+str(item[0])]=item[1]
        price_dic["CoolCat"+str(item[0])]=item[1]
with open('DD_currentDataFrame.csv',encoding='utf-8') as f:
  #reader=csv.reader(f)
    reader = csv.reader(x.replace('\0', '') for x in f)
    for item in reader:
        bestOfferDic["Doodles"+str(item[0])]=item[1]
        price_dic["Doodles"+str(item[0])]=item[1]

all_prices=[]
# for p in all_paths:
#     s=p.split("/")[-1]
#     token_id=''.join([i for i in s if i.isdecimal()])
#     if "CoolCat" in p:
#         all_prices.append(bestOfferDic["CoolCat"+token_id])
#     elif "Doodle" in p:
#         all_prices.append(bestOfferDic["Doodles"+token_id])
#     else:
#         all_prices.append(0)

im = Image.open("785.png") 
width, height = im.size
imgs=[preprocess(im)]
image_input = torch.tensor(np.stack(imgs)).cuda()
with torch.no_grad():
    image_features = model.encode_image(image_input).float()
image_features /= image_features.norm(dim=-1, keepdim=True)
im_features=image_features.cpu().numpy()
print(im_features.shape)

# collection_names=["Cryptopunk","Doodles","CoolCat","VeeFriends","PXN Ghost Division","HAPE PRIME","CyberBrokers","Creature World"]
# collection_names=collection_names+os.listdir("C:/Users/24197/Documents/NFT/More NFT collections")+os.listdir("C:/Users/24197/Documents/NFT/More NFT collections2")
# collection_names=["Cryptopunk","Doodles","CoolCat","VeeFriends","Bored Ape Yacht Club","PXN Ghost Division","Pudgy Penguins","The Doge Pound","Killer GF"]
collection_names=["Cryptopunk","CoolCat","VeeFriends","Bored Ape Yacht Club","PXN Ghost Division","Pudgy Penguins","The Doge Pound","Killer GF"]
from numpy import random,argsort,sqrt, sort
from pylab import plot,show
def cosDis(x,D):
    return 1-np.dot(D,x)/(np.linalg.norm(D,axis=1)*np.linalg.norm(x))
def cosD(x,y):
    return 1-(np.dot(x,y)/np.linalg.norm(y))/np.linalg.norm(x)
def EuDis(x,D):
    return np.linalg.norm(x-D,axis=1)
def EuD(x,y):
    return np.linalg.norm(x-y)
def knn_search_C(x, D, K):
    """ find K nearest neighbours of data among D """
    ndata = D.shape[0]
    K = K if K < ndata else ndata
    sqd=cosDis(x,D)
    print(sqd.shape)
    idx = sqd.argsort() # sorting
    return idx[:K]
def GPTProcess(textQuery):
    poslist=[]
    clothlist=[]
    price_adj=""
    neglist=[]
    priceRange=[-1,100]
    # resp=openai.Completion.create(
    #     model="text-davinci-003",
    #     prompt="I say "+"'"+textQuery+"'"+"\n\nBased on the text above, list the things I like or want, the things I don't like, the clothing I like, the price I want (high or low), the price upperbound, and the price lowerbound. Use N/A for empty list, separate list elements with comma, only use high or low to describe the price:",
    #     temperature=0.5,
    #     max_tokens=60,
    #     top_p=1.0,
    #     frequency_penalty=0.8,
    #     presence_penalty=0.0
    #     )

    # print(resp["choices"][0]["text"])
    # try:
    #     results0=resp["choices"][0]["text"].split("\n\n")[1].split("\n")
    # except:
    #     print(resp["choices"][0]["text"])

    # Likes=results0[0]
    # Nlikes=results0[1]
    # Clothes=results0[2]
    # PriceAjs=results0[3]
    # PriceUp=results0[4]
    # PriceDown=results0[5]
    # if "N/A" not in Likes:
    #     try:
    #         Likelist=Likes.split(":")[1].split(",")
    #     except:
    #         try:
    #             Likelist=[Likes.split(":")[1]]
    #         except:
    #             Likelist=[Likes]
    #     for l in Likelist:
    #         if l.strip()!="":
    #             poslist.append(l.strip())
    # if len(poslist)==0:
    #     poslist.append(textQuery)


    # if "N/A" not in Nlikes:
    #     Nlikelist=Nlikes.split(":")[1].split(",")
    #     for l in Nlikelist:
    #         if l.strip()!="":
    #             neglist.append(l.strip())
    # if "N/A" not in Clothes:
    #     Clothslist=Clothes.split(":")[1].split(",")
    #     for l in Clothslist:
    #         if l.strip()!="":
    #             clothlist.append(l.strip())
    # if "N/A" not in PriceAjs:
    #     PriceAjlist=PriceAjs.split(":")[1].split(",")
    #     price_adj=PriceAjlist[0].strip()  
    # if "N/A" not in PriceUp:  
    #     try:
    #         PriceUplist=PriceUp.split(":")[1].split(",")
    #         if len(re.findall(r"\d+\.?\d*",PriceUplist[0]))>0:
    #             priceRange[1]=float(re.findall(r"\d+\.?\d*",PriceUplist[0])[0])
    #     except:
    #         print(PriceUp)
    # if "N/A" not in PriceDown:
    #     try:
    #         PriceDownlist=PriceDown.split(":")[1].split(",")
    #         if len(re.findall(r"\d+\.?\d*",PriceDownlist[0]))>0:
    #             priceRange[0]=float(re.findall(r"\d+\.?\d*",PriceDownlist[0])[0])
    #     except:
    #         print(PriceDown)
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt='''Q: long hair girl in Azuki collection wearing glasses, the hair should be purple or red; price lower than 1ETH.I prefer higher rarity. But I don’t like gray background and red shirt.
    A: Positive: [["S: long hair girl with glasses and purple hair", "long hair", "girl", "glasses","purple hair","Collection_Azuki"], ["S: long hair girl with glasses and red hair", "long hair", "girl", "glasses","red hair","Collection_Azuki"]]
    Negative: [["gray background"],["red shirt"]]
    Price bounds: ["","1ETH"]
    Ranking requirements: {"rarity": "higher"}
    Q: I want cat wearing glasses or dog wearing red shirt.The background can be red or blue.Price should be lower than 3ETH and no less than 1ETH.Warmer color is better. But I don't like yellow background.
    A: Positive: [["S: cat with glasses and red background", "cat", "glasses", "red background"], ["S: cat with glasses and blue background", "cat", "glasses", "blue background"], ["S: dog with red shirt and red background", "dog","red shirt","red background"], ["S: dog with red shirt and blue background", "dog","red shirt","blue background"]]
    Negative: [["yellow background"]]
    Price bounds: ["1ETH","3ETH"]
    Ranking requirements: {"color": "warmer"}
    Q: Red cat
    A: Positive: [["S: red cat", "red cat"]]
    Negative: []
    Price bounds: ["",""]
    Ranking requirements: {}
    Q: Give me the most expensive cat
    A: Positive: [["S: cat", "cat"]]
    Negative: []
    Price bounds: ["",""]
    Ranking requirements: {"price": "higher"}
    Q: '''+textQuery+'''
    ''',
    # prompt='''Q: long hair girl in Azuki collection wearing glasses, the hair should be purple or red; price lower than 1ETH.I prefer higher rarity. But I don’t like gray background and red shirt.
    # A: Positive: [["S: long hair girl with glasses and purple hair", "long hair", "girl", "glasses","purple hair","Collection_Azuki"], ["S: long hair girl with glasses and red hair", "long hair", "girl", "glasses","red hair","Collection_Azuki"]]
    # Negative: [["gray background"],["red shirt"]]
    # Price bounds: ["","1ETH"]
    # Ranking requirements: {"rarity": "higher"}
    # Keywords: ["long hair", "girl", "glasses", "C_Azuki collection", "the hair should be purple or red", "price lower than 1ETH", "higher rarity", "N_gray background", "N_red shirt"]
    # Q: I want cat wearing glasses or dog wearing red shirt.The background can be red or blue.Price should be lower than 3ETH and no less than 1ETH.Warmer color is better. But I don't like yellow background.
    # A: Positive: [["S: cat with glasses and red background", "cat", "glasses", "red background"], ["S: cat with glasses and blue background", "cat", "glasses", "blue background"], ["S: dog with red shirt and red background", "dog","red shirt","red background"], ["S: dog with red shirt and blue background", "dog","red shirt","blue background"]]
    # Negative: [["yellow background"]]
    # Price bounds: ["1ETH","3ETH"]
    # Ranking requirements: {"color": "warmer"}
    # Keywords: ["cat", "glasses", "dog", "red shirt", "The background can be red or blue", "Price should be lower than 3ETH and no less than 1ETH", "Warmer color", "N_yellow background"]
    # Q: Red cat
    # A: Positive: [["S: red cat", "red cat"]]
    # Negative: []
    # Price bounds: ["",""]
    # Ranking requirements: {}
    # Keywords: ["Red cat"]
    # Q: Give me the most expensive cat
    # A: Positive: [["S: cat", "cat"]]
    # Negative: []
    # Price bounds: ["",""]
    # Ranking requirements: {"price": "higher"}
    # Keywords: ["Red cat"]
    # Q: '''+textQuery+'''
    # ''',
    temperature=0.5,
    max_tokens=220,
    top_p=1.0,
    frequency_penalty=0.8,
    presence_penalty=0.0
    )
    print(response)
    resps=response["choices"][0]["text"].split("\n")[1:]
    print(resps)
    for i in range(len(resps)):
        resps[i]=resps[i].replace("\n","")
    print("resps0")
    print(resps[0][-2:])
    if resps[0][-2:]==",]":
        resps[0]=resps[0][:-2]+"]"
    pos_list0 = json.loads(resps[0].split("Positive: ")[1])
    neg_list0 = json.loads(resps[1].split("Negative: ")[1])
    price_bnd0 = json.loads(resps[2].split("Price bounds: ")[1])
    rank_rq0 = json.loads(resps[3].split("Ranking requirements: ")[1])
    print(pos_list0)
    print(neg_list0)
    print(price_bnd0)
    print(rank_rq0)
    pos_elements=[]
    for pl in pos_list0:
        x=pl[0].split(": ")[1]
        poslist.append(x)
        els=[]
        for i in range(1,len(pl)):
            els.append(pl[i])
        pos_elements.append(els)
    for pl in neg_list0:
        x=pl[0]
        neglist.append(x)
    priceRange=[-1,100]
    try:
        priceRange[0]=float(re.findall(r"\d+\.?\d*",price_bnd0[0])[0])
    except:
        priceRange[0]=priceRange[0]
    try:
        priceRange[1]=float(re.findall(r"\d+\.?\d*",price_bnd0[1])[0])
    except:
        priceRange[1]=priceRange[1]  
    print(poslist)
    print(neglist)
    print(priceRange)   
    if "price" in list(rank_rq0.keys()):
        if "igh" in rank_rq0["price"] or "ex" in rank_rq0["price"]:
            price_adj="high"
        else:
            price_adj="low"

    return poslist, neglist, clothlist, priceRange, price_adj, pos_elements

all_embeds=np.array([0,1])
# all_embeds0=np.load("static/Crypto.npy")
# all_embeds1=np.load("static/Doodle_A.npy")
# all_embeds2=np.load("static/CoolCat.npy")

# all_embeds0=np.load("static/Cryptopunk.npy")
# all_embeds1=np.load("static/Doodles.npy")
# all_embeds2=np.load("static/CoolCat.npy")
# all_ems=[all_embeds0, all_embeds1,all_embeds2]
all_ems=[]
all_ems0=[]
all_ems1=[]
# tag_embeds=np.load("D:/NFTSearch/NFTSearch/Backend/traitEmbeds.npy")
# tag_embeds=np.load("D:/NFTSearch/NFTSearch/Backend/traitEms.npy")
tag_embeds=np.load("D:/NFTSearch/NFTSearch/Backend/traitEms_new.npy")
print("tag_embeds: ")
print(tag_embeds.shape)
# all_es=np.concatenate((all_embeds0, all_embeds1,all_embeds2))
for na in collection_names:
    # all_ems0.append(np.load("static/"+na+".npy"))
    # all_ems1.append(np.load("static/"+na+"_new.npy"))
    try:
        # all_ems.append((5*np.load("static/"+na+".npy")+2*np.load("static/new_col/"+na+"_new.npy"))/7)
        # all_ems.append((5*np.load("static/"+na+".npy")+2*np.load("static/new_col1/"+na+"_new.npy"))/7)
        # all_ems.append((5*np.load("static/"+na+".npy")+2*np.load("static/new_col2/"+na+"_new.npy"))/7)
        # all_ems.append((5*np.load("static/"+na+".npy")+2*np.load("static/"+na+".npy"))/7)
        # all_ems.append((w1*np.load("static/"+na+".npy")+w2*np.load("static/new_col3/"+na+"_new.npy"))/(w1+w2))
        all_ems.append((w1*np.load("static/"+na+".npy")+w2*np.load("static/"+na+".npy"))/(w1+w2))
    except:
        print(na)
        print(np.load("static/"+na+".npy").shape)
        print(np.load("static/"+na+"_new.npy").shape)
all_es=np.concatenate(all_ems)

print("all_es")
print(all_es.shape)
# all_es1=np.concatenate(all_ems1)
all_text_ems=np.load("D:/text_embeds.npy")
###
###加速测试
from sklearn.neighbors import BallTree
embeds_norm=np.linalg.norm(all_es,axis=1)
tag_embeds_norm=np.linalg.norm(tag_embeds,axis=1)
trait_embeds_norm=np.linalg.norm(trait_features,axis=1)
norm_embeds=all_es/(embeds_norm.reshape((-1,1)))
norm_tag_embeds=tag_embeds/(tag_embeds_norm.reshape((-1,1)))
norm_trait_embeds=trait_features/(trait_embeds_norm.reshape((-1,1)))
ball=BallTree(norm_embeds, metric='euclidean')
tag_ball=BallTree(norm_tag_embeds, metric='euclidean')
trait_ball=BallTree(norm_trait_embeds, metric='euclidean')

text_norm=np.linalg.norm(all_text_ems,axis=1)
norm_text=all_text_ems/(text_norm.reshape((-1,1)))
text_ball=BallTree(norm_text, metric='euclidean')

combine_ems=np.concatenate([2*all_es,all_text_ems],axis=1)
combine_ems_norm=np.linalg.norm(combine_ems,axis=1)
norm_combine_ems=combine_ems/(combine_ems_norm.reshape((-1,1)))
combine_ball=BallTree(norm_combine_ems, metric='euclidean')
print("combine_shape")
print(combine_ems.shape)
def ball_search_C(x, K):
    """ find K nearest neighbours of data among D """
    x_norm=np.linalg.norm(x)
    x=x.reshape((1,-1))
    x=x/x_norm
    result=ball.query(x, k=K)
    idx=result[1]
    idx=idx[0]
    # print(idx)
    # print(result)
    return idx
def tag_ball_search_C(x, K):
    """ find K nearest neighbours of data among D """
    x_norm=np.linalg.norm(x)
    x=x.reshape((1,-1))
    x=x/x_norm
    result=tag_ball.query(x, k=K)
    idx=result[1]
    idx=idx[0]
    # print(idx)
    # print(result)
    return idx
def trait_ball_search_C(x, K):
    """ find K nearest neighbours of data among D """
    x_norm=np.linalg.norm(x)
    x=x.reshape((1,-1))
    x=x/x_norm
    result=trait_ball.query(x, k=K)
    idx=result[1]
    idx=idx[0]
    # print(idx)
    # print(result)
    return idx

def text_ball_search_C(x, K):
    """ find K nearest neighbours of data among D """
    x_norm=np.linalg.norm(x)
    x=x.reshape((1,-1))
    x=x/x_norm
    result=text_ball.query(x, k=K)
    idx=result[1]
    idx=idx[0]
    # print(idx)
    # print(result)
    return idx
def combine_ball_search_C(x, K):
    """ find K nearest neighbours of data among D """
    x_norm=np.linalg.norm(x)
    x=x.reshape((1,-1))
    x=x/x_norm
    result=combine_ball.query(x, k=K)
    idx=result[1]
    idx=idx[0]
    # print(idx)
    # print(result)
    return idx
###
###

# print(all_es.shape)
# print(all_es[0].shape)
# print("all_ems: ")
# print(len(all_ems))
# print("all_cols:")
# print(len(collection_names))
# signature0=np.mean(all_embeds0, axis=0)
# signature1=np.mean(all_embeds1, axis=0)
# signature2=np.mean(all_embeds2, axis=0)
# all_signatures=[signature0, signature1, signature2]
all_signatures=[]
for i in range(len(all_ems)):
    all_signatures.append(np.mean(all_ems[i],axis=0))

import json
# with open('static/cryptopunk.json') as json_file:
#     data1 = json.load(json_file)
# with open('static/Doodles1.json') as json_file:
#     data2 = json.load(json_file)
# with open('static/CoolCat.json') as json_file:
#     data3 = json.load(json_file)
# labs2=[]
# all_cats2=[]
# names1=[]
# names2=[]
# names3=[]
# labs3=[]
# all_cats3=[]
# labs1=[]
# all_cats1=[]
all_labs=[]
# all_prices=[]
# features1=[]
# features2=[]
# path2="D:/Doodles-PartA/DOOD_PartA/"
# file_list2=os.listdir(path2)
# path1="static/Crypto/"
# file_list1=os.listdir(path1)
# path3="D:/CoolCat/CoolCat/"
# file_list3=os.listdir(path3)
all_file_lists=[]
collection_paths=[]
labs_list=[]
all_categories=[]

# for i in range(len(data1["tokens"])):
#     try:
#         attr=data1["tokens"][i]["metadata"]["attributes"]
#         name=data1["tokens"][i]["metadata"]["name"]
#     except:
#         attr=json.loads(data1["tokens"][i]["metadata"])["attributes"]
#         name=json.loads(data1["tokens"][i]["metadata"])["name"]
#     if name+".png" in file_list1:
#         lab_str=''
#         names1.append(name)
#         typename=str(type(attr[0])).split("'")[1]
#         label_list=[]
#         for k in range(len(attr)):
#             if typename=='dict':
#                 label_list.append(attr[k]["trait_type"]+": "+attr[k]["value"])
#                 lab_str=lab_str+attr[k]["trait_type"]+": "+attr[k]["value"]+", "
#                 if attr[k]["trait_type"]+": "+attr[k]["value"] not in all_cats1:
#                     all_cats1.append(attr[k]["trait_type"]+": "+attr[k]["value"])
#             else:
#                 label_list.append(attr[k])
#                 lab_str=lab_str+attr[k]+", "
#                 if attr[k] not in all_cats1:
#                     all_cats1.append(attr[k])
#         labs1.append(label_list)
#         all_labs.append(lab_str)
# all_file_lists.append(names1)
# all_categories.append(all_cats1)
# labs_list.append(labs1)
# for i in range(len(data2["tokens"])):
#     try:
#         attr=data2["tokens"][i]["metadata"]["attributes"]
#         name=data2["tokens"][i]["metadata"]["name"]
#     except:
#         attr=json.loads(data2["tokens"][i]["metadata"])["attributes"]
#         name=json.loads(data2["tokens"][i]["metadata"])["name"]
#     if name+".png" in file_list2:
#         lab_str=''
#         names2.append("Doodle"+name.split("#")[-1])
#         typename=str(type(attr[0])).split("'")[1]
#         label_list=[]
#         for k in range(len(attr)):
#             if typename=='dict':
#                 label_list.append(attr[k]["trait_type"]+": "+attr[k]["value"])
#                 lab_str=lab_str+attr[k]["trait_type"]+": "+attr[k]["value"]+", "
#                 if attr[k]["trait_type"]+": "+attr[k]["value"] not in all_cats2:
#                     all_cats2.append(attr[k]["trait_type"]+": "+attr[k]["value"])
#             else:
#                 label_list.append(attr[k])
#                 lab_str=lab_str+attr[k]+", "
#                 if attr[k] not in all_cats2:
#                     all_cats2.append(attr[k])
#         labs2.append(label_list)
#         all_labs.append(lab_str)
# all_file_lists.append(names2)
# all_categories.append(all_cats2)
# labs_list.append(labs2)
# for i in range(len(data3["tokens"])):
#     try:
#         attr=data3["tokens"][i]["metadata"]["attributes"]
#         name=data3["tokens"][i]["metadata"]["name"]
#     except:
#         attr=json.loads(data3["tokens"][i]["metadata"])["attributes"]
#         name=json.loads(data3["tokens"][i]["metadata"])["name"]
#     if name+".png" in file_list3:
#         lab_str=''
#         names3.append("Cool"+name.split("#")[-1])
#         typename=str(type(attr[0])).split("'")[1]
#         label_list=[]
#         for k in range(len(attr)):
#             if typename=='dict':
#                 label_list.append(attr[k]["trait_type"]+": "+attr[k]["value"])
#                 lab_str=lab_str+attr[k]["trait_type"]+": "+attr[k]["value"]+", "
#                 if attr[k]["trait_type"]+": "+attr[k]["value"] not in all_cats3:
#                     all_cats3.append(attr[k]["trait_type"]+": "+attr[k]["value"])
#             else:
#                 label_list.append(attr[k])
#                 lab_str=lab_str+attr[k]+", "
#                 if attr[k] not in all_cats3:
#                     all_cats3.append(attr[k])
#         labs3.append(label_list)
#         all_labs.append(lab_str)
# all_file_lists.append(names3)
# all_categories.append(all_cats3)
# labs_list.append(labs3)
# All_traits=[]
trait2imID_dic={}
other_ks=['Pants', 'Type', 'Hair Style', 'Pants Color', 'Necklace', 'Hat', 'Shirt', 'Hat Color', 'Shirt Color', 'Shoes', 'value']
all_ks=[]
all_tns=[]
all_cats0=[]
all_labels=[]
for na in collection_names:

    # pa="D:/NFTSearch/NFTSearch/Frontend/src/assets/static/compressed/"+na+"/"
    pa="D:/Limited_col1/Limited_col1/"+na+"/"+na+"/"
    unorder=os.listdir(pa)
    l=len(unorder)
    collection_paths.append(pa)
    files=[]
    labs=[]

    with open('D:/NFTSearch/NFTSearch/Frontend/src/assets/static/'+"more-meta-data/"+na+'.json') as json_file:
        data = json.load(json_file)
    
    for i in range(len(data["tokens"])):
        try:
            attr=data["tokens"][i]["metadata"]["attributes"]
            name=data["tokens"][i]["token_id"]
        except:
            try:
                attr=json.loads(data["tokens"][i]["metadata"])["attributes"]
                name=data["tokens"][i]["token_id"]
            except:
                name=data["tokens"][i]["token_id"]
                attr=''
        if name+".png" in unorder:
            lab_str=''
            files.append("./static/compressed/"+na+"/"+name+".png")
            try:
                price_dic[na+name]=bestOfferDic[na+name]
            except:
                price_dic[na+name]=int(5*np.random.rand()*1000)/1000
            label_list=[]
            if attr!='' and len(attr)>0:
                # try:
                # print(type(attr[0]))
                typename=str(type(attr[0])).split("'")[1]
                for k in range(len(attr)):
                    if typename=='dict':
                        try:
                            label_list.append(str(attr[k]["trait_type"])+": "+str(attr[k]["value"]))
                            lab_str=lab_str+str(attr[k]["trait_type"])+": "+str(attr[k]["value"])+", "
                            # if str(attr[k]["trait_type"])+": "+str(attr[k]["value"]) not in all_cats0:
                            all_cats0.append(str(attr[k]["trait_type"])+": "+str(attr[k]["value"]))
                        except:
                            # print(attr[k].keys())
                            ks=list(attr[k].keys())
                            for ke in ks:
                                if ke not in all_ks:
                                    all_ks.append(ke)
                                #label_list.append(str(ke)+": "+str(attr[k][ke]))
                                lab_str=lab_str+str(ke)+": "+str(attr[k][ke])+", "
                                # if str(ke)+": "+str(attr[k][ke]) not in all_cats0:
                                    # print(str(ke)+": "+str(attr[k][ke]))
                                if str(ke)!="value" and str(ke)!="Type":
                                    all_cats0.append(str(ke)+": "+str(attr[k][ke]))
                                    label_list.append(str(ke)+": "+str(attr[k][ke]))
                                else:
                                    all_cats0.append(str(attr[k][ke]))
                                    label_list.append(str(attr[k][ke]))
                            # if 'Type' in ks:
                            #     print(ks)
                            #     print(attr[k])
                    # elif typename=='list':

                    else:
                        if typename not in all_tns:
                            all_tns.append(typename)
                        label_list.append(attr[k])
                        lab_str=lab_str+attr[k]+", "
                        # if attr[k] not in all_cats0:
                        all_cats0.append(attr[k])
                # except:
                #     u=1
                #     if len(attr)>0:
                #         print(type(attr[0]))
            labs.append(label_list)
            all_labels.append(label_list)
            all_labs.append(lab_str)
        
    # all_categories.append(all_cats0)
    all_file_lists.append(files)
    labs_list.append(labs)
    all_paths.append(files)
    all_ps=all_ps+files
print("all_labels")
print(len(all_labels))
All_CAS=list(set(all_cats0))
print("all_tns: ")
print(all_tns)
print("all_ks: ")
print(all_ks)
print("all_ps: ")
print(len(all_ps))
print("all_traits: ")
print(len(All_CAS))
# print(All_CAS)
# with open('D:/NFTSearch/NFTSearch/Backend/traits_new.csv', 'w', encoding="utf-8", newline='') as file:            
#     writer = csv.writer(file)
#     writer.writerow(["trait"])
#     for i in range(len(All_CAS)):
#         writer.writerow([All_CAS[i]])
price_string = json.dumps(price_dic)
# with open('./static/price_dic.json', 'w') as outfile:
#     outfile.write(price_string)
with open('./static/price_dic.json') as json_file:
    PriceDic = json.load(json_file)

with open('./static/new_price_dic.json') as json_file:
    newPriceDic = json.load(json_file)
new_keys=list(newPriceDic.keys())
for nk in new_keys:
    PriceDic[nk]=newPriceDic[nk]
# all_text_tokens=clip.tokenize([desc for desc in all_labs]).cuda()

# textfeatures=[]
# for lab_str in all_labs:
#     tokens=clip.tokenize(lab_str).cuda()
#     with torch.no_grad():
#         text_features = model.encode_text(tokens).float()
#     text_features1=text_features.cpu().numpy()
#     textfeatures.append(text_features1[0])
# all_text_features1=np.array(textfeatures)  
all_directories=collection_names
# all_text_features1=np.load("static/all_test_features.npy")
# all_im_tx_es=(all_text_features1+all_es)/2
# APP_ROOT=os.path.dirname(os.path.abspath(__file__))
# app = Flask(__name__)
# @app.route('/')
Keywords=['']
Elements=['']
Related_Tags=['']

lens=[100]
import base64
FILE_ABS_PATH = os.path.dirname(__file__)

app = Flask(__name__)
CORS(app)

@app.route('/api/test/hello/', methods=['POST'])
def hello_resp():
    params = request.json
    # msg = int(params['msg'])
    # print(msg)
    return "hello VUE"

@app.route('/api/test/getValue/',methods=['GET'])
def getValue():
    path_dic={}
    for i in range(lens[0]):
        # with open(va[i], 'rb') as img_f:
        #     img_stream = img_f.read()
        #     img_stream = base64.b64encode(img_stream)
        path_dic[str(i)] = va[i]+"#"+str(bestOffer[i])+"#"+Keywords[0]+"----"+Elements[0]+"----"+Related_Tags[0]+"----"+TextQuery[-1] +"#"+addresses[i]
    # print(path_dic)
    return jsonify(path_dic)#jsonify({"0": va[0], "1": va[1],"2":va[2],"3": va[3],"4":va[4],"5": va[5], "6": va[6], "7": va[7], "8": va[8], "9": va[9]})

# def getValue():
#     path_dic={}
#     for i in range(100):
#         path_dic[str(i)]=va[i]
#     return jsonify(path_dic)#jsonify({"0": va[0], "1": va[1],"2":va[2],"3": va[3],"4":va[4],"5": va[5], "6": va[6], "7": va[7], "8": va[8], "9": va[9]})
@app.route('/api/test/getImg/',methods=['GET','POST'])
def getImg():
	# 通过表单中name值获取图片
    start_time=time.time()
    # f_d_data = request.get_json(silent=True)
    # print(f_d_data)
    # f = f_d_data['data']['image']
    f = request.files["image"]
    imgData=f.read()
    byte_stream = io.BytesIO(imgData)  
    im = Image.open(byte_stream) 
    width, height = im.size
    imgs=[preprocess(im)]
    image_input = torch.tensor(np.stack(imgs)).cuda()
    with torch.no_grad():
        image_features0 = model.encode_image(image_input).float()
        image_features1 = model2.encode_image(image_input).float()
    image_features0 /= image_features0.norm(dim=-1, keepdim=True)
    im_features0=image_features0.cpu().numpy()
    image_features1 /= image_features1.norm(dim=-1, keepdim=True)
    im_features1=image_features1.cpu().numpy()
    im_features=(w1*im_features0+w2*im_features1)/(w1+w2)
    print(im_features.shape)
    img=im_features[0]


    KNNs=ball_search_C(img, 100)
    price_cans=[]
    im_paths=[]
    for i in range(100):
        im_paths.append(str(all_ps[KNNs[i]]))
    for i in range(len(im_paths)):
        s=im_paths[i].split('/')[-1]
        token_id=''.join([i for i in s if i.isdecimal()])
        if "Doodle" in im_paths[i]:
            try:
                price_c=float(PriceDic["Doodles"+str(int(token_id))])#float(re.findall(r"\d+\.?\d*",bestOfferDic["Doodles"+str(int(token_id))])[0])
            except:
                price_c=0
                # print(bestOfferDic["Doodles"+str(int(token_id))])
            
        elif "CoolCat" in im_paths[i]:
            try:
                price_c=float(PriceDic["CoolCat"+str(int(token_id))])#float(re.findall(r"\d+\.?\d*",bestOfferDic["CoolCat"+str(int(token_id))])[0])
            except:
                price_c=0
                # print(bestOfferDic["CoolCat"+str(int(token_id))])
        else:
            #bestOffer[i]=0
            try:
                na=im_paths[i].split("/")[3]
                price_c=float(PriceDic[na+str(int(token_id))])
            except:
                price_c=int(5*np.random.rand()*1000)/1000
        price_cans.append(price_c)
    for i in range(100):
        va[i]=str(im_paths[i])
        s=im_paths[i].split('/')[-1]
        token_id=''.join([i for i in s if i.isdecimal()])
        bestOffer[i]=str(price_cans[i])+" WETH"
        colname=im_paths[i].split("/")[3]
        addresses.append(address_dic[colname])
    # collection_distances=[np.linalg.norm(img-sig) for sig in all_signatures]
    # idx=np.argmin(collection_distances)
    # all_embeds=all_ems[idx]
    # im_paths=all_paths[idx]
    # print("image:")
    # print(img.shape)
    # print(all_embeds.shape)
    # KNNs=knn_search_C(img, all_embeds, 100)
    # for i in range(100):
    #     va[i]=str(im_paths[KNNs[i]])
    #     s=im_paths[i].split('/')[-1]
    #     token_id=''.join([i for i in s if i.isdecimal()])
    #     if "Doodle" in im_paths[i]:
    #         bestOffer[i]=bestOfferDic["Doodles"+str(int(token_id))]
    #     elif "CoolCat" in im_paths[i]:
    #         bestOffer[i]=bestOfferDic["CoolCat"+str(int(token_id))]
    #     else:
    #         # bestOffer[i]=0
    #         bestOffer[i]=str(int(5*np.random.rand()*1000)/1000)+" WETH"
        
    #     colname=im_paths[i].split("/")[3]
    #     addresses.append(address_dic[colname])
    
    end_time=time.time()
    print("Time per query: ")
    print(end_time-start_time)
    # for i in range(11):
    #     va[i]=str((2*pre[i]+pre1[i])/3)
        #va[i]=str(pre1[i])
    os.remove("static/query.png")
    im.save("static/query.png")

    return jsonify({"error": 1001, "msg": "上传失败"})
@app.route('/api/test/getText/',methods=['Get','POST'])
def getText():
    f = request.form["text"]
    tag_fil=0
    try:
        ft= request.form["tag"]
        query_tags=json.loads(ft)
        # if query_tags!=[[""]]:
        for qtl in query_tags:
            for qt in qtl:
                if qt!="":
                    tag_fil=1
        print("tag_pass")
        print(ft)
    except:
        ft=""
    #因为测试前端接收问题暂时把tag disable掉因为相关的后端代码导致im_paths2被定死在previous_paths上了（上面try的逻辑判断有问题导致tag为空时也会set tag_fil=1）
    # tag_fil=0
    #
    print(f)
    print("tag:")
    print(ft)

    keylist=f.split(",")
    for i in range(len(keylist)):
        keylist[i]=keylist[i].strip()
    # if f!="" and len(keylist)>0:
    #     Keywords[0]=""  
    #     for i in range(len(keylist)):
    #         Keywords[0]=Keywords[0]+","+keylist[i]
    TextQuery.append(f)
    price=0
    # if "price" in f:
    #     pr=f.split("/")[-1]
    #     pr_text=''.join([i for i in pr if i.isdecimal() or i=="."])
    #     price=float(pr_text)
    f0=f
    f1=""
    GPT=0
    multiple=0
    f_list=[]
    neg=0
    poslist=[]
    neglist=[]
    clothlist=[]
    priceRange=[-1,100]
    price_adj=""
    pos_elements=[]
    # if "I " in f or " me " in f:
    poslist, neglist, clothlist, priceRange, price_adj, pos_elements=GPTProcess(f)
    # F0,F1,F2,F3=GPTProcess(f)
    # f0=F0[0]
    # f1=F1[0]
    print("poslist")
    print(poslist)
    print("neglist")
    print(neglist)
    print("pos_elements")
    print(pos_elements)
    related_tags=[]
    for i in range(len(pos_elements)):
        rtags=[]
        for j in range(len(pos_elements[i])):
            # tag_tokens0 = clip.tokenize(pos_elements[i][j]).cuda()
            # with torch.no_grad():
            #     tag_features00 = model.encode_text(tag_tokens0).float()
                # tag_features01= model2.encode_text(tag_tokens0).float()
            # tag_features00 /= tag_features00.norm(dim=-1, keepdim=True)
            # tag_features00=tag_features00.cpu().numpy()
            # tag_features01 /= tag_features01.norm(dim=-1, keepdim=True)
            # tag_features01=tag_features01.cpu().numpy()
            # tag_features0=tag_features00
            tag_features0=text_model.encode([pos_elements[i][j]], convert_to_tensor=False)
            tag_KNNs=tag_ball_search_C(tag_features0[0], 5)
            rtags.append(tag_KNNs)
        related_tags.append(rtags)
    print(related_tags)
    print("All_traits")
    print(len(All_traits))
    Related_Tags[0]=""
    for tl in related_tags:
        print(tl)
        for d in range(len(tl)):
            for i in range(5):
                Related_Tags[0]=Related_Tags[0]+","+All_traits[tl[d][i]]
            # Related_Tags[0]=Related_Tags[0]+"****"
        Related_Tags[0]=Related_Tags[0]+"++++"
    print(Related_Tags[0])
    GPT=1
    if GPT==0 and "," in f:
        fs=f.split(",")
        for t in fs:
            f_list.append(t)
    elif GPT==1:
        fs=poslist
        for t in fs:
            f_list.append(t)        
        if f!="" and len(poslist)>0:
            Keywords[0]=""  
            for i in range(len(poslist)):
                Keywords[0]=Keywords[0]+","+poslist[i]  
            Elements[0]=""
            for i in range(len(pos_elements)):
                for j in range(len(pos_elements[i])):
                    Elements[0]=Elements[0]+","+pos_elements[i][j]
                Elements[0]=Elements[0]+"****"      
    # keylist=f.split(",")
    # for i in range(len(keylist)):
    #     keylist[i]=keylist[i].strip()


    candidates=[]
    
        # for i in range(len(labs1)):
        #     indi=0
        #     for lab in labs1[i]:
        #         if f.lower() in lab.lower():
        #             indi=1
        #             break
        #     if indi==1:
        #         candidates.append("static/Crypto/"+names1[i]+".png")
        # for i in range(len(labs2)):
        #     indi=0
        #     for lab in labs2[i]:
        #         if f.lower() in lab.lower():
        #             indi=1
        #             break
        #     if indi==1:
        #         candidates.append("static/Doodle_A/"+names2[i]+".png")
        # for i in range(len(labs3)):
        #     indi=0
        #     for lab in labs3[i]:
        #         if f.lower() in lab.lower():
        #             indi=1
        #             break
        #     if indi==1:
        #         candidates.append("static/CoolCat/"+names3[i]+".png")

        #text attribute
        # for k in range(len(labs_list)):
        #     labs=labs_list[k]
        #     for i in range(len(labs)):
        #         indi=0
        #         for lab in labs[i]:
        #             if f.lower() in lab.lower():
        #                 indi=1
        #                 break
        #         if indi==1:
        #             candidates.append("./static/"+all_directories[k]+"/"+all_paths[k][i].split("/")[-1])

        # random.shuffle(candidates)
    im_paths=[]
    im_paths0=[]
    im_paths01=[]
    im_paths1=[]
    im_paths2=[]
    intent_id_list=[]
    intent_id_list0=[]
    intent_id_list01=[]
    intent_id_list1=[]
    intent_id_list2=[]
    if len(candidates)>=1000:
        im_paths0=candidates[:1000]
    ##not use GPT
    # elif GPT==0:
    #     if len(f_list)==0:
    #         im_paths0=candidates
    #         text_tokens = clip.tokenize(f).cuda()
    #         with torch.no_grad():
    #             text_features0 = model.encode_text(text_tokens).float()
    #             text_features1= model2.encode_text(text_tokens).float()
    #         text_features0 /= text_features0.norm(dim=-1, keepdim=True)
    #         text_features0=text_features0.cpu().numpy()
    #         text_features1 /= text_features1.norm(dim=-1, keepdim=True)
    #         text_features1=text_features1.cpu().numpy()
    #         text_features=(6*text_features0+text_features1)/7
    #         # test ball search
    #         # KNNs=knn_search_C(text_features1[0], all_es, 500)
    #         KNNs=ball_search_C(text_features[0], 1000)

    #         # print("all_es: ")
    #         # print(all_es.shape)
    #         embeds1=[]
    #         for u in range(1000):
    #             embeds1.append(all_es[KNNs[u]].reshape((1,-1)))
    #         embeds2=np.concatenate(embeds1)
    #         for k in range(0,1000-len(candidates)):
    #             im_paths0.append(str(all_ps[KNNs[k]]))
    #         # imgs2=[]
    #         # for t in range(len(im_paths0)):
    #         #     im=Image.open("D:/NFTSearch/NFTSearch/Frontend/src/assets"+im_paths0[t][1:])
    #         #     imgs2.append(preprocess(im))
    #         # image_input2 = torch.tensor(np.stack(imgs2)).cuda()
    #         # with torch.no_grad():
    #         #     image_features2 = model2.encode_image(image_input2).float()
    #         # image_features2 /= image_features2.norm(dim=-1, keepdim=True)
    #         # im_features2=image_features2.cpu().numpy()
    #         # KNNs2=knn_search_C(text_features2[0], embeds2, 480)
    #         # KNNs=ball_search_C(text_features2[0], 400)
    #         pointer=0
    #         count=0
    #         indexes=[]
    #         while count<100 and pointer<1000:
    #             duplicate=0
    #             for j in range(len(im_paths1)):
    #                 if cosD(embeds2[pointer],embeds2[indexes[j]])<0.00001:
    #                     duplicate=1
    #                     break
    #             if duplicate==0:
    #                 im_paths1.append(im_paths0[pointer])
    #                 indexes.append(pointer)
    #                 count=count+1
    #             pointer=pointer+1
    #         # for k in range(100):
    #         #     im_paths.append(im_paths0[k])
    #         # for j in range(len(candidates),10):
    #         #    im_paths.append(0) 
    #     else:
    #         im_paths0=candidates
    #         key_paths=[[] for t in f_list]
    #         for g in range(len(f_list)):
    #             t=f_list[g]
    #             key_paths0=[]
    #             text_tokens = clip.tokenize(t).cuda()
    #             unit_num=int(1000/len(f_list))
    #             scope=len(f_list)*unit_num
    #             with torch.no_grad():
    #                 text_features0 = model.encode_text(text_tokens).float()
    #                 text_features1= model2.encode_text(text_tokens).float()
    #             text_features0 /= text_features0.norm(dim=-1, keepdim=True)
    #             text_features0=text_features0.cpu().numpy()
    #             text_features1 /= text_features1.norm(dim=-1, keepdim=True)
    #             text_features1=text_features1.cpu().numpy()
    #             text_features=(6*text_features0+text_features1)/7
                
    #             KNNs=ball_search_C(text_features[0], scope)

    #             # print("all_es: ")
    #             # print(all_es.shape)
    #             embeds1=[]
    #             for u in range(scope):
    #                 embeds1.append(all_es[KNNs[u]].reshape((1,-1)))
    #             embeds2=np.concatenate(embeds1)
    #             for k in range(0,scope-len(candidates)):
    #                 key_paths0.append(str(all_ps[KNNs[k]]))

    #             pointer=0
    #             count=0
    #             indexes=[]
    #             while count<int(unit_num*0.3) and pointer<scope:
    #                 duplicate=0
    #                 for j in range(len(key_paths[g])):
    #                     if cosD(embeds2[pointer],embeds2[indexes[j]])<0.00001:
    #                         duplicate=1
    #                         break
    #                 if duplicate==0:
    #                     key_paths[g].append(key_paths0[pointer])
    #                     indexes.append(pointer)
    #                     count=count+1
    #                 pointer=pointer+1 
    #         for u in range(int(unit_num*0.28)):
    #             for r in range(len(key_paths)):
    #                 im_paths1.append(key_paths[r][u]) 
    #     # print("im_paths1:")
    #     # print(im_paths1)
    #     im_paths2=im_paths1         
    
    ##use GPT
    elif tag_fil==1:
        query_tags=json.loads(ft)
        change_id=[]
        tag_max=[]
        cor_tag_paths0=[[] for tg in query_tags]
        cor_tag_ids0=[[] for tg in query_tags]
        for int_id in range(len(query_tags)):
            filtag_list=[]
            for ta in query_tags[int_id]:
                if ta!="":
                    filtag_list.append(ta)
                    if int_id not in change_id:
                        change_id.append(int_id)
            tag_max.append(filtag_list)
        print("tag_max")
        print(tag_max)
        change_KNNs={}
        for cid in change_id:            
            for h in range(len(all_labels)):
                mat=1
                for ta in tag_max[cid]:
                    if ta not in all_labels[h]:
                        mat=0
                        break
                if mat==1:
                    cor_tag_paths0[cid].append(all_ps[h])
                    cor_tag_ids0[cid].append(h)
            print("cor_tag_ids0[cid]")
            print(cor_tag_ids0[cid])
            #     if mat==0:
            #         break
            # if mat==1:
            #     cor_tag_paths0.append(all_ps[h])
            #     cor_tag_ids0.append(h)
            embeds1=[]
            for u in range(len(cor_tag_ids0[cid])):
                embeds1.append(all_es[cor_tag_ids0[cid][u]].reshape((1,-1)))
            embeds2=np.concatenate(embeds1)
            remain_text=poslist[cid]
            text_tokens0 = clip.tokenize(remain_text).cuda()
            with torch.no_grad():
                text_features00 = model.encode_text(text_tokens0).float()
                text_features01= model2.encode_text(text_tokens0).float()
            text_features00 /= text_features00.norm(dim=-1, keepdim=True)
            text_features00=text_features00.cpu().numpy()
            text_features01 /= text_features01.norm(dim=-1, keepdim=True)
            text_features01=text_features01.cpu().numpy()
            text_features0=(w1*text_features00+w2*text_features01)/(w1+w2)
            result_len=min(len(cor_tag_ids0[cid]),int(len(previous_intent_ids[0])/len(query_tags)))
            # print("text-image")
            # print(text_features0.shape)
            # print(embeds2.shape)
            knns=knn_search_C(text_features0.reshape((1024,)),embeds2,result_len)
            #KNNs=ball_search_C(text_features0[0], inital_scope)
            change_KNNs[str(cid)]=knns
            print("change_knn")
            print(len(change_KNNs[str(cid)]))
        im_paths2=previous_paths[0]
        intent_id_list2=previous_intent_ids[0]
        # print("im_path2")
        # print(im_paths2)
        for cid in change_id:
            pter=0
            for pid in range(len(previous_intent_ids[0])):
                if pter<len(change_KNNs[str(cid)]):
                    if previous_intent_ids[0][pid]==cid:
                        im_paths2[pid]=all_ps[cor_tag_ids0[cid][change_KNNs[str(cid)][pter]]]
                        pter=pter+1
                        print("get_you")
        # print("n_im_path2")
        # print(im_paths2)
    else:
        inital_scope=1500
        embeds3=[]
        if len(poslist)==1:
            im_paths0=candidates
            text_tokens0 = clip.tokenize(poslist[0]).cuda()
            with torch.no_grad():
                text_features00 = model.encode_text(text_tokens0).float()
                text_features01= model2.encode_text(text_tokens0).float()
            text_features00 /= text_features00.norm(dim=-1, keepdim=True)
            text_features00=text_features00.cpu().numpy()
            text_features01 /= text_features01.norm(dim=-1, keepdim=True)
            text_features01=text_features01.cpu().numpy()
            text_features0=(w1*text_features00+w2*text_features01)/(w1+w2)
               
           ##add element similarity
            features_list=[]

            for pos_el in pos_elements[0]:
                t_tokens0=clip.tokenize(pos_el).cuda()
                with torch.no_grad():
                    t_features00 = model.encode_text(t_tokens0).float()
                    t_features01= model2.encode_text(t_tokens0).float()
                t_features00 /= t_features00.norm(dim=-1, keepdim=True)
                t_features00=t_features00.cpu().numpy()
                t_features01 /= t_features01.norm(dim=-1, keepdim=True)
                t_features01=t_features01.cpu().numpy()
                t_features0=(w1*t_features00+w2*t_features01)/(w1+w2)
                features_list.append(t_features0)
            

            
            
            el_count=1
            el_features=features_list[0]
            for t_fea in features_list:
                el_features=el_features+t_fea
                el_count=el_count+1
            el_features=el_features/el_count      
            # el_features=np.mean(features_list)     
            # text_features0=(7*text_features0+3*el_features)/10 
            text_features0=(20*text_features0+0*el_features)/20 
            print("el")
            print(el_features.shape)
            ##add element similarity
            print("combine")
            text_emb=text_model.encode([f], convert_to_tensor=False)
            combine_q=np.concatenate([2*text_features0,text_emb], axis=1)
            print(combine_q.shape)
            KNNs=ball_search_C(text_features0[0], inital_scope)
            # KNNs=combine_ball_search_C(combine_q[0], inital_scope)

            # print("all_es: ")
            # print(all_es.shape)
            embeds1=[]
            text_embeds1=[]
            for u in range(inital_scope):
                embeds1.append(all_es[KNNs[u]].reshape((1,-1)))
                text_embeds1.append(all_text_ems[KNNs[u]].reshape((1,-1)))

            embeds2=np.concatenate(embeds1)
            element_sim_list=[]
            element_per_list=[]
            for kf in range(len(features_list)):
                text_fea=features_list[kf]
                el_sim=text_fea @ embeds2.T
                # if kf==0:
                #     el_per=np.percentile(el_sim,50)
                # else:
                #     el_per=np.percentile(el_sim,20)
                el_per=np.mean(el_sim)-0.22*(np.mean(el_sim)-np.min(el_sim))
                print("el_per:")
                print(el_per)
                element_sim_list.append(el_sim)
                element_per_list.append(el_per)
            text_embeds2=np.concatenate(text_embeds1)
            tag_feature_list=[]
            for pos_el in pos_elements[0]:
                tag_features0=text_model.encode([pos_el], convert_to_tensor=False)
                tag_feature_list.append(tag_features0)

            for kf in range(len(tag_feature_list)):
                tag_fea=tag_feature_list[kf]
                el_sim=tag_fea @ text_embeds2.T
                # if kf==0:
                #     el_per=np.percentile(el_sim,50)
                # else:
                #     el_per=np.percentile(el_sim,20)
                el_per=np.mean(el_sim)#+0.22*(np.mean(el_sim)-np.min(el_sim))
                print("el_per:")
                print(el_per)
                element_sim_list.append(el_sim)
                element_per_list.append(el_per)
            # tag_features=np.mean(tag_feature_list,axis=1)

            # tag_count=0
            # tag_features=tag_feature_list[0]-tag_feature_list[0]
            # for t_fea in tag_feature_list:
            #     tag_features0=tag_features+t_fea
            #     tag_count=tag_count+1
            # tag_features=tag_features/tag_count
            # print("tag")
            # print(tag_features.shape)

            # tag_knns_100=knn_search_C(tag_features.reshape((384,)),text_embeds2,30)

            # embeds10=[]
            # for u in range(50):
            #     embeds10.append(all_es[KNNs[tag_knns_100[u]]].reshape((1,-1)))
            # embeds20=np.concatenate(embeds10)
            # pic_knns_100=knn_search_C(el_features[0],embeds20,50)



            # embeds10=[]
            # for u in range(60):
            #     embeds10.append(all_es[KNNs[tag_knns_100[u]]].reshape((1,-1)))
            # embeds20=np.concatenate(embeds10)
            # start_fil=60
            # embeds_list=[embeds20]
            # if len(features_list)>1:

            #     for fi in range(len(features_list)):
            #         embeds_temp1=[]
            #         ems_knn=knn_search_C(features_list[fi][0],embeds_list[-1],start_fil)
            #         temp_tag_knns=tag_knns_100/2
            #         for fj in range(start_fil):
            #             temp_tag_knns[fj]=tag_knns_100[ems_knn[fj]]
            #         for fj in range(start_fil):
            #             tag_knns_100[fj]=temp_tag_knns[fj]
            #         embeds10=[]  
            #         start_fil=start_fil-10 
            #         for u in range(start_fil):
            #             embeds10.append(all_es[KNNs[tag_knns_100[u]]].reshape((1,-1)))
            #         embeds20=np.concatenate(embeds10) 
            #         embeds_list.append(embeds20)                

            # embeds10=[]
            # for u in range(start_fil):
            #     embeds10.append(all_es[KNNs[tag_knns_100[u]]].reshape((1,-1)))
            # embeds20=np.concatenate(embeds10)
            # ems_knn=knn_search_C(el_features[0],embeds20,start_fil)
            # temp_tag_knns=tag_knns_100/2
            # for fj in range(start_fil):
            #     temp_tag_knns[fj]=tag_knns_100[ems_knn[fj]]
            # for fj in range(start_fil):
            #     tag_knns_100[fj]=temp_tag_knns[fj]


            # for k in range(0,inital_scope-len(candidates)):
            #     im_paths0.append(str(all_ps[KNNs[k]]))
            #     intent_id_list0.append(0)
            embeds21=[]
            for k in range(0,inital_scope-len(candidates)):
                kick=0
                for kj in range(len(features_list)):
                    if element_sim_list[kj][0][k]<element_per_list[kj]:
                        kick=1
                if kick==0:
                    im_paths0.append(str(all_ps[KNNs[k]]))
                    embeds21.append(all_es[KNNs[k]].reshape((1,-1)))
                    intent_id_list0.append(0)
            print("len")
            print(len(im_paths0))
            embeds22=np.concatenate(embeds21)
            # for k in range(0,inital_scope-len(candidates)):
            #     if k<30:
            #         im_paths0.append(str(all_ps[KNNs[tag_knns_100[k]]]))
            #         # im_paths0.append(str(all_ps[KNNs[tag_knns_100[pic_knns_100[k]]]]))
            #         intent_id_list0.append(0)
            #     # elif k>=50 and k<150:
            #     #     im_paths0.append(str(all_ps[KNNs[tag_knns_100[k]]]))
            #     #     intent_id_list0.append(0)
            #     else:
            #         im_paths0.append(str(all_ps[KNNs[k]]))
            #         intent_id_list0.append(0)
            pointer=0
            count=0
            indexes=[]
            
            while count<inital_scope*0.2 and pointer<len(im_paths0):
                duplicate=0
                for j in range(len(im_paths1)):
                    if cosD(embeds22[pointer],embeds22[indexes[j]])<0.00001:#0.00001:
                        duplicate=1
                        break
                if duplicate==0:
                    im_paths1.append(im_paths0[pointer])
                    intent_id_list1.append(intent_id_list0[pointer])
                    indexes.append(pointer)
                    embeds3.append(embeds22[pointer].reshape((1,-1)))
                    count=count+1
                pointer=pointer+1
            
        else:
            im_paths0=candidates
            key_paths=[[] for t in poslist]
            unit_num=int(inital_scope/len(poslist))
            scope=len(poslist)*unit_num
            include_embeds=[]
            for g in range(len(poslist)):
                t=poslist[g]
                key_paths0=[]
                text_tokens = clip.tokenize(t).cuda()
                unit_num=int(inital_scope/len(poslist))
                scope=len(poslist)*unit_num
                with torch.no_grad():
                    text_features0 = model.encode_text(text_tokens).float()
                    text_features1= model2.encode_text(text_tokens).float()
                text_features0 /= text_features0.norm(dim=-1, keepdim=True)
                text_features0=text_features0.cpu().numpy()
                text_features1 /= text_features1.norm(dim=-1, keepdim=True)
                text_features1=text_features1.cpu().numpy()
                text_features=(w1*text_features0+w2*text_features1)/(w1+w2)
                
                KNNs=ball_search_C(text_features[0], scope)

                # print("all_es: ")
                # print(all_es.shape)
                embeds1=[]
                for u in range(scope):
                    embeds1.append(all_es[KNNs[u]].reshape((1,-1)))
                embeds2=np.concatenate(embeds1)
                for k in range(0,scope-len(candidates)):
                    key_paths0.append(str(all_ps[KNNs[k]]))

                pointer=0
                count=0
                indexes=[]
                incl_embed=[]
                while count<int(unit_num*0.3) and pointer<scope:
                    duplicate=0
                    for j in range(len(key_paths[g])):
                        if cosD(embeds2[pointer],embeds2[indexes[j]])<0.00001:
                            duplicate=1
                            break
                    if duplicate==0:
                        key_paths[g].append(key_paths0[pointer])
                        indexes.append(pointer)
                        incl_embed.append(embeds2[pointer].reshape((1,-1)))
                        count=count+1
                    pointer=pointer+1 
                include_embeds.append(incl_embed)
            for u in range(int(unit_num*0.28)):
                for r in range(len(key_paths)):
                    im_paths1.append(key_paths[r][u]) 
                    embeds3.append(include_embeds[r][u])
                    intent_id_list1.append(r) 
        print("im_paths1")
        print(im_paths1)
        embeds4=np.concatenate(embeds3)
        if len(neglist)!=0:
            if len(neglist)>1:
                mean_sim_list=[]
                min_sim_list=[]
                sim_list=[]
                for g in range(len(neglist)):
                    text_tokens1 = clip.tokenize(neglist[g]).cuda()
                    with torch.no_grad():
                        text_features10 = model.encode_text(text_tokens1).float()
                        text_features11= model2.encode_text(text_tokens1).float()
                    text_features10 /= text_features10.norm(dim=-1, keepdim=True)
                    text_features10=text_features10.cpu().numpy()
                    text_features11 /= text_features11.norm(dim=-1, keepdim=True)
                    text_features11=text_features11.cpu().numpy()
                    text_features1=(w1*text_features10+w2*text_features11)/(w1+w2)
                    # similarity=text_features1 @ embeds2.T
                    similarity=text_features1 @ embeds4.T
                    mean_sim=np.mean(similarity)
                    min_sim=np.min(similarity)
                    mean_sim_list.append(mean_sim)
                    min_sim_list.append(min_sim)
                    sim_list.append(similarity)
                for u in range(len(im_paths1)):
                    kick=0
                    for k in range(len(neglist)):
                        if sim_list[k][0][u]>(min_sim_list[k]+30*mean_sim_list[k])/31:
                            kick=1
                    if kick==0:
                        im_paths2.append(im_paths1[u])
                        intent_id_list2.append(intent_id_list1[u])
            else:
                text_tokens1 = clip.tokenize(neglist[0]).cuda()
                with torch.no_grad():
                    text_features10 = model.encode_text(text_tokens1).float()
                    text_features11= model2.encode_text(text_tokens1).float()
                text_features10 /= text_features10.norm(dim=-1, keepdim=True)
                text_features10=text_features10.cpu().numpy()
                text_features11 /= text_features11.norm(dim=-1, keepdim=True)
                text_features11=text_features11.cpu().numpy()
                text_features1=(w1*text_features10+w2*text_features11)/(w1+w2)
                # similarity=text_features1 @ embeds2.T
                similarity=text_features1 @ embeds4.T
                mean_sim=np.mean(similarity)
                min_sim=np.min(similarity)
                median=np.percentile(similarity,30)
                temp_ps2=np.argsort(similarity[0])
                # dtype=[('pat')]
                print("im_paths1_len")
                print(len(im_paths1))
                print("similarity_len")
                print(similarity.shape)
                print(temp_ps2)
                for u in range(len(im_paths1)):
                    # if similarity[0][u]<=median:
                    # if similarity[0][u]<=(16*mean_sim+15*min_sim)/31:
                    # if similarity[0][u]<=(30*mean_sim+min_sim)/31:
                        # temp_ps2.append[{}]
                        # im_paths2.append(im_paths1[u])
                        # intent_id_list2.append(intent_id_list1[u])
                    im_paths2.append(im_paths1[temp_ps2[u]])
                    intent_id_list2.append(intent_id_list1[u])
        else:
            im_paths2=im_paths1
            intent_id_list2=intent_id_list1
    print("im_paths2")
    print(im_paths2)
    # if "igh" in price_adj:
    #     priceRange[0]=2
    # if "ow" in price_adj:
    #     priceRange[1]=0.5  
    price_cans=[]  
    if priceRange[0]!=-1 or priceRange[1]<50:
        for i in range(len(im_paths2)):
            s=im_paths2[i].split('/')[-1]
            token_id=''.join([i for i in s if i.isdecimal()])
            if "Doodle" in im_paths2[i]:
                try:
                    price_c=float(PriceDic["Doodles"+str(int(token_id))])#float(re.findall(r"\d+\.?\d*",bestOfferDic["Doodles"+str(int(token_id))])[0])
                except:
                    price_c=0
            elif "CoolCat" in im_paths2[i]:
                try:
                    price_c=float(PriceDic["CoolCat"+str(int(token_id))])#float(re.findall(r"\d+\.?\d*",bestOfferDic["CoolCat"+str(int(token_id))])[0])
                except:
                    price_c=0
            else:
                #bestOffer[i]=0
                # price_c=int(5*np.random.rand()*1000)/1000
                try:
                    na=im_paths2[i].split("/")[3]
                    price_c=float(PriceDic[na+str(int(token_id))])
                except:
                    price_c=int(5*np.random.rand()*1000)/1000
            if  price_c>priceRange[0] and price_c<priceRange[1]:
                im_paths.append(im_paths2[i])
                intent_id_list.append(intent_id_list2[i])
                price_cans.append(price_c)
    else:
        for i in range(len(im_paths2)):
            s=im_paths2[i].split('/')[-1]
            token_id=''.join([i for i in s if i.isdecimal()])
            if "Doodle" in im_paths2[i]:
                try:
                    price_c=float(PriceDic["Doodles"+str(int(token_id))])#float(re.findall(r"\d+\.?\d*",bestOfferDic["Doodles"+str(int(token_id))])[0])
                except:
                    price_c=0
                    # print(bestOfferDic["Doodles"+str(int(token_id))])
                
            elif "CoolCat" in im_paths2[i]:
                try:
                    price_c=float(PriceDic["CoolCat"+str(int(token_id))])#float(re.findall(r"\d+\.?\d*",bestOfferDic["CoolCat"+str(int(token_id))])[0])
                except:
                    price_c=0
                    # print(bestOfferDic["CoolCat"+str(int(token_id))])
            else:
                #bestOffer[i]=0
                try:
                    na=im_paths2[i].split("/")[3]
                    price_c=float(PriceDic[na+str(int(token_id))])
                except:
                    price_c=int(5*np.random.rand()*1000)/1000
            price_cans.append(price_c)
        im_paths=im_paths2
        intent_id_list=intent_id_list2
    # else:
    limit=min(40,len(price_cans))
    if "ow" in price_adj or "ch" in price_adj:
        price_order=np.argsort(price_cans[:limit])
        t_paths=[]
        intent_id_paths=[]
        t_prices=[]
        for idx in price_order:
            t_paths.append(im_paths[idx])
            intent_id_paths.append(intent_id_list[idx])
            t_prices.append(price_cans[idx])
        for q in range(limit):
            im_paths[q]=t_paths[q]
            intent_id_list[q]=intent_id_paths[q]
            price_cans[q]=t_prices[q]
        #priceRange[0]=2
    if "igh" in price_adj or "ex" in price_adj:
        price_order=np.argsort(price_cans[:limit])
        t_paths=[]
        intent_id_paths=[]
        t_prices=[]
        for q in range(limit):
            idx=price_order[limit-1-q]
            t_paths.append(im_paths[idx])
            intent_id_paths.append(intent_id_list[idx])
            t_prices.append(price_cans[idx])
        for q in range(limit):
            im_paths[q]=t_paths[q]
            intent_id_list[q]=intent_id_paths[q]
            price_cans[q]=t_prices[q]
    lens[0]=len(im_paths)
    # print(im_paths)
    for i in range(len(im_paths)):
        va[i]=str(im_paths[i])
        s=im_paths[i].split('/')[-1]
        token_id=''.join([i for i in s if i.isdecimal()])
        bestOffer[i]=str(price_cans[i])+" WETH"
        # if "Doodle" in im_paths[i]:
        #     bestOffer[i]=bestOfferDic["Doodles"+str(int(token_id))]
        # elif "CoolCat" in im_paths[i]:
        #     bestOffer[i]=bestOfferDic["CoolCat"+str(int(token_id))]
        # else:
        #     #bestOffer[i]=0
        #     bestOffer[i]=str(int(5*np.random.rand()*1000)/1000)+" WETH"
        
        colname=im_paths[i].split("/")[3]
        addresses.append(address_dic[colname])
    previous_intent_ids[0]=intent_id_list
    previous_paths[0]=im_paths
    print(intent_id_list)    
    os.remove("static/query.png")
    im.save("static/query.png")

    return jsonify({"error": 1001, "msg": "上传失败"})

@app.route('/api/test/reviseText/',methods=['POST'])
def reviseText():
    f = request.form["text"]
    print("annotation: ")
    # print(f)
    annotations=f.split("|")
    with open("static/annotations3.csv","a",newline='') as file:
        writer=csv.writer(file)
        print(TextQuery)
        row=[TextQuery[-1]]
        for anno in annotations:
            if len(anno)>1:
                dual=anno.split(":")
                row.append(dual[0])
                row.append(dual[-1])
        writer.writerow(row)

    return jsonify({"error": 1001, "msg": "上传失败"})

@app.route('/api/test/getGenerate/',methods=['GET','POST'])
def getGenerate():
	# 通过表单中name值获取图片
    start_time=time.time()
    print("hello-hello")
    # f_d_data = request.get_json(silent=True)
    # print(f_d_data)
    # f = f_d_data['data']['image']
    imf = request.files["image"]
    txtf = request.form["text"]
    imgData=imf.read()
    byte_stream = io.BytesIO(imgData)  
    # im = Image.open(byte_stream) 
    png = Image.open(byte_stream).convert('RGBA')
    background = Image.new('RGBA', png.size, (255,255,255))

    init_image = Image.alpha_composite(background, png).convert("RGB")
    init_image.thumbnail((768, 768))
    t=time.localtime()
    t_suffix=str(t.tm_year)+"_"+str(t.tm_mon)+"_"+str(t.tm_mday)+"_"+str(t.tm_hour)+"_"+str(t.tm_min)+"_"+str(t.tm_sec)
    init_image.save("canvas"+t_suffix+".png")
    im_array=np.array(init_image)
    genstate[0]=0
    if np.mean(im_array)==255:
        image = pipe1(txtf).images[0]      
        image.save("generate"+t_suffix+".png")
        image.save("D:/NFTSearch/NFTSearch/Frontend/src/assets/"+"generate"+t_suffix+".png")
    else:
        images = pipe(prompt=txtf, image=init_image, strength=0.85, guidance_scale=5.5).images
        images[0].save("generate"+t_suffix+".png")
        images[0].save("D:/NFTSearch/NFTSearch/Frontend/src/assets/"+"generate"+t_suffix+".png")
    genstate[0]=1
    print("Prompt:")
    print(txtf)
    return jsonify({"error": 1001, "msg": "上传失败","address": "generate"+t_suffix+".png"})

@app.route('/api/test/getInstruct/',methods=['GET','POST'])
def getInstruct():
	# 通过表单中name值获取图片
    start_time=time.time()
    t=time.localtime()
    print("hello-instruct")
    im_p=request.form["im_p"]
    c_text=request.form["composed"]
    col=im_p.split("/")[-2].replace("%20"," ")
    f_name=im_p.split("/")[-1]
    IMAGE_PATH="D:/Limited_col1/Limited_col1/"+col+"/"+col+"/"+f_name
    image=PIL.Image.open(IMAGE_PATH)
    image=image.convert("RGB")
    print(image.size)
    results = pipe(c_text, image=image, num_inference_steps=20, image_guidance_scale=1.5, guidance_scale=7).images
    print(results[0].size)
    try:
        os.remove("static/instruct.png")
    except:
        cod=1
    results[0].save("static/instruct.png")
    image_bgr = cv2.imread("static/instruct.png")
    origin_bgr=cv2.imread(current_seg_path[0])
    mask0=mask_list[0]
    if mask0.shape[0]==image_bgr.shape[0] and mask0.shape[1]==image_bgr.shape[1]:
        mask=mask0
    else:
        mask=mask0[:image_bgr.shape[0],:image_bgr.shape[1]]
        origin_bgr=origin_bgr[:image_bgr.shape[0],:image_bgr.shape[1]]
    inv_mask=np.logical_not(mask)
    sub=image_bgr*mask
    bac=origin_bgr*inv_mask
    edit_im=sub+bac
    mask_obj=edit_im*mask
    combine = cv2.addWeighted(mask_obj,0.9,edit_im,0.1,0)
    t_suffix=str(t.tm_year)+"_"+str(t.tm_mon)+"_"+str(t.tm_mday)+"_"+str(t.tm_hour)+"_"+str(t.tm_min)+"_"+str(t.tm_sec)
    fn="D:/NFTSearch/NFTSearch/Frontend/src/assets/static/temp_ins/"+t_suffix+"instruct.png"
    fn1="D:/NFTSearch/NFTSearch/Frontend/src/assets/static/temp_ins/"+t_suffix+"instruct_obj.png"
    fn2="D:/NFTSearch/NFTSearch/Frontend/src/assets/static/temp_ins/"+t_suffix+"instruct_combine.png"
    # results[0].save(fn)
    cv2.imwrite(fn, edit_im)
    cv2.imwrite(fn1, mask_obj)
    cv2.imwrite(fn2, combine)
    instruct_state[0]=1
    return jsonify({"error": 1001, "msg": "上传失败", "address": t_suffix+"instruct.png", "obj_address": t_suffix+"instruct_obj.png"})

@app.route('/api/test/getInstructSeg/',methods=['GET','POST'])
def getInstructSeg():
	# 通过表单中name值获取图片
    start_time=time.time()

    # f = request.files["image"]
    # imgData=f.read()
    # byte_stream = io.BytesIO(imgData)  
    seg_path0=request.form["ins_seg_path"]
    # seg_path1=request.form["ins_seg_path"]
    # composed=request.form["composed"]
    seg_path="D:/NFTSearch/NFTSearch/Frontend/src/assets/static/temp_ins/"+seg_path0
    im=Image.open(seg_path)
    # im = Image.open(byte_stream) 
    width, height = im.size
    imgs=[preprocess(im)]
    image_input = torch.tensor(np.stack(imgs)).cuda()
    with torch.no_grad():
        image_features0 = model.encode_image(image_input).float()
        image_features1 = model2.encode_image(image_input).float()
    image_features0 /= image_features0.norm(dim=-1, keepdim=True)
    im_features0=image_features0.cpu().numpy()
    image_features1 /= image_features1.norm(dim=-1, keepdim=True)
    im_features1=image_features1.cpu().numpy()
    im_features=(w1*im_features0+w2*im_features1)/(w1+w2)

    im1 = Image.open("D:/NFTSearch/NFTSearch/Frontend/src/assets/static/temp_ins/"+seg_path0[:-4]+"_combine"+".png") 
    width, height = im1.size
    imgs1=[preprocess(im1)]
    image_input1 = torch.tensor(np.stack(imgs1)).cuda()
    with torch.no_grad():
        image_features01 = model.encode_image(image_input1).float()
        image_features11 = model2.encode_image(image_input1).float()
    image_features01 /= image_features01.norm(dim=-1, keepdim=True)
    im_features01=image_features01.cpu().numpy()
    image_features11 /= image_features11.norm(dim=-1, keepdim=True)
    im_features11=image_features11.cpu().numpy()
    im_features1=(w1*im_features01+w2*im_features11)/(w1+w2)
    # im_features2=(4*im_features+3*im_features1)/7
    im_features2=(3*im_features+7*im_features1)/10

    print(im_features.shape)
    # img=im_features[0]
    img=im_features2[0]

    KNNs=ball_search_C(img, 100)
    price_cans=[]
    im_paths0=[]
    im_paths=[]
    for i in range(100):
        im_paths0.append(str(all_ps[KNNs[i]]))
    # if composed!="":
    #     embeds1=[]
    #     for u in range(100):
    #         embeds1.append(all_es[KNNs[u]].reshape((1,-1)))
    #     embeds2=np.concatenate(embeds1)
    #     text_tokens1 = clip.tokenize(composed).cuda()
    #     with torch.no_grad():
    #         text_features10 = model.encode_text(text_tokens1).float()
    #         text_features11= model2.encode_text(text_tokens1).float()
    #     text_features10 /= text_features10.norm(dim=-1, keepdim=True)
    #     text_features10=text_features10.cpu().numpy()
    #     text_features11 /= text_features11.norm(dim=-1, keepdim=True)
    #     text_features11=text_features11.cpu().numpy()
    #     text_features1=(w1*text_features10+w2*text_features11)/(w1+w2)
    #     knns1=knn_search_C(text_features1.reshape((1024,)),embeds2,100)
    #     for i in range(100):
    #         im_paths.append(im_paths0[knns1[i]])
    #     # similarity=text_features1 @ embeds4.T
    # else:
    #     for i in range(100):
    #         im_paths.append(im_paths0[i])
    for i in range(100):
        im_paths.append(im_paths0[i])
    for i in range(len(im_paths)):
        s=im_paths[i].split('/')[-1]
        token_id=''.join([i for i in s if i.isdecimal()])
        if "Doodle" in im_paths[i]:
            try:
                price_c=float(PriceDic["Doodles"+str(int(token_id))])#float(re.findall(r"\d+\.?\d*",bestOfferDic["Doodles"+str(int(token_id))])[0])
            except:
                price_c=0
                # print(bestOfferDic["Doodles"+str(int(token_id))])
            
        elif "CoolCat" in im_paths[i]:
            try:
                price_c=float(PriceDic["CoolCat"+str(int(token_id))])#float(re.findall(r"\d+\.?\d*",bestOfferDic["CoolCat"+str(int(token_id))])[0])
            except:
                price_c=0
                # print(bestOfferDic["CoolCat"+str(int(token_id))])
        else:
            #bestOffer[i]=0
            try:
                na=im_paths[i].split("/")[3]
                price_c=float(PriceDic[na+str(int(token_id))])
            except:
                price_c=int(5*np.random.rand()*1000)/1000
        price_cans.append(price_c)
    for i in range(100):
        va[i]=str(im_paths[i])
        s=im_paths[i].split('/')[-1]
        token_id=''.join([i for i in s if i.isdecimal()])
        bestOffer[i]=str(price_cans[i])+" WETH"
        colname=im_paths[i].split("/")[3]
        addresses.append(address_dic[colname])
   
    end_time=time.time()
    print("Time per query: ")
    print(end_time-start_time)
    # for i in range(11):
    #     va[i]=str((2*pre[i]+pre1[i])/3)
        #va[i]=str(pre1[i])
    os.remove("static/query.png")
    im.save("static/query.png")
    mask_state[0]=0
    mask_list.remove(mask_list[0])
    instruct_state[0]=0
    return jsonify({"error": 1001, "msg": "上传失败"})

@app.route('/api/test/getSam/',methods=['GET','POST'])
def getSam():
    t=time.localtime()
    up_x= float(request.form["up_x"])
    up_y= float(request.form["up_y"])
    down_x= float(request.form["down_x"])
    down_y= float(request.form["down_y"])  
    im_p=request.form["im_p"]
    print("down_x")
    print(down_x) 
    print("up_x")
    print(up_x) 
    print("im_p")
    print(im_p)
    col=im_p.split("/")[-2].replace("%20"," ")
    f_name=im_p.split("/")[-1]
    IMAGE_PATH="D:/Limited_col1/Limited_col1/"+col+"/"+col+"/"+f_name
    

    image_bgr = cv2.imread(IMAGE_PATH)
    im_w=image_bgr.shape[1]
    im_h=image_bgr.shape[0]
    x1=int(up_x*im_w/300.0)
    y1=int(up_y*im_h/300.0)
    x2=int(down_x*im_w/300.0)
    y2=int(down_y*im_h/300.0)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask_predictor.set_image(image_rgb)

    box = np.array([x2, y2, x1, y1])
    print("shape")
    print(image_rgb.shape)
    print("box")
    print(box)
    masks, scores, logits = mask_predictor.predict(
        box=box,
        multimask_output=True
    )
    color = np.array([30/255, 144/255, 255/255, 0.6])
    mask_id=np.argmax(scores)
    print(scores)
    mask=masks[mask_id].reshape((masks.shape[1],masks.shape[2],1))
    if mask_state[0]==0:
        mask_list.append(mask)
        mask_image = mask.reshape(im_h, im_w, 1) * color.reshape(1, 1, -1)
        mask_im=image_bgr*mask
        
        combine = cv2.addWeighted(mask_im,0.9,image_bgr,0.1,0)
        combine_v = cv2.addWeighted(mask_im,0.7,image_bgr,0.3,0)
        t_suffix=str(t.tm_year)+"_"+str(t.tm_mon)+"_"+str(t.tm_mday)+"_"+str(t.tm_hour)+"_"+str(t.tm_min)+"_"+str(t.tm_sec)
        # cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+".png", mask_im)
        white=255*np.ones(image_rgb.shape)
        black=0*np.ones(image_rgb.shape)
        bl_mask=white*mask
        inv_mask=np.logical_not(mask)
        background=white*inv_mask
        hole_im=image_bgr*inv_mask
        combine1=mask_im+background
        combine_de = cv2.addWeighted(hole_im,0.7,image_bgr,0.3,0)
        # cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+".png", combine)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+".png", combine)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+"white"+".png", combine1)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+"hole"+".png", hole_im)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+"comb_v"+".png", combine_v)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+"black_mask"+".png", bl_mask)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+"original"+".png", image_bgr)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+"del"+".png", combine_de)
        current_seg_path[0]="D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+"original"+".png"
    else:
        mask0=mask_list[0]
        mask1=np.logical_or(mask0,mask)
        mask_list[0]=mask1
        mask_image = mask1.reshape(im_h, im_w, 1) * color.reshape(1, 1, -1)
        mask_im=image_bgr*mask1
        
        combine = cv2.addWeighted(mask_im,0.9,image_bgr,0.1,0)
        combine_v = cv2.addWeighted(mask_im,0.7,image_bgr,0.3,0)
        t_suffix=str(t.tm_year)+"_"+str(t.tm_mon)+"_"+str(t.tm_mday)+"_"+str(t.tm_hour)+"_"+str(t.tm_min)+"_"+str(t.tm_sec)
        # cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+".png", mask_im)
        white=255*np.ones(image_rgb.shape)
        black=0*np.ones(image_rgb.shape)
        bl_mask=white*mask1
        inv_mask=np.logical_not(mask1)
        background=white*inv_mask
        hole_im=image_bgr*inv_mask
        combine1=mask_im+background
        combine_de = cv2.addWeighted(hole_im,0.7,image_bgr,0.3,0)
        # cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+".png", combine)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+".png", combine)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+"white"+".png", combine1)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+"hole"+".png", hole_im)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+"comb_v"+".png", combine_v)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+"black_mask"+".png", bl_mask)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+"original"+".png", image_bgr)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+"del"+".png", combine_de)
        current_seg_path[0]="D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+"original"+".png"
    im = Image.open("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+".png") 
    width, height = im.size
    imgs=[preprocess(im)]
    image_input = torch.tensor(np.stack(imgs)).cuda()
    with torch.no_grad():
        image_features0 = model.encode_image(image_input).float()
        image_features1 = model2.encode_image(image_input).float()
    image_features0 /= image_features0.norm(dim=-1, keepdim=True)
    im_features0=image_features0.cpu().numpy()
    image_features1 /= image_features1.norm(dim=-1, keepdim=True)
    im_features1=image_features1.cpu().numpy()
    im_features=(w1*im_features0+w2*im_features1)/(w1+w2)
    # im1 = Image.open("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+".png") 
    # width, height = im1.size
    # imgs1=[preprocess(im1)]
    # image_input1 = torch.tensor(np.stack(imgs1)).cuda()
    # with torch.no_grad():
    #     image_features01 = model.encode_image(image_input1).float()
    #     image_features11 = model2.encode_image(image_input1).float()
    # image_features01 /= image_features01.norm(dim=-1, keepdim=True)
    # im_features01=image_features01.cpu().numpy()
    # image_features11 /= image_features11.norm(dim=-1, keepdim=True)
    # im_features11=image_features11.cpu().numpy()
    # im_features1=(5*im_features01+2*im_features11)/7
    # im_features2=(4*im_features+3*im_features1)/7
    trait_KNNs=trait_ball_search_C(im_features[0], 5)
    related_traits=[]
    for i in range(5):
        related_traits.append(All_traits[trait_KNNs[i]])
        print(All_traits[trait_KNNs[i]])
    # All_traits[i]
    mask_state[0]=1
    return jsonify({"error": 1001, "msg": "上传失败","address": "sam"+t_suffix+".png"})


@app.route('/api/test/getImgSeg/',methods=['GET','POST'])
def getImgSeg():
	# 通过表单中name值获取图片
    start_time=time.time()

    # f = request.files["image"]
    # imgData=f.read()
    # byte_stream = io.BytesIO(imgData)  
    seg_path0=request.form["seg_path"]
    composed=request.form["composed"]
    seg_path="D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+seg_path0
    im=Image.open(seg_path)
    # im = Image.open(byte_stream) 
    width, height = im.size
    imgs=[preprocess(im)]
    image_input = torch.tensor(np.stack(imgs)).cuda()
    with torch.no_grad():
        image_features0 = model.encode_image(image_input).float()
        image_features1 = model2.encode_image(image_input).float()
    image_features0 /= image_features0.norm(dim=-1, keepdim=True)
    im_features0=image_features0.cpu().numpy()
    image_features1 /= image_features1.norm(dim=-1, keepdim=True)
    im_features1=image_features1.cpu().numpy()
    im_features=(w1*im_features0+w2*im_features1)/(w1+w2)

    im1 = Image.open("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+seg_path0[:-4]+"white"+".png") 
    width, height = im1.size
    imgs1=[preprocess(im1)]
    image_input1 = torch.tensor(np.stack(imgs1)).cuda()
    with torch.no_grad():
        image_features01 = model.encode_image(image_input1).float()
        image_features11 = model2.encode_image(image_input1).float()
    image_features01 /= image_features01.norm(dim=-1, keepdim=True)
    im_features01=image_features01.cpu().numpy()
    image_features11 /= image_features11.norm(dim=-1, keepdim=True)
    im_features11=image_features11.cpu().numpy()
    im_features1=(w1*im_features01+w2*im_features11)/(w1+w2)
    # im_features2=(4*im_features+3*im_features1)/7
    im_features2=(7*im_features+3*im_features1)/10

    print(im_features.shape)
    # img=im_features[0]
    img=im_features2[0]

    KNNs=ball_search_C(img, 100)
    price_cans=[]
    im_paths0=[]
    im_paths=[]
    for i in range(100):
        im_paths0.append(str(all_ps[KNNs[i]]))
    if composed!="":
        embeds1=[]
        for u in range(100):
            embeds1.append(all_es[KNNs[u]].reshape((1,-1)))
        embeds2=np.concatenate(embeds1)
        text_tokens1 = clip.tokenize(composed).cuda()
        with torch.no_grad():
            text_features10 = model.encode_text(text_tokens1).float()
            text_features11= model2.encode_text(text_tokens1).float()
        text_features10 /= text_features10.norm(dim=-1, keepdim=True)
        text_features10=text_features10.cpu().numpy()
        text_features11 /= text_features11.norm(dim=-1, keepdim=True)
        text_features11=text_features11.cpu().numpy()
        text_features1=(w1*text_features10+w2*text_features11)/(w1+w2)
        knns1=knn_search_C(text_features1.reshape((1024,)),embeds2,100)
        for i in range(100):
            im_paths.append(im_paths0[knns1[i]])
        # similarity=text_features1 @ embeds4.T
    else:
        for i in range(100):
            im_paths.append(im_paths0[i])
    for i in range(len(im_paths)):
        s=im_paths[i].split('/')[-1]
        token_id=''.join([i for i in s if i.isdecimal()])
        if "Doodle" in im_paths[i]:
            try:
                price_c=float(PriceDic["Doodles"+str(int(token_id))])#float(re.findall(r"\d+\.?\d*",bestOfferDic["Doodles"+str(int(token_id))])[0])
            except:
                price_c=0
                # print(bestOfferDic["Doodles"+str(int(token_id))])
            
        elif "CoolCat" in im_paths[i]:
            try:
                price_c=float(PriceDic["CoolCat"+str(int(token_id))])#float(re.findall(r"\d+\.?\d*",bestOfferDic["CoolCat"+str(int(token_id))])[0])
            except:
                price_c=0
                # print(bestOfferDic["CoolCat"+str(int(token_id))])
        else:
            #bestOffer[i]=0
            try:
                na=im_paths[i].split("/")[3]
                price_c=float(PriceDic[na+str(int(token_id))])
            except:
                price_c=int(5*np.random.rand()*1000)/1000
        price_cans.append(price_c)
    for i in range(100):
        va[i]=str(im_paths[i])
        s=im_paths[i].split('/')[-1]
        token_id=''.join([i for i in s if i.isdecimal()])
        bestOffer[i]=str(price_cans[i])+" WETH"
        colname=im_paths[i].split("/")[3]
        addresses.append(address_dic[colname])
   
    end_time=time.time()
    print("Time per query: ")
    print(end_time-start_time)
    # for i in range(11):
    #     va[i]=str((2*pre[i]+pre1[i])/3)
        #va[i]=str(pre1[i])
    os.remove("static/query.png")
    im.save("static/query.png")
    mask_state[0]=0
    mask_list.remove(mask_list[0])
    instruct_state[0]=0
    return jsonify({"error": 1001, "msg": "上传失败"})


@app.route('/api/test/getImgSegNeg/',methods=['GET','POST'])
def getImgSegNeg():

    start_time=time.time()

    # f = request.files["image"]
    # imgData=f.read()
    # byte_stream = io.BytesIO(imgData)  
    seg_path0=request.form["seg_path"]
    composed=request.form["composed"]
    seg_path="D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+seg_path0
    im=Image.open(seg_path)
    # im = Image.open(byte_stream) 
    width, height = im.size
    imgs=[preprocess(im)]
    image_input = torch.tensor(np.stack(imgs)).cuda()
    with torch.no_grad():
        image_features0 = model.encode_image(image_input).float()
        image_features1 = model2.encode_image(image_input).float()
    image_features0 /= image_features0.norm(dim=-1, keepdim=True)
    im_features0=image_features0.cpu().numpy()
    image_features1 /= image_features1.norm(dim=-1, keepdim=True)
    im_features1=image_features1.cpu().numpy()
    im_features=(w1*im_features0+w2*im_features1)/(w1+w2)

    im1 = Image.open("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+seg_path0[:-4]+"hole"+".png") 
    width, height = im1.size
    imgs1=[preprocess(im1)]
    image_input1 = torch.tensor(np.stack(imgs1)).cuda()
    with torch.no_grad():
        image_features01 = model.encode_image(image_input1).float()
        image_features11 = model2.encode_image(image_input1).float()
    image_features01 /= image_features01.norm(dim=-1, keepdim=True)
    im_features01=image_features01.cpu().numpy()
    image_features11 /= image_features11.norm(dim=-1, keepdim=True)
    im_features11=image_features11.cpu().numpy()
    im_features1=(w1*im_features01+w2*im_features11)/(w1+w2)
    # im_features2=(4*im_features+3*im_features1)/7
    # im_features2=(7*im_features+3*im_features1)/10


    img=im_features1[0]


    KNNs=ball_search_C(img, 400)
    price_cans=[]
    im_paths0=[]
    im_paths=[]
    for i in range(400):
        im_paths0.append(str(all_ps[KNNs[i]]))
    embeds1=[]
    for u in range(400):
        embeds1.append(all_es[KNNs[u]].reshape((1,-1)))
    embeds5=np.concatenate(embeds1)

    similarity=im_features @ embeds5.T
    mean_sim=np.mean(similarity)
    min_sim=np.min(similarity)
    median=np.percentile(similarity,30)
    temp_ps2=np.argsort(similarity[0])
    # dtype=[('pat')]

    print("similarity_len")
    print(similarity.shape)

    for u in range(len(im_paths0)):
        # if similarity[0][u]<=median:
        # if similarity[0][u]<=(16*mean_sim+15*min_sim)/31:
        # if similarity[0][u]<=(30*mean_sim+min_sim)/31:
            # temp_ps2.append[{}]
            # im_paths2.append(im_paths1[u])
            # intent_id_list2.append(intent_id_list1[u])
        im_paths.append(im_paths0[temp_ps2[u]])
        # intent_id_list2.append(intent_id_list1[u])

    for i in range(len(im_paths)):
        s=im_paths[i].split('/')[-1]
        token_id=''.join([i for i in s if i.isdecimal()])
        if "Doodle" in im_paths[i]:
            try:
                price_c=float(PriceDic["Doodles"+str(int(token_id))])#float(re.findall(r"\d+\.?\d*",bestOfferDic["Doodles"+str(int(token_id))])[0])
            except:
                price_c=0
                # print(bestOfferDic["Doodles"+str(int(token_id))])
            
        elif "CoolCat" in im_paths[i]:
            try:
                price_c=float(PriceDic["CoolCat"+str(int(token_id))])#float(re.findall(r"\d+\.?\d*",bestOfferDic["CoolCat"+str(int(token_id))])[0])
            except:
                price_c=0
                # print(bestOfferDic["CoolCat"+str(int(token_id))])
        else:
            #bestOffer[i]=0
            try:
                na=im_paths[i].split("/")[3]
                price_c=float(PriceDic[na+str(int(token_id))])
            except:
                price_c=int(5*np.random.rand()*1000)/1000
        price_cans.append(price_c)
    for i in range(100):
        va[i]=str(im_paths[i])
        s=im_paths[i].split('/')[-1]
        token_id=''.join([i for i in s if i.isdecimal()])
        bestOffer[i]=str(price_cans[i])+" WETH"
        colname=im_paths[i].split("/")[3]
        addresses.append(address_dic[colname])
   
    end_time=time.time()
    print("Time per query: ")
    print(end_time-start_time)
    # for i in range(11):
    #     va[i]=str((2*pre[i]+pre1[i])/3)
        #va[i]=str(pre1[i])
    os.remove("static/query.png")
    im.save("static/query.png")

    mask_state[0]=0
    mask_list.remove(mask_list[0])
    instruct_state[0]=0
    return jsonify({"error": 1001, "msg": "上传失败"})


@app.route('/api/test/getSamClear/',methods=['GET','POST'])
def getSamClear():
    mask_state[0]=0
    mask_list.remove(mask_list[0])
    return jsonify({"error": 1001, "msg": "上传失败"})


@app.route('/api/test/getSamNew/',methods=['GET','POST'])
def getSamNew():
    t=time.localtime()
    f = request.files["image"]
    imgData=f.read()
    byte_stream = io.BytesIO(imgData)  
    im = Image.open(byte_stream) 
    temp_mark="D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+str(t)+".png"
    im.save(temp_mark)
    up_x= float(request.form["up_x"])
    up_y= float(request.form["up_y"])
    down_x= float(request.form["down_x"])
    down_y= float(request.form["down_y"])  
    # im_p=request.form["im_p"]
    print("down_x")
    print(down_x) 
    print("up_x")
    print(up_x) 
    # print("im_p")
    # print(im_p)
    # col=im_p.split("/")[-2].replace("%20"," ")
    # f_name=im_p.split("/")[-1]
    # IMAGE_PATH="D:/Limited_col1/Limited_col1/"+col+"/"+col+"/"+f_name
    

    # image_bgr = cv2.imread(IMAGE_PATH)
    # image_bgr = cv2.imread(byte_stream)
    image_bgr = cv2.imread(temp_mark)
    im_w=image_bgr.shape[1]
    im_h=image_bgr.shape[0]
    x1=int(up_x*im_w/300.0)
    y1=int(up_y*im_h/300.0)
    x2=int(down_x*im_w/300.0)
    y2=int(down_y*im_h/300.0)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask_predictor.set_image(image_rgb)

    box = np.array([x2, y2, x1, y1])
    print("shape")
    print(image_rgb.shape)
    print("box")
    print(box)
    masks, scores, logits = mask_predictor.predict(
        box=box,
        multimask_output=True
    )
    color = np.array([30/255, 144/255, 255/255, 0.6])
    mask_id=np.argmax(scores)
    print(scores)
    mask=masks[mask_id].reshape((masks.shape[1],masks.shape[2],1))
    if new_mask_state[0]==0:
        new_mask_list.append(mask)
        mask_image = mask.reshape(im_h, im_w, 1) * color.reshape(1, 1, -1)
        mask_im=image_bgr*mask
        
        combine = cv2.addWeighted(mask_im,0.9,image_bgr,0.1,0)
        combine_v = cv2.addWeighted(mask_im,0.7,image_bgr,0.3,0)
        t_suffix=str(t.tm_year)+"_"+str(t.tm_mon)+"_"+str(t.tm_mday)+"_"+str(t.tm_hour)+"_"+str(t.tm_min)+"_"+str(t.tm_sec)
        # cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+".png", mask_im)
        white=255*np.ones(image_rgb.shape)
        black=0*np.ones(image_rgb.shape)
        bl_mask=white*mask
        inv_mask=np.logical_not(mask)
        background=white*inv_mask
        hole_im=image_bgr*inv_mask
        combine1=mask_im+background
        combine_de = cv2.addWeighted(hole_im,0.7,image_bgr,0.3,0)
        # cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+".png", combine)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"samNew"+t_suffix+".png", combine)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"samNew"+t_suffix+"white"+".png", combine1)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"samNew"+t_suffix+"hole"+".png", hole_im)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"samNew"+t_suffix+"comb_v"+".png", combine_v)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"samNew"+t_suffix+"black_mask"+".png", bl_mask)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"samNew"+t_suffix+"original"+".png", image_bgr)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"samNew"+t_suffix+"del"+".png", combine_de)
        new_current_seg_path[0]="D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"samNew"+t_suffix+"original"+".png"
    else:
        mask0=new_mask_list[0]
        mask1=np.logical_or(mask0,mask)
        new_mask_list[0]=mask1
        mask_image = mask1.reshape(im_h, im_w, 1) * color.reshape(1, 1, -1)
        mask_im=image_bgr*mask1
        
        combine = cv2.addWeighted(mask_im,0.9,image_bgr,0.1,0)
        combine_v = cv2.addWeighted(mask_im,0.7,image_bgr,0.3,0)
        t_suffix=str(t.tm_year)+"_"+str(t.tm_mon)+"_"+str(t.tm_mday)+"_"+str(t.tm_hour)+"_"+str(t.tm_min)+"_"+str(t.tm_sec)
        # cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+".png", mask_im)
        white=255*np.ones(image_rgb.shape)
        black=0*np.ones(image_rgb.shape)
        bl_mask=white*mask1
        inv_mask=np.logical_not(mask1)
        background=white*inv_mask
        hole_im=image_bgr*inv_mask
        combine1=mask_im+background
        combine_de = cv2.addWeighted(hole_im,0.7,image_bgr,0.3,0)
        # cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"sam"+t_suffix+".png", combine)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"samNew"+t_suffix+".png", combine)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"samNew"+t_suffix+"white"+".png", combine1)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"samNew"+t_suffix+"hole"+".png", hole_im)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"samNew"+t_suffix+"comb_v"+".png", combine_v)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"samNew"+t_suffix+"black_mask"+".png", bl_mask)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"samNew"+t_suffix+"original"+".png", image_bgr)
        cv2.imwrite("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"samNew"+t_suffix+"del"+".png", combine_de)
        new_current_seg_path[0]="D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"samNew"+t_suffix+"original"+".png"
    im = Image.open("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+"samNew"+t_suffix+".png") 
    width, height = im.size
    imgs=[preprocess(im)]
    image_input = torch.tensor(np.stack(imgs)).cuda()
    with torch.no_grad():
        image_features0 = model.encode_image(image_input).float()
        image_features1 = model2.encode_image(image_input).float()
    image_features0 /= image_features0.norm(dim=-1, keepdim=True)
    im_features0=image_features0.cpu().numpy()
    image_features1 /= image_features1.norm(dim=-1, keepdim=True)
    im_features1=image_features1.cpu().numpy()
    im_features=(w1*im_features0+w2*im_features1)/(w1+w2)

    new_mask_state[0]=1
    return jsonify({"error": 1001, "msg": "上传失败","address": "samNew"+t_suffix+".png"})


@app.route('/api/test/getSamClearNew/',methods=['GET','POST'])
def getSamClearNew():
    new_mask_state[0]=0
    new_mask_list.remove(new_mask_list[0])
    return jsonify({"error": 1001, "msg": "上传失败"})


@app.route('/api/test/getCompose/',methods=['GET','POST'])
def getCompose():
	# 通过表单中name值获取图片
    start_time=time.time()

    # f = request.files["image"]
    # imgData=f.read()
    # byte_stream = io.BytesIO(imgData)  
    seg_path0=request.form["seg_path"]
    # composed=request.form["composed"]
    extra_intent=request.form["extra_intent"]
    extra_logic=request.form["extra_logic"]
    extra_type=request.form["extra_type"]
    seg_path="D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+seg_path0
    im=Image.open(seg_path)
    # im = Image.open(byte_stream) 
    width, height = im.size
    imgs=[preprocess(im)]
    image_input = torch.tensor(np.stack(imgs)).cuda()
    with torch.no_grad():
        image_features0 = model.encode_image(image_input).float()
        image_features1 = model2.encode_image(image_input).float()
    image_features0 /= image_features0.norm(dim=-1, keepdim=True)
    im_features0=image_features0.cpu().numpy()
    image_features1 /= image_features1.norm(dim=-1, keepdim=True)
    im_features1=image_features1.cpu().numpy()
    im_features=(w1*im_features0+w2*im_features1)/(w1+w2)

    im1 = Image.open("D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+seg_path0[:-4]+"white"+".png") 
    width, height = im1.size
    imgs1=[preprocess(im1)]
    image_input1 = torch.tensor(np.stack(imgs1)).cuda()
    with torch.no_grad():
        image_features01 = model.encode_image(image_input1).float()
        image_features11 = model2.encode_image(image_input1).float()
    image_features01 /= image_features01.norm(dim=-1, keepdim=True)
    im_features01=image_features01.cpu().numpy()
    image_features11 /= image_features11.norm(dim=-1, keepdim=True)
    im_features11=image_features11.cpu().numpy()
    im_features1=(w1*im_features01+w2*im_features11)/(w1+w2)
    # im_features2=(4*im_features+3*im_features1)/7
    im_features2=(7*im_features+3*im_features1)/10

    print(im_features.shape)
    # img=im_features[0]
    img=im_features2[0]

    if extra_type=="text":
        text_tokens1 = clip.tokenize(extra_intent).cuda()
        with torch.no_grad():
            text_features10 = model.encode_text(text_tokens1).float()
            text_features11= model2.encode_text(text_tokens1).float()
        text_features10 /= text_features10.norm(dim=-1, keepdim=True)
        text_features10=text_features10.cpu().numpy()
        text_features11 /= text_features11.norm(dim=-1, keepdim=True)
        text_features11=text_features11.cpu().numpy()
        add_features1=(w1*text_features10+w2*text_features11)/(w1+w2)
    else:
        extra_path="D:/NFTSearch/NFTSearch/Frontend/src/assets/mask/"+extra_intent
        im=Image.open(extra_path)
        # im = Image.open(byte_stream) 
        width, height = im.size
        imgs=[preprocess(im)]
        image_input = torch.tensor(np.stack(imgs)).cuda()
        with torch.no_grad():
            image_features0 = model.encode_image(image_input).float()
            image_features1 = model2.encode_image(image_input).float()
        image_features0 /= image_features0.norm(dim=-1, keepdim=True)
        im_features0=image_features0.cpu().numpy()
        image_features1 /= image_features1.norm(dim=-1, keepdim=True)
        im_features1=image_features1.cpu().numpy()
        im_features=(w1*im_features0+w2*im_features1)/(w1+w2)

        im1 = Image.open(extra_path[:-4]+"white"+".png") 
        width, height = im1.size
        imgs1=[preprocess(im1)]
        image_input1 = torch.tensor(np.stack(imgs1)).cuda()
        with torch.no_grad():
            image_features01 = model.encode_image(image_input1).float()
            image_features11 = model2.encode_image(image_input1).float()
        image_features01 /= image_features01.norm(dim=-1, keepdim=True)
        im_features01=image_features01.cpu().numpy()
        image_features11 /= image_features11.norm(dim=-1, keepdim=True)
        im_features11=image_features11.cpu().numpy()
        im_features1=(w1*im_features01+w2*im_features11)/(w1+w2)
        # im_features2=(4*im_features+3*im_features1)/7
        add_features1=(7*im_features+3*im_features1)/10

    price_cans=[]
    im_paths0=[]
    im_paths=[]
    if extra_logic=="&":
        feature_aver=(img+add_features1[0])/2
        KNNs=ball_search_C(feature_aver, 500)
        for i in range(100):
            im_paths0.append(str(all_ps[KNNs[i]]))
        for i in range(100):
            im_paths.append(im_paths0[i])
    elif extra_logic=="||":
        KNNs1=ball_search_C(img, 300)
        KNNs2=ball_search_C(add_features1[0], 300)
        even=0
        KNNS=[KNNs1, KNNs2]
        knns_idx=[0,0]
        while len(im_paths0)<100:
            eve_idx=even%2
            knn=KNNS[eve_idx]
            knn_id=knns_idx[eve_idx]
            if all_ps[knn[knn_id]] not in im_paths0:
                im_paths0.append(all_ps[knn[knn_id]])
            even=even+1
            knns_idx[eve_idx]=knns_idx[eve_idx]+1
            for i in range(100):
                im_paths.append(im_paths0[i])
    elif extra_logic=="&!":
        # KNNs1=ball_search_C(img, 500)
        # KNNs2=ball_search_C(img, 500)
        # embeds1=[]
        # for u in range(100):
        #     embeds1.append(all_es[KNNs2[u]].reshape((1,-1)))
        # embeds2=np.concatenate(embeds1)
        # similarity=text_features1 @ embeds2.T
        # mean_sim=np.mean(similarity)
        # min_sim=np.min(similarity)
        # median=np.percentile(similarity,30)
        # temp_ps2=np.argsort(similarity[0])
        # # dtype=[('pat')]
        # print("im_paths1_len")
        # print(len(im_paths1))
        # print("similarity_len")
        # print(similarity.shape)
        # print(temp_ps2)
        # for u in range(100):
        #     im_paths2.append(im_paths1[temp_ps2[u]])
        feature_aver=(img-add_features1[0])/2
        KNNs=ball_search_C(feature_aver, 500)
        for i in range(100):
            im_paths0.append(str(all_ps[KNNs[i]]))
        for i in range(100):
            im_paths.append(im_paths0[i])
    else:
        KNNs=ball_search_C(img, 200)
        for i in range(200):
            im_paths0.append(str(all_ps[KNNs[i]]))
        embeds1=[]
        for u in range(200):
            embeds1.append(all_es[KNNs[u]].reshape((1,-1)))
        embeds2=np.concatenate(embeds1)
        knns1=knn_search_C(add_features[0],embeds2,100)
        for i in range(100):
            im_paths.append(im_paths0[knns1[i]])

    # KNNs=ball_search_C(img, 100)
    # price_cans=[]
    # im_paths0=[]
    # im_paths=[]
    # for i in range(100):
    #     im_paths0.append(str(all_ps[KNNs[i]]))


    # if composed!="":
    #     embeds1=[]
    #     for u in range(100):
    #         embeds1.append(all_es[KNNs[u]].reshape((1,-1)))
    #     embeds2=np.concatenate(embeds1)
    #     text_tokens1 = clip.tokenize(composed).cuda()
    #     with torch.no_grad():
    #         text_features10 = model.encode_text(text_tokens1).float()
    #         text_features11= model2.encode_text(text_tokens1).float()
    #     text_features10 /= text_features10.norm(dim=-1, keepdim=True)
    #     text_features10=text_features10.cpu().numpy()
    #     text_features11 /= text_features11.norm(dim=-1, keepdim=True)
    #     text_features11=text_features11.cpu().numpy()
    #     text_features1=(w1*text_features10+w2*text_features11)/(w1+w2)
    #     knns1=knn_search_C(text_features1.reshape((1024,)),embeds2,100)
    #     for i in range(100):
    #         im_paths.append(im_paths0[knns1[i]])

    # else:
    #     for i in range(100):
    #         im_paths.append(im_paths0[i])

    for i in range(len(im_paths)):
        s=im_paths[i].split('/')[-1]
        token_id=''.join([i for i in s if i.isdecimal()])
        if "Doodle" in im_paths[i]:
            try:
                price_c=float(PriceDic["Doodles"+str(int(token_id))])#float(re.findall(r"\d+\.?\d*",bestOfferDic["Doodles"+str(int(token_id))])[0])
            except:
                price_c=0
                # print(bestOfferDic["Doodles"+str(int(token_id))])
            
        elif "CoolCat" in im_paths[i]:
            try:
                price_c=float(PriceDic["CoolCat"+str(int(token_id))])#float(re.findall(r"\d+\.?\d*",bestOfferDic["CoolCat"+str(int(token_id))])[0])
            except:
                price_c=0
                # print(bestOfferDic["CoolCat"+str(int(token_id))])
        else:
            #bestOffer[i]=0
            try:
                na=im_paths[i].split("/")[3]
                price_c=float(PriceDic[na+str(int(token_id))])
            except:
                price_c=int(5*np.random.rand()*1000)/1000
        price_cans.append(price_c)
    for i in range(100):
        va[i]=str(im_paths[i])
        s=im_paths[i].split('/')[-1]
        token_id=''.join([i for i in s if i.isdecimal()])
        bestOffer[i]=str(price_cans[i])+" WETH"
        colname=im_paths[i].split("/")[3]
        addresses.append(address_dic[colname])
   
    end_time=time.time()
    print("Time per query: ")
    print(end_time-start_time)
    # for i in range(11):
    #     va[i]=str((2*pre[i]+pre1[i])/3)
        #va[i]=str(pre1[i])
    # os.remove("static/query.png")
    # im.save("static/query.png")
    mask_state[0]=0
    mask_list.remove(mask_list[0])
    instruct_state[0]=0
    new_mask_state[0]=0
    if len(new_mask_list)>0:
        new_mask_list.remove(new_mask_list[0])
    return jsonify({"error": 1001, "msg": "上传失败"})