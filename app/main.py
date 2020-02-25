import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import argparse
from torchvision import datasets, models, transforms
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO
from pathlib import Path
import os
import sys
import json

#from unrar import rarfile
from pyunpack import Archive

model_file_url = 'https://drive.google.com/uc?export=download&id=1nLbLcm1uv-nGA_KNGzA47cqpt_lyLTV_' # inception model
# model_file_url = 'https://drive.google.com/uc?export=download&id=1DfQMqvHKENNQBjxBmmpjJi_YGVLCTYyP' # densenet201 model

model_file_name = 'modelInception'
#model_file_name = 'modelDensenet201'

if model_file_name == 'modelInception':
    input_size = 299
elif model_file_name == 'modelDensenet201':
    input_size = 224

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_device():
    model_path = path/'models'/f'{model_file_name}.pth'
    await download_file(model_file_url, model_path) # download model from Google Drive
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # load model on gpu and set it on test mode
        model = torch.load(model_path)
        model.eval()
        model.cuda(device)
    else:
        # load model on cpu and set it on test mode
        model = torch.load(model_path, map_location='cpu')
        model.eval()

    return model, device

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_device())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

async def download_images(url_dir):
    data_path = path/'dataset_test.rar'
    await download_file(url_dir, data_path) # download data from Dropbox
    Archive(data_path).extractall(".")
    #rar = rarfile.RarFile(data_path)
    #rar.extractall()

    # r=root, d=directory, f=files
    for r, d, f in os.walk(path/'reto_deep_learning'):
        for directory in d:
            if directory == 'test_img':
                data_dir = os.path.join(r, directory)
    
    return data_dir

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    print('oli')
    data = await request.files('file')
    data.save(os.path.join('app/', 'inputjson.xml'))
    #data = await (data['file'].read())
    #root = json.load(data)
    itemUrl = root['imageUrl']

    data_dir = download_images(itemUrl)
    dataloaders_dict = dataLoaders(input_size, data_dir)
    predictions = test_model(model_inception, dataloaders_dict)
    
    return JSONResponse({'result': str(f'{len(predictions)} images were processed')})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=8080)
