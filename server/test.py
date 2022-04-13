import requests

images = ['../pytorch/data/test/Closed/_102.jpg', '../pytorch/data/test/Open/_117.jpg',
          '../pytorch/data/test/yawn/118.jpg', '../pytorch/data/test/no_yawn/1042.jpg',
          '../pytorch/data/test/no_yawn/1008.jpg', '../pytorch/data/test/yawn/79.jpg',
          '../pytorch/data/test/no_yawn/833.jpg', '../pytorch/data/test/no_yawn/1202.jpg',
          '../pytorch/data/val/yawn/11.jpg', '../pytorch/data/val/yawn/532.jpg']

for image in images:
    resp = requests.post("http://localhost:5000/predict",
                         files={"file": open(image, 'rb')})
    print(image, resp.json(), '\n')
