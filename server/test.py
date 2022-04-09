import requests

images = ['../driver/data/test/Closed/_102.jpg', '../driver/data/test/Open/_117.jpg',
          '../driver/data/test/yawn/118.jpg', '../driver/data/test/no_yawn/1042.jpg',
          '../driver/data/test/no_yawn/1008.jpg', '../driver/data/test/yawn/79.jpg',
          '../driver/data/test/no_yawn/833.jpg', '../driver/data/test/no_yawn/1202.jpg',
          '../driver/data/val/yawn/11.jpg', '../driver/data/val/yawn/532.jpg']

for image in images:
    resp = requests.post("http://localhost:5000/predict",
                         files={"file": open(image, 'rb')})
    print(image, resp.json(), '\n')
