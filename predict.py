import  tensorflow as tf 
from tensorflow.keras.models import load_model
import  tensorflow as tf 
import  matplotlib.pyplot as plt 
import  numpy as np
from    tensorflow.keras.preprocessing.image import ImageDataGenerator
from    tensorflow.keras.preprocessing import image
import  matplotlib.image as mpimg
from    PIL import ImageOps, Image, ImageDraw, ImageFont


class_names = ['น้ำพริกกุ้งเสียบ', 'หมูฮ้อง', 'โลบะ', 'โอต้าว']

new_model = load_model(r'C:\Users\Admin\Desktop\dataset\Phuketfood')

img = image.load_img(r'C:\Users\Admin\Desktop\dataset\test\6.jpg',target_size=(200,200))

x = image.img_to_array(img)
x = np.expand_dims(x,axis = 0)
images = np.vstack([x])
val = new_model.predict(images)

index = np.argmax(val)
print(f'Prediction is {class_names[index]}')
print(val)
draw = ImageDraw.Draw(img)
font = ImageFont.truetype(r'C:\Windows\Fonts\angsana.ttc', 30)
draw.text((10, 160), class_names[index] ,(255,0,0), font=font)

img.show()

