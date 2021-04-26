import  tensorflow as tf 
import  matplotlib.pyplot as plt 
import  numpy as np
from    tensorflow.keras.preprocessing.image import ImageDataGenerator
from    tensorflow.keras.preprocessing import image
import  matplotlib.image as mpimg
from    PIL import ImageOps, Image, ImageDraw, ImageFont
from sklearn.metrics import classification_report, confusion_matrix



train = ImageDataGenerator(rescale = 1/255)
validation = ImageDataGenerator(rescale = 1/255)
batch_size = 3
num_of_train_samples = 48
num_of_test_samples = 12
epochs = 80

train_dataset = train.flow_from_directory(
    r'C:\Users\Admin\Desktop\dataset\train',
    
    target_size =(200,200),
    batch_size = batch_size,
    class_mode ='categorical'
    
)

validation_dataset = validation.flow_from_directory(
    r'C:\Users\Admin\Desktop\dataset\validation',
    
    target_size =(200,200),
    batch_size = batch_size,
    class_mode ='categorical'
)

model = tf.keras.models.Sequential([
    #
    tf.keras.layers.Conv2D(16,(3,3),activation = 'relu' , input_shape = (200,200,3)),
    tf.keras.layers.MaxPool2D(2,2),
    #
    tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    #
    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    ##
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation = 'relu'),
    tf.keras.layers.Dense(4,activation='softmax')
])

model.compile(
    loss = 'categorical_crossentropy',
    optimizer = tf.keras.optimizers.RMSprop(lr=0.001),
    metrics = ['accuracy']
)

model.fit(
    train_dataset,
    steps_per_epoch=num_of_train_samples // batch_size,
    epochs = epochs,
    validation_data = validation_dataset,
    validation_steps=num_of_test_samples // batch_size
)

model.save('Phuketfood')

print(train_dataset.class_indices)



Y_pred = model.predict_generator(validation_dataset, epochs // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_dataset.classes, y_pred))
print('Classification Report')
class_names = ['น้ำพริกกุ้งเสียบ', 'หมูฮ้อง', 'โลบะ', 'โอต้าว']
print(classification_report(validation_dataset.classes, y_pred, target_names=class_names))