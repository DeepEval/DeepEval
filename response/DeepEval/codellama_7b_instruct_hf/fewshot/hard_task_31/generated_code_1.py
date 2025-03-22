import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator

 和 return model
def dl_model():
    input_shape = (32, 32, 3)


    def block1(x):
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        return x


    def block2(x):
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        return x


    def block3(x):
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        return x


    def block4(x):
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        return x


    def final_dense(x):
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(10, activation='softmax')(x)
        return x


    model = Model(inputs=input_shape, outputs=final_dense(block4(block3(block2(block1(input_shape))))))

    return model


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('data/train_data', target_size=(32, 32), batch_size=32, class_mode='categorical')
test_generator = test_datagen.flow_from_directory('data/test_data', target_size=(32, 32), batch_size=32, class_mode='categorical')

model.fit_generator(train_generator, epochs=10, validation_data=test_generator)


test_loss, test_acc = model.evaluate(test_generator)
print('Test accuracy:', test_acc)