import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    main_path = Lambda(lambda x: tf.split(x, 3, axis=1))(input_layer)
    main_path = [Conv2D(filters=128, kernel_size=(1, 1), activation='relu')(x) for x in main_path]
    main_path = [Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(x) for x in main_path]
    main_path = [Conv2D(filters=512, kernel_size=(5, 5), activation='relu')(x) for x in main_path]
    main_path = Concatenate()(main_path)

    branch_path = Conv2D(filters=512, kernel_size=(1, 1), activation='relu')(input_layer)

    output = main_path + branch_path
    output = BatchNormalization()(output)
    output = Flatten()(output)
    output = Dense(units=128, activation='relu')(output)
    output = Dense(units=64, activation='relu')(output)
    output = Dense(units=10, activation='softmax')(output)

    model = Model(inputs=input_layer, outputs=output)

    return model