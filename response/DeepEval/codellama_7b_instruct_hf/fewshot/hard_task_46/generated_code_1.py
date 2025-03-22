import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, Dense, Reshape
from keras.applications.cifar10 import CIFAR10
from keras.preprocessing.image import ImageDataGenerator


cifar10 = CIFAR10(train_dir='path/to/cifar10/train', validation_dir='path/to/cifar10/validation')

 和 return model
def dl_model():

    input_shape = (32, 32, 3)


    def block_1(input_tensor):
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x)
        x = Flatten()(x)
        return x


    def block_2(input_tensor):
        x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x)
        x = Flatten()(x)
        return x


    input_layer = Input(shape=input_shape)
    block_1_output = block_1(input_layer)
    block_2_output = block_2(input_layer)
    concat_output = Concatenate()([block_1_output, block_2_output])
    x = Flatten()(concat_output)
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dense(units=10, activation='softmax')(x)
    model = keras.Model(inputs=input_layer, outputs=x)


    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory('path/to/cifar10/train', target_size=(32, 32), batch_size=32, class_mode='categorical')
    test_generator = test_datagen.flow_from_directory('path/to/cifar10/test', target_size=(32, 32), batch_size=32, class_mode='categorical')

    model.fit(train_generator, epochs=10, validation_data=test_generator)

    return model