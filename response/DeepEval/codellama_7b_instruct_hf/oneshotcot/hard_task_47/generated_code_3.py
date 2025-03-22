import keras
from keras.layers import Input, Conv2D, DepthwiseSeparableConv2D, MaxPooling2D, BatchNormalization, Flatten, Dense
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


def dl_model():
    
    input_shape = (32, 32, 3)


    def block_1(input_tensor):
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_tensor)
        x = DepthwiseSeparableConv2D(3, (1, 1), activation='relu', padding='same')(x)
        x = DepthwiseSeparableConv2D(3, (5, 5), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        return x


    def block_2(input_tensor):
        x = Conv2D(64, (1, 1), activation='relu', padding='same')(input_tensor)
        x = DepthwiseSeparableConv2D(3, (1, 1), activation='relu', padding='same')(x)
        x = DepthwiseSeparableConv2D(3, (1, 7), activation='relu', padding='same')(x)
        x = DepthwiseSeparableConv2D(3, (7, 1), activation='relu', padding='same')(x)
        x = DepthwiseSeparableConv2D(3, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        return x


    def block_3(input_tensor):
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_tensor)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        return x


    model = Model(inputs=Input(shape=input_shape), outputs=block_1(block_2(block_3(block_1(block_2(block_3(Input(shape=input_shape))))))))


    model.compile(optimizer='adam',
                loss=to_categorical(10),
                metrics=['accuracy'])


    datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(train_data_dir, target_size=(32, 32), batch_size=32, class_mode='categorical')
    validation_generator = val_datagen.flow_from_directory(validation_data_dir, target_size=(32, 32), batch_size=32, class_mode='categorical')

    model.fit_generator(train_generator, validation_data=validation_generator, epochs=10, verbose=2)


    test_loss, test_acc = model.evaluate(test_data_dir, verbose=2)
    print('Test accuracy:', test_acc)

    return model