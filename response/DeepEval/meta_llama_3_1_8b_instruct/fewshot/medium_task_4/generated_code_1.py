import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 dataset has 32x32x3 images
    
    def path1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
        conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
        pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
        return pool2
    
    def path2(input_tensor):
        conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv
    
    path1_output = path1(input_tensor=input_layer)
    path2_output = path2(input_tensor=input_layer)
    adding_layer = Add()([path1_output, path2_output])
    
    flatten_layer = Flatten()(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model