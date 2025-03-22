import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    def path1(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv1
    
    def path2(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 7), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(7, 1), padding='same', activation='relu')(conv1)
        return Concatenate()([conv2, conv3])
    
    def path3(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 7), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(7, 1), strides=(7, 1), padding='same', activation='relu')(conv1)
        conv4 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 7), padding='same', activation='relu')(conv2)
        conv5 = Conv2D(filters=64, kernel_size=(7, 1), strides=(7, 1), padding='same', activation='relu')(conv3)
        return Concatenate()([conv4, conv5])
    
    def path4(input_tensor):
        pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool)
        return conv1
    
    path1_output = path1(input_tensor=input_layer)
    path2_output = path2(input_tensor=input_layer)
    path3_output = path3(input_tensor=input_layer)
    path4_output = path4(input_tensor=input_layer)
    fused_output = Concatenate()([path1_output, path2_output, path3_output, path4_output])
    
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model