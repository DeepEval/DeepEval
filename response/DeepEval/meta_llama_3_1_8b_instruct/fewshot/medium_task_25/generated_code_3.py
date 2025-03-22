import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense
from keras.layers import Lambda

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    def path1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv1
    
    def path2(input_tensor):
        avgpool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avgpool)
        return conv1
    
    def path3(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        concat = Concatenate()([conv2, conv3])
        return concat
    
    def path4(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv4 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        concat = Concatenate()([conv2, conv3, conv4])
        return concat
    
    path1_output = path1(input_tensor=input_layer)
    path2_output = path2(input_tensor=input_layer)
    path3_output = path3(input_tensor=input_layer)
    path4_output = path4(input_tensor=input_layer)
    
    output_tensor = Concatenate()([path1_output, path2_output, path3_output, path4_output])
    
    flatten = Flatten()(output_tensor)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])