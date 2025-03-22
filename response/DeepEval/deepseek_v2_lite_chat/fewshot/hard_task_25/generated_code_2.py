import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Add, ZeroPadding2D, UpSampling2D, Conv2DTranspose, Multiply, Flatten, Reshape, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    avg_pool1 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)
    avg_pool2 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1_1)
    avg_pool3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1_2)
    
    deconv1 = Conv2DTranspose(filters=64, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(Concatenate()[(avg_pool1, avg_pool2, avg_pool3)])
    deconv2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(Concatenate()[(deconv1, avg_pool1, avg_pool2)])
    deconv3 = Conv2DTranspose(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(Concatenate()[(deconv2, avg_pool2, avg_pool3)])
    
    concat_main_path = Concatenate()[(deconv3, conv1_1, conv1_2)]
    
    conv2 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat_main_path)
    
    # Branch Path
    conv_branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_branch = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_branch)
    
    avg_pool_branch = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv_branch)
    deconv_branch = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(Concatenate()[(avg_pool_branch, conv_branch)])
    
    # Add main path and branch path
    add_layer = Add()[(conv2, deconv_branch)]
    
    conv3 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(add_layer)
    
    # Fully connected layer
    flatten = Flatten()(conv3)
    output = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model