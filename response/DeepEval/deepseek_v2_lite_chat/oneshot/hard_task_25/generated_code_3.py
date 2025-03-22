import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D, UpSampling2D, ZeroPadding2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    main_conv = Conv2D(filters=32, kernel_size=3, activation='relu')(input_layer)
    main_pool = MaxPooling2D(pool_size=2, strides=2)(main_conv)
    
    # Branch path
    branch_conv1 = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(main_pool)
    branch_pool1 = AveragePooling2D(pool_size=2, strides=2)(input_layer)
    branch_conv2 = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(branch_pool1)
    branch_upconv1 = UpSampling2D(size=2)(branch_conv2)
    branch_conv3 = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(branch_upconv1)
    branch_upconv2 = UpSampling2D(size=2)(branch_conv3)
    branch_pool2 = AveragePooling2D(pool_size=2, strides=2)(branch_upconv2)
    
    # Concatenate the outputs
    concat_layer = Concatenate()([main_conv, branch_conv1, branch_conv2])
    
    # 1x1 convolution to match channels
    conv1x1 = Conv2D(filters=64, kernel_size=1, activation='relu')(concat_layer)
    
    # Fully connected layer for 10-class classification
    output_layer = Dense(units=10, activation='softmax')(conv1x1)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and return the model
model = dl_model()
model.summary()