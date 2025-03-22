import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D, Conv2DTranspose

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(conv1)
    
    # Branch path
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pool1)
    avg_pool = AveragePooling2D(pool_size=(1, 1), strides=2, padding='valid')(conv2)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(avg_pool)
    deconv1 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid')(conv3)
    up_conv1 = UpSampling2D(size=(2, 2))(deconv1)
    concat1 = Concatenate()([conv1, conv2, deconv1, up_conv1])
    
    # Batch normalization and flattening
    batch_norm1 = BatchNormalization()(concat1)
    flatten1 = Flatten()(batch_norm1)
    
    # Dense layers
    dense1 = Dense(units=128, activation='relu')(flatten1)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()
model.summary()