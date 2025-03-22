import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, GlobalAveragePooling2D, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Split the input into three groups and apply separable convolution
    lambda_layer = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    conv1 = layers.SeparableConv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(lambda_layer[0])
    conv2 = layers.SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(lambda_layer[1])
    conv3 = layers.SeparableConv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(lambda_layer[2])
    output_tensor = layers.Concatenate()([conv1, conv2, conv3])
    
    # Block 2: Enhanced feature extraction
    conv_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(output_tensor)
    series_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output_tensor)
    series_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(series_path)
    series_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(series_path)
    max_pool_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(output_tensor)
    output_tensor = layers.Concatenate()([conv_path, series_path, max_pool_path])
    
    # Global average pooling and dense layer
    bath_norm = BatchNormalization()(output_tensor)
    global_avg_pool = GlobalAveragePooling2D()(bath_norm)
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model