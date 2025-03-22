import keras
from keras.models import Model
from keras.layers import Input, Conv2D, LayerNormalization, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # 7x7 depthwise separable convolutional layer with layer normalization
    conv = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu',
                  depthwise_initializer='he_normal', pointwise_initializer='glorot_uniform')(input_layer)
    conv = LayerNormalization(epsilon=1e-6)(conv)
    
    # Two fully connected layers for channel-wise feature transformation
    dense1 = Dense(units=512, activation='relu')(conv)
    dense2 = Dense(units=256, activation='relu')(dense1)
    
    # Add the processed features with the original input
    added_layer = Add()([input_layer, conv])
    
    # Output layer with 10 categories
    output_layer = Dense(units=10, activation='softmax')(added_layer)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model