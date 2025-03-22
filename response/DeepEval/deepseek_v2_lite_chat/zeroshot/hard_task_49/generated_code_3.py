import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Concatenate, Lambda, Reshape, Dense
from keras.models import Model
from keras.layers import Layer

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))
    
    # Block 1: Three average pooling layers
    p1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(inputs)
    p2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(p1)
    p3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(p2)
    
    # Flatten and concatenate the pooled outputs
    flat_p1 = Flatten()(p3)
    flat_p2 = Flatten()(p2)
    flat_p3 = Flatten()(p1)
    concatenated = Concatenate()([flat_p3, flat_p2, flat_p1])
    
    # Reshape the concatenated output to prepare for the second block
    reshaped = Reshape((-1, 3))(concatenated)
    
    # Block 2: Multi-branch feature extraction
    split_1 = Lambda(lambda x: x[:, :, None, :])(reshaped)
    split_2 = Lambda(lambda x: x[:, None, :, None])(reshaped)
    split_3 = Lambda(lambda x: x[:, None, None, :])(reshaped)
    split_4 = Lambda(lambda x: x[:, None, None, None])(reshaped)
    
    # Depthwise separable convolutions
    dconv_1 = Conv2D(64, (1, 3), padding='valid', use_bias=False)(split_1)
    dconv_2 = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False)(dconv_1)
    dconv_2 = Conv2D(64, (1, 1), padding='same', use_bias=False)(dconv_2)
    
    dconv_3 = Conv2D(128, (3, 5), padding='valid', use_bias=False)(split_2)
    dconv_4 = DepthwiseConv2D((5, 5), strides=(1, 1), padding='same', use_bias=False)(dconv_3)
    dconv_4 = Conv2D(128, (1, 1), padding='same', use_bias=False)(dconv_4)
    
    dconv_5 = Conv2D(256, (5, 7), padding='valid', use_bias=False)(split_3)
    dconv_6 = DepthwiseConv2D((7, 7), strides=(1, 1), padding='same', use_bias=False)(dconv_5)
    dconv_6 = Conv2D(256, (1, 1), padding='same', use_bias=False)(dconv_6)
    
    dconv_7 = Conv2D(512, (7, 1), padding='valid', use_bias=False)(split_4)
    dconv_8 = DepthwiseConv2D((1, 1), strides=(1, 1), padding='same', use_bias=False)(dconv_7)
    dconv_8 = Conv2D(512, (1, 1), padding='same', use_bias=False)(dconv_8)
    
    # Concatenate the outputs from all branches
    concatenated_features = Concatenate(axis=-1)([dconv_2, dconv_4, dconv_6, dconv_8])
    
    # Flatten and pass through a fully connected layer
    flattened = Flatten()(concatenated_features)
    outputs = Dense(10, activation='softmax')(flattened)
    
    # Build the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Create the model
model = dl_model()
model.summary()