import keras
from keras.models import Model
from keras.layers import Input, DepthwiseConv2D, LayerNormalization, Conv2D, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    conv = DepthwiseConv2D(kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    norm = LayerNormalization()(conv)
    
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(norm)
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Branch path
    branch_input = Input(shape=input_layer.shape)
    branch_conv = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch_input)
    branch_norm = LayerNormalization()(branch_conv)
    
    # Combine paths
    combined = Add()([conv, branch_norm])
    flatten = Flatten()(combined)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    # Model
    model = Model(inputs=[input_layer, branch_input], outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()
model.summary()