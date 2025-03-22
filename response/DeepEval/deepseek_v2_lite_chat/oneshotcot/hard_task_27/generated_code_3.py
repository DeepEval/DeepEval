import keras
from keras.layers import Input, Conv2D, LayerNormalization, Activation, Add, Dense, Flatten

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Depthwise separable convolutional layer with layer normalization
    dw_conv = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), padding='same',
                      use_depthwise=True, kernel_regularizer=keras.regularizers.l2(0.0005),
                      bias_regularizer=keras.regularizers.l2(0.0005),
                      depth_momentum=0.1, depth_epsilon=1e-6)(input_layer)
    dw_layer_norm = LayerNormalization()(dw_conv)
    dw_conv_act = Activation('relu')(dw_layer_norm)
    
    # Fully connected layer for channel-wise feature transformation
    fc1 = Dense(units=512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005),
                bias_regularizer=keras.regularizers.l2(0.0005))(dw_conv_act)
    fc2 = Dense(units=256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005),
                 bias_regularizer=keras.regularizers.l2(0.0005))(dw_conv_act)
    
    # Add the original input with the processed features
    combined = Add()([input_layer, dw_conv])
    
    # Flatten and pass through two more fully connected layers
    flat = Flatten()(combined)
    dense3 = Dense(units=256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005),
                    bias_regularizer=keras.regularizers.l2(0.0005))(flat)
    dense4 = Dense(units=128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0005),
                    bias_regularizer=keras.regularizers.l2(0.0005))(dense3)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense4)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model