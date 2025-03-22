import keras
from keras.models import Model
from keras.layers import Input, Conv2D, LayerNormalization, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Spatial feature extraction with depthwise separable convolution
    dw_conv = Conv2D(filters=32, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu',
                     kernel_initializer='he_normal', use_bias=False, depth_multiplier=1)(input_layer)
    dw_conv = LayerNormalization()(dw_conv)
    
    # Two fully connected layers for channel-wise feature transformation
    dense1 = Dense(units=128, activation='relu')(dw_conv)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Add the processed features with the original input
    combined = Add()([input_layer, dense2])
    
    # Output layer for classification
    output_layer = Dense(units=10, activation='softmax')(combined)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()