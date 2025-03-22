import keras
from keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Flatten, Dense, Concatenate
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups
    outputs = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Depthwise separable convolutional layers for each group
    def depthwise_separable_conv(filters, kernel_size, strides, activation):
        return Conv2D(filters, kernel_size, strides=strides, padding='same', activation=activation, use_bias=False, kernel_regularizer=keras.regularizers.l2())
    
    # Block 1: 1x1, 3x3, 5x5 depthwise separable convolutions
    conv1 = depthwise_separable_conv(filters=64, kernel_size=1, strides=1, activation='relu')(outputs[0])
    conv2 = depthwise_separable_conv(filters=64, kernel_size=3, strides=1, activation='relu')(outputs[1])
    conv3 = depthwise_separable_conv(filters=64, kernel_size=5, strides=1, activation='relu')(outputs[2])
    
    # Concatenate and fuse the outputs
    fused_features = Concatenate(axis=-1)([conv1, conv2, conv3])
    
    # Flatten and fully connected layer for classification
    flattened = Flatten()(fused_features)
    dense = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# This will output the architecture of the model
print(model.summary())