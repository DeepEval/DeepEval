import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # 1x1 Convolution to increase dimensionality
    x = Conv2D(32, (1, 1), strides=1, padding='same', activation='relu')(input_layer)

    # 3x3 Depthwise separable convolution for feature extraction
    x = DepthwiseConv2D((3, 3), strides=1, padding='same', activation='relu')(x)

    # 1x1 Convolution to reduce dimensionality
    x = Conv2D(1, (1, 1), strides=1, padding='same', activation='relu')(x)

    # Adding the output with the original input
    x = Add()([x, input_layer])

    # Flatten the output
    x = Flatten()(x)

    # Fully connected layer to generate final classification probabilities
    output_layer = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()

# Display model summary
model.summary()