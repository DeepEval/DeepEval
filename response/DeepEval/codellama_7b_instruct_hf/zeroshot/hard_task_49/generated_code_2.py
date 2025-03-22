from keras.models import Model
from keras.layers import Input, Dense, Flatten, Lambda, MaxPooling2D, AvgPooling2D, Conv2D, DepthwiseSeparableConv2D
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet_v2 import MobileNetV2


def dl_model():
    
    # Define the input shape
    input_shape = (28, 28, 1)

    # First block
    # Use VGG16 as the base model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Add three average pooling layers with varying scales
    pooling_layers = []
    for pool_size in [1, 2, 4]:
        pooling_layers.append(
            MaxPooling2D(pool_size=(pool_size, pool_size), strides=(pool_size, pool_size))
        )

    # Flatten and concatenate the outputs from the pooling layers
    pooling_layers_output = base_model.output
    for pooling_layer in pooling_layers:
        pooling_layers_output = pooling_layer(pooling_layers_output)
        pooling_layers_output = Flatten()(pooling_layers_output)

    # Add a fully connected layer
    fc_layer = Dense(512, activation='relu')(pooling_layers_output)

    # Reshape the output to 4-dimensional tensor
    reshape_layer = Reshape((4, 4, 128))(fc_layer)

    # Second block
    # Use MobileNetV2 as the base model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(4, 4, 128))

    # Split the input into four groups
    groups = tf.split(base_model.output, 4, axis=-1)

    # Apply depthwise separable convolutional layers with different kernel sizes
    for group in groups:
        group = Conv2D(128, kernel_size=(1, 1), padding='same')(group)
        group = DepthwiseSeparableConv2D(128, kernel_size=(3, 3), padding='same')(group)
        group = DepthwiseSeparableConv2D(128, kernel_size=(5, 5), padding='same')(group)
        group = DepthwiseSeparableConv2D(128, kernel_size=(7, 7), padding='same')(group)

    # Concatenate the outputs from the depthwise separable convolutional layers
    output = tf.concat(groups, axis=-1)

    # Flatten the output
    output = Flatten()(output)

    # Add a fully connected layer
    fc_layer = Dense(512, activation='relu')(output)

    # Define the model
    model = Model(inputs=base_model.input, outputs=fc_layer)

    # Compile the model with a loss function and an optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Return the constructed model
    return model