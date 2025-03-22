import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer with shape corresponding to CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))

    # Increase the dimensionality of the input's channels threefold using a 1x1 convolution
    x = Conv2D(filters=9, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Extract initial features using a 3x3 depthwise separable convolution
    x = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(x)

    # Global average pooling to compute channel attention weights
    channel_attention = GlobalAveragePooling2D()(x)
    
    # Fully connected layers for channel attention
    fc1 = Dense(units=9 // 2, activation='relu')(channel_attention)
    fc2 = Dense(units=9, activation='sigmoid')(fc1)

    # Reshape the channel attention weights to match the initial features
    channel_attention_weights = tf.reshape(fc2, [-1, 1, 1, 9])

    # Multiply the initial features by the channel attention weights
    x = Multiply()([x, channel_attention_weights])

    # Reduce dimensionality with a 1x1 convolution
    x = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(x)

    # Combine the output with the initial input
    x = Add()([x, input_layer])

    # Flatten the output and pass through a fully connected layer for classification
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)  # CIFAR-10 has 10 classes

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Use this function to get the model
model = dl_model()