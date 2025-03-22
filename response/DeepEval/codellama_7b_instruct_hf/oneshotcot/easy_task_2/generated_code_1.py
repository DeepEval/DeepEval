import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (224, 224, 3)

    # Define the first sequential feature extraction layer
    conv_layer1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu')
    avg_pooling_layer1 = AveragePooling2D(pool_size=(2, 2), strides=2)

    # Define the second sequential feature extraction layer
    conv_layer2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')
    avg_pooling_layer2 = AveragePooling2D(pool_size=(2, 2), strides=2)

    # Define the third convolutional layer
    conv_layer3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu')

    # Define the fourth convolutional layer
    conv_layer4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu')

    # Define the fifth convolutional layer
    conv_layer5 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu')

    # Define the average pooling layer to reduce dimensionality
    avg_pooling_layer3 = AveragePooling2D(pool_size=(2, 2), strides=2)

    # Define the flatten layer to flatten the feature maps
    flatten_layer = Flatten()

    # Define the first fully connected layer
    dense_layer1 = Dense(units=128, activation='relu')

    # Define the second fully connected layer
    dense_layer2 = Dense(units=64, activation='relu')

    # Define the third fully connected layer
    dense_layer3 = Dense(units=10, activation='softmax')

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Define the first sequential feature extraction layer
    x = conv_layer1(input_layer)
    x = avg_pooling_layer1(x)

    # Define the second sequential feature extraction layer
    x = conv_layer2(x)
    x = avg_pooling_layer2(x)

    # Define the third convolutional layer
    x = conv_layer3(x)

    # Define the fourth convolutional layer
    x = conv_layer4(x)

    # Define the fifth convolutional layer
    x = conv_layer5(x)

    # Define the average pooling layer to reduce dimensionality
    x = avg_pooling_layer3(x)

    # Define the flatten layer to flatten the feature maps
    x = flatten_layer(x)

    # Define the first fully connected layer
    x = dense_layer1(x)

    # Define the second fully connected layer
    x = dense_layer2(x)

    # Define the third fully connected layer
    x = dense_layer3(x)

    # Define the output layer
    output_layer = dense_layer3(x)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model