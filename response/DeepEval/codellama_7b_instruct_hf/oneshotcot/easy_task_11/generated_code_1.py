import keras
from keras.layers import Input, AveragePooling2D, Conv2D, Flatten, Dropout, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the average pooling layer
    avg_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(input_layer)

    # Define the 1x1 convolutional layer
    conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(avg_pool)

    # Flatten the feature maps
    flat = Flatten()(conv1x1)

    # Define the dropout layer
    dropout = Dropout(0.2)(flat)

    # Define the two fully connected layers
    fc1 = Dense(units=128, activation='relu')(dropout)
    fc2 = Dense(units=10, activation='softmax')(fc1)

    # Define the output layer
    output_layer = fc2

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model