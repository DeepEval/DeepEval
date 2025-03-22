import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense


def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first path
    path1 = Input(shape=input_shape)
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    path1_output = Flatten()(pool2)

    # Define the second path
    path2 = Input(shape=input_shape)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)
    path2_output = Flatten()(pool3)

    # Add the outputs from both paths
    merged_output = Add()([path1_output, path2_output])

    # Flatten the merged output
    flattened_output = Flatten()(merged_output)

    # Define the fully connected layers
    fc1 = Dense(units=512, activation='relu')(flattened_output)
    fc2 = Dense(units=10, activation='softmax')(fc1)

    # Define the model
    model = keras.Model(inputs=[path1, path2], outputs=fc2)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])