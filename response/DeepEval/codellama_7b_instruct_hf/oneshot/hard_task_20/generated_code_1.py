import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the main path of the model
    main_path = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path_output = Concatenate()([conv1, conv2, conv3])

    # Define the branch path of the model
    branch_path = Input(shape=(32, 32, 3))
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path_output = BatchNormalization()(conv1)

    # Combine the main and branch paths using addition
    fused_features = Add()([main_path_output, branch_path_output])

    # Apply a final convolutional layer and flatten the output
    fused_features = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(fused_features)
    fused_features = Flatten()(fused_features)

    # Apply two fully connected layers for classification
    fc1 = Dense(units=128, activation='relu')(fused_features)
    fc2 = Dense(units=10, activation='softmax')(fc1)

    # Define the model
    model = keras.Model(inputs=[main_path, branch_path], outputs=fc2)

    return model