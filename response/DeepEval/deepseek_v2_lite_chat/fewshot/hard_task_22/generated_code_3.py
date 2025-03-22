import keras
from keras.layers import Input, Lambda, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split input into three groups along the channel axis
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Multi-scale feature extraction in the main path
    def main_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor[0])
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor[1])
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor[2])
        concat = Concatenate(axis=-1)([conv1, conv2, conv3])
        return concat

    # 1x1 convolutional layer in the branch path
    def branch_path(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return conv

    # Fusion of outputs from main and branch paths using addition
    main_output = main_path(input_tensor=split_layer)
    branch_output = branch_path(input_tensor=split_layer)
    fusion_output = Add()([main_output, branch_output])

    # Flatten and pass through two fully connected layers for classification
    flatten_layer = Flatten()(fusion_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model

# Instantiate and return the constructed model
model = dl_model()