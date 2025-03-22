import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Multiply

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Two parallel convolutional branches with convolutional kernels of varying sizes (3x3 and 5x5 respectively)
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the outputs of both branches through addition
    branch_output = Add()([branch1, branch2])

    # Global average pooling layer to compress the features
    pooling_layer = GlobalAveragePooling2D()(branch_output)

    # Two fully connected layers that apply a softmax function to generate attention weights
    attention_layer1 = Dense(units=16, activation='relu')(pooling_layer)
    attention_layer2 = Dense(units=10, activation='softmax')(attention_layer1)

    # Multiply the output of each branch by its corresponding weight and add to produce the final weighted output
    weighted_output = Multiply()([branch1, attention_layer2]) + Multiply()([branch2, attention_layer2])

    # Flatten the output and deliver the probability distribution across the 10 classes
    output_layer = Flatten()(weighted_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model