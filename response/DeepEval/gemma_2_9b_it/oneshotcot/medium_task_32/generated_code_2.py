import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups
    split_tensor = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)

    # Process each group with depthwise separable convolutions
    group1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
    group2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
    group3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])

    # Concatenate the outputs
    concatenated_output = Concatenate(axis=3)([group1, group2, group3])

    # Flatten and classify
    flatten_layer = Flatten()(concatenated_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model