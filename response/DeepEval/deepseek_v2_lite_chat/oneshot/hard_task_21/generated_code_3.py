import tensorflow as tf
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input Layer
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    main_output1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_output2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_output3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    output1, output2, output3 = Lambda(lambda x: tf.split(x, 3, axis=-1))( [main_output1, main_output2, main_output3])
    main_output = Concatenate()([output1, output2, output3])

    # Branch Path
    branch_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Model Conclusions
    batch_norm_main = BatchNormalization()(main_output)
    batch_norm_branch = BatchNormalization()(branch_output)
    flatten_layer = Flatten()(batch_norm_main)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model