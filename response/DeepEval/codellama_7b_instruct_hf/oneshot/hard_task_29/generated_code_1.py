import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # First block
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv1)
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    merged_path = Concatenate()([conv2, branch_path])
    batch_norm = BatchNormalization()(merged_path)
    flatten_layer = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    model1 = keras.Model(inputs=input_layer, outputs=output_layer)

    # Second block
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv1)
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    merged_path = Concatenate()([conv2, branch_path])
    batch_norm = BatchNormalization()(merged_path)
    flatten_layer = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    model2 = keras.Model(inputs=input_layer, outputs=output_layer)

    # Combine the outputs of the two blocks
    output_layer = Concatenate()([model1.output, model2.output])
    model = keras.Model(inputs=model1.inputs, outputs=output_layer)

    return model