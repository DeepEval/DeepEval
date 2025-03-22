import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Multiply, Add
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = GlobalAveragePooling2D()(input_layer)
    main_path = Dense(units=64, activation='relu')(main_path)
    main_path = Dense(units=32, activation='relu')(main_path)
    weights = Dense(units=3, activation='sigmoid')(main_path)
    weights = weights.reshape((1, 1, 3))  # Adjusting the weights to match the input layer's shape
    weighted_features = Multiply()([input_layer, weights])

    # Branch path
    branch_path = Conv2D(filters=3, kernel_size=(3, 3), padding='same')(input_layer)

    # Concatenate the outputs from both paths
    combined = Add()([weighted_features, branch_path])

    # Flatten and add more fully connected layers
    combined = Flatten()(combined)
    combined = Dense(units=128, activation='relu')(combined)
    combined = Dense(units=64, activation='relu')(combined)
    output_layer = Dense(units=10, activation='softmax')(combined)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model