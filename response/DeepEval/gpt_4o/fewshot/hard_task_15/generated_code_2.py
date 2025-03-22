import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten
from keras.datasets import cifar10

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images have a shape of 32x32x3

    # Main Path
    global_avg_pool = GlobalAveragePooling2D()(input_layer)
    fc1 = Dense(units=64, activation='relu')(global_avg_pool)
    fc2 = Dense(units=3, activation='sigmoid')(fc1)  # Assuming we want 3 channels (for RGB)
    reshaped_weights = keras.layers.Reshape((1, 1, 3))(fc2)
    scaled_features = Multiply()([input_layer, reshaped_weights])

    # Branch Path
    branch_path = input_layer  # Direct connection

    # Combine main path and branch path
    combined = Add()([scaled_features, branch_path])

    # Final fully connected layers
    flatten = Flatten()(combined)
    fc3 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(fc3)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model