import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, multiply, Add, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    global_avg_pool = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense2 = Dense(units=32*32*3, activation='sigmoid')(dense1)  # Match input shape (32*32*3)
    reshaped_weights = Reshape((32, 32, 3))(dense2)
    multiplied = multiply([input_layer, reshaped_weights])

    # Branch Path
    branch_path = input_layer  # Directly using the input layer

    # Combining Main Path and Branch Path
    combined = Add()([multiplied, branch_path])

    # Final Layers
    flatten_layer = Flatten()(combined)
    final_dense1 = Dense(units=128, activation='relu')(flatten_layer)
    final_dense2 = Dense(units=10, activation='softmax')(final_dense1)  # 10 classes for CIFAR-10

    # Model Construction
    model = Model(inputs=input_layer, outputs=final_dense2)

    return model