import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Add
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB images

    # Main path
    global_avg_pool = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=512, activation='relu')(global_avg_pool)
    dense2 = Dense(units=32 * 32 * 3, activation='sigmoid')(dense1)  # Generate weights for the input layer
    reshaped_weights = Reshape((32, 32, 3))(dense2)  # Reshape to match the input layer's shape
    weighted_input = keras.layers.multiply([input_layer, reshaped_weights])  # Element-wise multiplication

    # Branch path
    branch_output = input_layer  # Directly connected to the input layer

    # Combine the outputs of both paths
    combined_output = Add()([weighted_input, branch_output])

    # Final dense layers
    flatten_output = GlobalAveragePooling2D()(combined_output)  # Pool the combined output
    final_dense1 = Dense(units=256, activation='relu')(flatten_output)
    final_output = Dense(units=10, activation='softmax')(final_dense1)  # 10 classes for CIFAR-10

    # Construct the model
    model = Model(inputs=input_layer, outputs=final_output)

    return model