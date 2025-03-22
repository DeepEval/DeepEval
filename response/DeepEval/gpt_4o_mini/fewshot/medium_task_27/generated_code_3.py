import keras
from keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First branch: 3x3 convolution
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Second branch: 5x5 convolution
    branch2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Merging branches with addition
    merged = Add()([branch1, branch2])

    # Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(merged)

    # Fully connected layers
    dense1 = Dense(units=64, activation='relu')(global_avg_pool)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Attention weights for both branches
    attention_weights = Dense(units=2, activation='softmax')(global_avg_pool)

    # Multiplying outputs by their corresponding weights
    weighted_output_branch1 = Multiply()([branch1, attention_weights[:, 0:1]])
    weighted_output_branch2 = Multiply()([branch2, attention_weights[:, 1:2]])

    # Adding weighted outputs
    final_output = Add()([weighted_output_branch1, weighted_output_branch2])

    # Final output layer
    final_output = GlobalAveragePooling2D()(final_output)
    final_output = Dense(units=10, activation='softmax')(final_output)

    model = Model(inputs=input_layer, outputs=final_output)

    return model