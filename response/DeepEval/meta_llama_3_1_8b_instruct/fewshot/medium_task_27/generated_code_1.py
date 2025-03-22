import keras
from keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 3x3 Convolutional Layer
    branch1_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 2: 5x5 Convolutional Layer
    branch2_conv = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine the outputs of the two branches using addition
    adding_layer = Add()([branch1_conv, branch2_conv])

    # Global Average Pooling
    global_avg_pool = GlobalAveragePooling2D()(adding_layer)

    # Fully Connected Layer 1
    dense1 = Dense(units=64, activation='relu')(global_avg_pool)

    # Fully Connected Layer 2
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Softmax Layer
    attention_weights1 = Dense(units=10, activation='softmax')(dense1)
    attention_weights2 = Dense(units=10, activation='softmax')(dense2)

    # Weighted Sum of the Outputs
    weighted_output1 = Add()([global_avg_pool, Multiply()([global_avg_pool, attention_weights1])])
    weighted_output2 = Add()([global_avg_pool, Multiply()([global_avg_pool, attention_weights2])])

    # Final Output Layer
    output_layer = Dense(units=10, activation='softmax')(Add()([weighted_output1, weighted_output2]))

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model