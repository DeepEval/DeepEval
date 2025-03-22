import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Path 1
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Path 2
    avg_pool = AveragePooling2D(pool_size=(2, 2))(input_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(avg_pool)

    # Path 3
    conv1_3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    conv1_3_1 = Conv2D(filters=64, kernel_size=(1, 3), activation='relu')(conv1_3)
    conv1_3_2 = Conv2D(filters=64, kernel_size=(3, 1), activation='relu')(conv1_3)
    path3_output = Concatenate()([conv1_3_1, conv1_3_2])

    # Path 4
    conv1_4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    conv1_4_1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1_4)
    conv1_4_2 = Conv2D(filters=64, kernel_size=(1, 3), activation='relu')(conv1_4_1)
    conv1_4_3 = Conv2D(filters=64, kernel_size=(3, 1), activation='relu')(conv1_4_1)
    path4_output = Concatenate()([conv1_4_2, conv1_4_3])

    # Concatenate outputs from all paths
    multi_scale_output = Concatenate()([conv1_1, conv1_2, path3_output, path4_output])

    flatten_layer = Flatten()(multi_scale_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model