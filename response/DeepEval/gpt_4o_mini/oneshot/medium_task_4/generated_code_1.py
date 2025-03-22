import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Add
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Pathway 1
    path1_block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path1_block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1_block1)
    path1_avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(path1_block2)
    
    # Pathway 2
    path2_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine pathways with addition
    combined_output = Add()([path1_avg_pool, path2_conv])

    # Flatten the combined output
    flatten_layer = Flatten()(combined_output)
    
    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model