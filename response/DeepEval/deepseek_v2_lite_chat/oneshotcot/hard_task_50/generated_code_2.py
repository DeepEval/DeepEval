import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Dropout
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    block1_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1_output = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(block1_output)
    block1_output = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block1_output)
    block1_output = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block1_output)
    block1_output = Flatten()(block1_output)
    block1_output = Dropout(rate=0.5)(block1_output)  # Apply dropout to mitigate overfitting
    
    # Split into four groups
    split1 = Lambda(lambda x: keras.backend.split(x, num_or_size_splits=4, axis=-1))(block1_output)
    
    # Block 2
    separable_convs = [
        (1, 1, 1, 3, 3, 64, 'relu', 'same'),
        (1, 1, 2, 5, 5, 64, 'relu', 'same'),
        (1, 1, 3, 7, 7, 64, 'relu', 'same'),
        (1, 1, 4, 4, 4, 64, 'relu', 'same')
    ]
    
    conv_outputs = []
    for (kernel_size, filters, activation, padding) in separable_convs:
        conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, activation=activation)(block1_output)
        conv = keras.layers.DepthwiseConv2D(kernel_size=kernel_size)(conv)
        conv = keras.layers.Conv2D(filters=filters, kernel_size=1, padding="same", activation=activation)(conv)
        conv_outputs.append(conv)
    
    block2_output = Concatenate()(conv_outputs)
    block2_output = Flatten()(block2_output)
    block2_output = Dropout(rate=0.5)(block2_output)  # Apply dropout before the final dense layers
    
    output_layer = Dense(units=10, activation='softmax')(block2_output)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model