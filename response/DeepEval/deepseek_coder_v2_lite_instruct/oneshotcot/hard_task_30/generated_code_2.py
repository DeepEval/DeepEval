import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense, Lambda, Concatenate
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First block
    def first_block(x):
        # Main path
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        
        # Branch path
        branch = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
        
        # Addition of main and branch paths
        x = Add()([x, branch])
        return x

    x = first_block(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Second block
    def second_block(x):
        # Split the input into three groups
        splits = Lambda(lambda z: tf.split(z, 3, axis=-1))(x)
        
        # Extract features using depthwise separable convolutions
        outputs = []
        for i, split in enumerate(splits):
            if i == 0:
                out = Conv2D(32, (1, 1), padding='same', activation='relu')(split)
                out = Conv2D(32, (3, 3), padding='same', activation='relu')(out)
            elif i == 1:
                out = Conv2D(32, (1, 1), padding='same', activation='relu')(split)
                out = Conv2D(32, (3, 3), padding='same', activation='relu')(out)
            elif i == 2:
                out = Conv2D(32, (1, 1), padding='same', activation='relu')(split)
                out = Conv2D(32, (3, 3), padding='same', activation='relu')(out)
            outputs.append(out)
        
        # Concatenate the outputs
        x = Concatenate()(outputs)
        return x

    x = second_block(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(10, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model