import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Lambda, Concatenate, BatchNormalization, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def block1(x):
        # Main path
        x1 = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x1 = Conv2D(32, (3, 3), padding='same', activation='relu')(x1)
        
        # Branch path
        x2 = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
        
        # Add paths
        x = Add()([x1, x2])
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        return x

    x = block1(input_layer)
    x = block1(x)

    # Second block
    def block2(x):
        # Split input into three groups
        splits = Lambda(lambda z: tf.split(z, num_or_size_splits=3, axis=3))(x)
        
        # Process each group with depthwise separable convolution
        def depthwise_conv_block(s, kernel):
            return Conv2D(1, (kernel, kernel), padding='same', activation='relu', depthwise_mode=True)(s)
        
        outputs = [depthwise_conv_block(s, 1) for s in splits]
        outputs += [depthwise_conv_block(s, 3) for s in splits]
        outputs += [depthwise_conv_block(s, 5) for s in splits]
        
        # Concatenate outputs
        x = Concatenate()(outputs)
        return x

    x = block2(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model