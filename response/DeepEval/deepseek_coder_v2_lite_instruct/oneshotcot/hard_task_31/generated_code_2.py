import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, Dropout, SeparableConv2D, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def block1(input_tensor):
        # Main path
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(input_tensor)
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        
        # Branch path
        y = input_tensor
        
        # Add paths
        output_tensor = Add()([x, y])
        return output_tensor

    x = block1(input_layer)
    x = Conv2D(32, (1, 1), padding='same', activation='relu')(x)  # Restore number of channels

    # Second block
    def block2(input_tensor):
        # Split input into three groups
        split_points = [1, 2]
        split_layers = []
        for i in range(3):
            if i == 0:
                split_layers.append(Lambda(lambda z: tf.split(z, split_points[i], axis=-1)[0])(input_tensor))
            elif i == 1:
                split_layers.append(Lambda(lambda z: tf.split(z, split_points[i], axis=-1)[1])(input_tensor))
            else:
                split_layers.append(Lambda(lambda z: tf.split(z, split_points[i], axis=-1)[2])(input_tensor))
        
        # Separate convolutional layers for each group
        conv_layers = []
        for i, split_layer in enumerate(split_layers):
            if i == 0:
                conv_layer = SeparableConv2D(32, (1, 1), padding='same', activation='relu')(split_layer)
                conv_layer = Dropout(0.2)(conv_layer)
                conv_layers.append(conv_layer)
            elif i == 1:
                conv_layer = SeparableConv2D(32, (3, 3), padding='same', activation='relu')(split_layer)
                conv_layer = Dropout(0.2)(conv_layer)
                conv_layers.append(conv_layer)
            else:
                conv_layer = SeparableConv2D(32, (5, 5), padding='same', activation='relu')(split_layer)
                conv_layer = Dropout(0.2)(conv_layer)
                conv_layers.append(conv_layer)
        
        # Concatenate outputs from the three groups
        output_tensor = tf.concat(conv_layers, axis=-1)
        return output_tensor

    x = block2(x)

    # Flatten and fully connected layer
    x = Flatten()(x)
    output_layer = Dense(10, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Build the model
model = dl_model()
model.summary()