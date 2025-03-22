from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

def identity_block(input_tensor, filters):
    # Main path
    conv_a = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(1, 1), padding='valid')(input_tensor)
    bn_a = layers.BatchNormalization()(conv_a)
    act_a = layers.Activation('relu')(bn_a)

    conv_b = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(act_a)
    bn_b = layers.BatchNormalization()(conv_b)
    act_b = layers.Activation('relu')(bn_b)

    conv_c = layers.Conv2D(filters=4*filters, kernel_size=(1, 1), strides=(1, 1), padding='valid')(act_b)
    bn_c = layers.BatchNormalization()(conv_c)

    # Branch path
    conv_shortcut = layers.Conv2D(filters=4*filters, kernel_size=(1, 1), strides=(1, 1), padding='valid')(input_tensor)
    bn_shortcut = layers.BatchNormalization()(conv_shortcut)

    # Output
    output_tensor = layers.add([bn_c, bn_shortcut])
    output_tensor = layers.Activation('relu')(output_tensor)

    return output_tensor

def basic_block(input_tensor, filters, s=1):
    # Shortcut path
    conv_shortcut = layers.Conv2D(filters=4*filters, kernel_size=(1, 1), strides=(s, s), padding='valid')(input_tensor)
    bn_shortcut = layers.BatchNormalization()(conv_shortcut)

    # Main path
    conv_a = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=(s, s), padding='valid')(input_tensor)
    bn_a = layers.BatchNormalization()(conv_a)
    act_a = layers.Activation('relu')(bn_a)

    conv_b = layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(act_a)
    bn_b = layers.BatchNormalization()(conv_b)
    act_b = layers.Activation('relu')(bn_b)

    conv_c = layers.Conv2D(filters=4*filters, kernel_size=(1, 1), strides=(1, 1), padding='valid')(act_b)
    bn_c = layers.BatchNormalization()(conv_c)

    # Output
    output_tensor = layers.add([bn_c, bn_shortcut])
    output_tensor = layers.Activation('relu')(output_tensor)

    return output_tensor

def resnet_model():
    # Input tensor
    img_input = layers.Input(shape=(32, 32, 3))

    # Stage 1
    x = layers.ZeroPadding2D((3, 3))(img_input)
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Stage 2
    x = layers.ZeroPadding2D((1, 1))(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = basic_block(x, filters=16)

    # Stage 3
    x = basic_block(x, filters=16)
    x = basic_block(x, filters=16)

    # Global branch
    pool_g = layers.MaxPooling2D((4, 4), strides=(4, 4))(img_input)
    conv_g = layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid')(pool_g)

    # Second-level residual structure
    res_level_2 = basic_block(x, filters=32, s=2)
    res_level_2 = basic_block(res_level_2, filters=32)

    # Third-level residual structure
    res_level_3 = basic_block(res_level_2, filters=64, s=2)
    res_level_3 = basic_block(res_level_3, filters=64)
    res_level_3 = basic_block(res_level_3, filters=64)

    # Feature fusion
    output = layers.add([res_level_3, conv_g])

    # Output layer
    x = layers.AveragePooling2D((4, 4))(output)
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation='softmax')(x)

    # Model
    model = models.Model(inputs=img_input, outputs=x)

    return model

model = resnet_model()