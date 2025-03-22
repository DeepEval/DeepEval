import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Add
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block1(x):
        # Split the input into three groups
        x1 = Conv2D(64, (1, 1), activation='relu')(x)
        x1 = Conv2D(64, (3, 3), activation='relu')(x1)
        x1 = Conv2D(64, (1, 1), activation='relu')(x1)

        x2 = Conv2D(64, (1, 1), activation='relu')(x)
        x2 = Conv2D(64, (3, 3), activation='relu')(x2)
        x2 = Conv2D(64, (1, 1), activation='relu')(x2)

        x3 = Conv2D(64, (1, 1), activation='relu')(x)
        x3 = Conv2D(64, (3, 3), activation='relu')(x3)
        x3 = Conv2D(64, (1, 1), activation='relu')(x3)

        # Concatenate outputs from the three groups
        x = Concatenate()([x1, x2, x3])
        return x

    # Apply Block 1
    x = block1(input_layer)

    # Transition convolution to adjust the number of channels
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = BatchNormalization()(x)

    # Block 2 with global max pooling and reshaping weights
    x_pool = MaxPooling2D((8, 8))(x)
    x_pool = Flatten()(x_pool)
    weights = Dense(128, activation='relu')(x_pool)
    weights = Dense(tf.keras.backend.int_shape(x)[-1], activation='sigmoid')(weights)
    weights = Lambda(lambda x: tf.linalg.diag(tf.squeeze(x)))(weights)  # Reshape weights to match the shape of x
    x_main = tf.matmul(x, weights)  # Multiply weights with the adjusted output

    # Branch directly connected to the input
    x_branch = input_layer

    # Add the main path and the branch
    x = Add()([x_main, x_branch])

    # Flatten and fully connected layers for classification
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(10, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model