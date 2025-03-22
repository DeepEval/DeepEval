import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, Dropout, Flatten, Dense

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # Split input into three groups
    split1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    split2 = Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)

    # Convolution and Pooling
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split1[0])
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split1[1])
    conv1_3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split1[2])
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(tf.concat([conv1_1, conv1_2, conv1_3], axis=-1))

    # Repeat similar operations for split2
    conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split2[0])
    conv2_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split2[1])
    conv2_3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split2[2])
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(tf.concat([conv2_1, conv2_2, conv2_3], axis=-1))

    # Dropout for feature selection
    drop1 = Dropout(rate=0.5)(pool1)
    drop2 = Dropout(rate=0.5)(pool2)

    # Concatenate outputs from different groups
    concat = Concatenate(axis=-1)([drop1, drop2])

    # Additional pathway with 1x1 convolution
    conv3_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(concat)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3_1)

    # Flatten and Fully connected layers
    flatten = Flatten()(pool3)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)

    # Model
    model = Model(inputs=inputs, outputs=output)

    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()