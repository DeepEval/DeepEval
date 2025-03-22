from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    inputs = keras.Input(shape=(32, 32, 3)) 

    # Branch 1
    branch1_conv1 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch1_conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(branch1_conv1)
    branch1_dropout = layers.Dropout(0.2)(branch1_conv2)

    # Branch 2
    branch2_conv1 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    branch2_conv2 = layers.Conv2D(64, (1, 7), activation='relu', padding='same')(branch2_conv1)
    branch2_conv3 = layers.Conv2D(64, (7, 1), activation='relu', padding='same')(branch2_conv2)
    branch2_conv4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(branch2_conv3)
    branch2_dropout = layers.Dropout(0.2)(branch2_conv4)

    # Branch 3
    branch3_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(inputs)
    branch3_conv = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(branch3_pool)
    branch3_dropout = layers.Dropout(0.2)(branch3_conv)

    # Concatenate branches
    concat_output = layers.Concatenate()([branch1_dropout, branch2_dropout, branch3_dropout])

    # Flatten and dense layers
    x = layers.Flatten()(concat_output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model