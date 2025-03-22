import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split input into three groups
    split1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    split1_1x1, split1_3x3, split1_5x5 = split1

    # Different branches for feature extraction
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split1_1x1)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(split1_3x3)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch2)
    branch3 = MaxPooling2D()(split1_5x5)

    # Concatenate outputs from different branches
    concat = Concatenate()( [branch1, branch2, branch3] )

    # Process through blocks
    block1 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(concat)
    block1 = BatchNormalization()(block1)
    block1 = MaxPooling2D()(block1)
    block2 = GlobalAveragePooling2D()(block1)

    # Fully connected layers for classification
    output_layer = Dense(units=10, activation='softmax')(block2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model