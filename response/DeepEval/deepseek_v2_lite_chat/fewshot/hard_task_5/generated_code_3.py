import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Concatenate, Add, Dense, Flatten, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # Assuming the input shape is 32x32 with 3 channels

    # Block 1
    split1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    split1[0] = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(split1[0])
    split1[1] = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(split1[1])
    split1[2] = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(split1[2])
    fused_features = Concatenate(axis=-1)(split1)

    # Block 2
    reshape_input = Reshape((-1, 1))(fused_features)
    reshape_input = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(reshape_input)
    reshape_output = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(reshape_input)
    reshape_output[0] = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(reshape_output[0])
    reshape_output[1] = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(reshape_output[1])
    reshape_output[2] = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(reshape_output[2])
    shuffled_features = Concatenate(axis=-1)(reshape_output)

    # Block 3
    dw_conv1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(shuffled_features)

    # Branch path
    branch_output = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(input_layer)

    # Combine outputs from main path and branch
    combined_output = Add()([dw_conv1, branch_output])

    # Final fully connected layer for classification
    dense = Dense(units=10, activation='softmax')(combined_output)

    model = Model(inputs=input_layer, outputs=dense)

    return model

model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])