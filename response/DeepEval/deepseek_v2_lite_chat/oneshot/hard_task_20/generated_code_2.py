import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    split1 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(split1[0])
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(split1[1])
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(split1[2])

    concat = Concatenate(axis=-1)([conv1, conv2, conv3])

    # Branch path
    branch_conv = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    fused_features = Add()([concat, branch_conv])

    # Batch normalization and flattening
    bn = BatchNormalization()(fused_features)
    flatten = Flatten()(bn)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Optional: Display the model summary
model.summary()