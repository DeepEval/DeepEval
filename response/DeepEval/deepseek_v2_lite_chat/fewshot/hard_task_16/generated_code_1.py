import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Add, Concatenate
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block1(x):
        x = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(x)  # 1x1 convolution
        x = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x)   # 3x3 convolution
        x = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(x)  # 1x1 convolution
        pool1 = MaxPooling2D(pool_size=(2, 2))(x)  # Pooling
        return pool1

    block1_output = block1(input_layer)
    
    # Transition layer
    x = Lambda(lambda tensors: tf.split(tensors, num_or_size_splits=3, axis=-1))(
        [block1_output, block1_output, block1_output]
    )
    x[0] = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(x[0])  # 1x1 convolution
    x[1] = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(x[1])   # 3x3 convolution
    x[2] = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(x[2])  # 1x1 convolution
    pool1 = MaxPooling2D(pool_size=(3, 3))(x[0])  # Pooling

    # Block 2
    x = Flatten()(pool1)
    x = Dense(units=128)(x)  # Fully connected layer
    x = Dense(units=64)(x)   # Fully connected layer
    x = Dense(units=10)(x)   # Output layer

    main_branch = Dense(units=10)(x)  # Fully connected layer for the branch

    # Concatenate the main path and branch outputs
    x = Concatenate()([main_branch, x])

    # Final fully connected layer for classification
    x = Dense(units=10, activation='softmax')(x)

    # Model
    model = Model(inputs=input_layer, outputs=x)

    return model

# Create the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()