import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution, 3x3 convolution
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(input_layer)
    branch1 = Dropout(0.5)(conv1_2)

    # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution
    conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    conv2_2 = Conv2D(filters=64, kernel_size=(1, 7), padding='same')(input_layer)
    conv2_3 = Conv2D(filters=64, kernel_size=(7, 1), padding='same')(input_layer)
    conv2_4 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(input_layer)
    branch2 = Dropout(0.5)(Concatenate()([conv2_2, conv2_3, conv2_4]))

    # Branch 3: Max Pooling
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    branch3 = Dropout(0.5)(maxpool)

    # Concatenate the outputs from all branches
    concat = Concatenate()([branch1, branch2, branch3])

    # Process through three fully connected layers
    dense1 = Dense(units=128, activation='relu')(concat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])