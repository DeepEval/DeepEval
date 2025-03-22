import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, SeparableConv2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block(input_tensor):
        # Separable Convolutional Layer
        separable_conv = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return separable_conv

    # Main Path
    main_path = block(input_tensor=input_layer)
    for _ in range(3):
        main_path = block(input_tensor=main_path)
        main_path = Concatenate(axis=-1)([main_path, block(input_tensor=main_path)])

    # Branch Path
    branch_path = Conv2D(filters=main_path.shape[-1], kernel_size=(3, 3), strides=(1, 1), padding='same')(main_path)

    # Fusion of Main and Branch Paths
    fused_features = keras.layers.add([main_path, branch_path])

    # Flatten the features
    flattened_features = Flatten()(fused_features)

    # Fully Connected Layer
    output_layer = Dense(units=10, activation='softmax')(flattened_features)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])