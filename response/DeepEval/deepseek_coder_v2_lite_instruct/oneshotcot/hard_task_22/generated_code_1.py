import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, BatchNormalization, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    main_path = []
    for kernel_size in [(1, 1), (3, 3), (5, 5)]:
        conv = Conv2D(filters=32, kernel_size=kernel_size, padding='same', activation='relu')(input_layer)
        main_path.append(conv)
    
    main_output = Concatenate(axis=-1)(main_path)

    # Branch Path
    branch_output = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse Paths
    fused_output = tf.add(main_output, branch_output)

    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(fused_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])