import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Three MaxPooling layers with varying window sizes
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_layer)

    # Concatenate the outputs from the three pooling layers
    pooled_outputs = Concatenate()(inputs=[pool1, pool2, pool3])

    # Flatten the concatenated features
    flattened = Flatten()(pooled_outputs)

    # Two fully connected layers
    dense1 = Dense(units=256, activation='relu')(flattened)
    dense2 = Dense(units=128, activation='relu')(dense1)

    # Output layer with 10 classes
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Construct the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])