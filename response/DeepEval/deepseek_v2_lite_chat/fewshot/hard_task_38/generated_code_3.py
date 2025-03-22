import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Concatenate, MaxPooling2D, Flatten, Dense

def dl_model():
    # Pathway 1: Three 3x3 convolutional layers with max-pooling
    pathway1_input = Input(shape=(28, 28, 1))
    pathway1_blocks = [
        Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation=ReLU()),
        Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=ReLU()),
        Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=ReLU())
    ]
    pathway1_blocks = [block(input_tensor) for block in pathway1_blocks]
    pathway1_output = Concatenate()(pathway1_blocks)

    # Pathway 2: Three 3x3 convolutional layers with max-pooling
    pathway2_input = Input(shape=(28, 28, 1))
    pathway2_blocks = [
        Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation=ReLU()),
        Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=ReLU()),
        Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=ReLU())
    ]
    pathway2_blocks = [block(input_tensor) for block in pathway2_blocks]
    pathway2_output = Concatenate()(pathway2_blocks)

    # Merge Pathways
    merged_output = Concatenate()([pathway1_output, pathway2_output])

    # Fully Connected Layers for Classification
    dense1 = Dense(units=128, activation='relu')(merged_output)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=[pathway1_input, pathway2_input], outputs=dense2)

    return model

# Define a function to create a single block
def block(input_tensor):
    bn = BatchNormalization()(input_tensor)
    conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(bn)
    return ReLU()(conv)

# Construct the deep learning model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])