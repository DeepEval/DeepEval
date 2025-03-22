import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Step 2: Global Average Pooling
    gap = GlobalAveragePooling2D()(input_layer)

    # Step 3: Fully Connected Layers to generate weights
    fc1 = Dense(units=128, activation='relu')(gap)
    fc2 = Dense(units=64, activation='relu')(fc1)

    # Step 4: Reshape and Multiply weights with input feature map
    weights = Reshape((32, 32, 3))(fc2)
    weighted_input = Multiply()([input_layer, weights])

    # Step 5: Flatten and Output
    flatten = Flatten()(weighted_input)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model