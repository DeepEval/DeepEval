import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Add, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    global_avg_pool = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=512, activation='relu')(global_avg_pool)
    dense2 = Dense(units=32 * 32 * 3, activation='sigmoid')(dense1)  # Output shape to match input layer
    weights = Reshape((32, 32, 3))(dense2)  # Reshape to match input shape
    main_path = Add()([input_layer, weights])  # Element-wise addition with input layer

    # Branch path
    branch_path = input_layer  # Directly connected to input layer without modification

    # Combine both paths
    combined = Add()([main_path, branch_path])

    # Fully connected layers for classification
    flatten_layer = Flatten()(combined)
    dense3 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)  # CIFAR-10 has 10 classes

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model