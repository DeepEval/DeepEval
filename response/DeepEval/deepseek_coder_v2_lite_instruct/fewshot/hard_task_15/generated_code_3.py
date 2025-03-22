import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # Input shape for CIFAR-10

    # Main path
    main_conv = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    main_avg_pool = GlobalAveragePooling2D()(main_conv)
    main_dense1 = Dense(units=64, activation='relu')(main_avg_pool)
    main_dense2 = Dense(units=32, activation='relu')(main_dense1)
    main_weights = Dense(units=32*32*3, activation='sigmoid')(main_dense2)
    main_weights = keras.backend.reshape(main_weights, (32, 32, 3))
    main_weighted = Multiply()([input_layer, main_weights])

    # Branch path (identity)
    branch_path = input_layer

    # Addition of main path and branch path
    added = Add()([main_weighted, branch_path])

    # Final layers
    flatten_layer = Flatten()(added)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()  # To print the model architecture