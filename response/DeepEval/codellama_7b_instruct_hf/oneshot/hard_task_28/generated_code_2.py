import keras
from keras.layers import Input, DepthwiseConv2D, PointwiseConv2D, LayerNormalization, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    main_path = input_layer
    branch_path = input_layer

    # Main path
    main_path = DepthwiseConv2D(kernel_size=(7, 7), padding='same')(main_path)
    main_path = LayerNormalization()(main_path)
    main_path = PointwiseConv2D(filters=main_path.shape[1], kernel_size=(1, 1))(main_path)
    main_path = PointwiseConv2D(filters=main_path.shape[1], kernel_size=(1, 1))(main_path)

    # Branch path
    branch_path = PointwiseConv2D(filters=branch_path.shape[1], kernel_size=(1, 1))(branch_path)
    branch_path = PointwiseConv2D(filters=branch_path.shape[1], kernel_size=(1, 1))(branch_path)

    # Combine main and branch paths
    output = Add()([main_path, branch_path])

    # Flatten output
    output = Flatten()(output)

    # Process output through two fully connected layers
    output = Dense(units=128, activation='relu')(output)
    output = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output)
    return model