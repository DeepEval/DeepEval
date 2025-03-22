from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the multi-branch convolutional structure
    path1 = Conv2D(filters=32, kernel_size=1, activation='relu')(input_shape)
    path2 = Conv2D(filters=32, kernel_size=1, activation='relu')(input_shape)
    path3 = Conv2D(filters=32, kernel_size=1, activation='relu')(input_shape)
    path4 = MaxPooling2D(pool_size=2)(input_shape)
    path5 = Conv2D(filters=32, kernel_size=1, activation='relu')(path4)
    path6 = Conv2D(filters=32, kernel_size=7, activation='relu')(path5)
    path7 = Conv2D(filters=32, kernel_size=7, activation='relu')(path6)
    path8 = Conv2D(filters=32, kernel_size=1, activation='relu')(path7)
    path9 = Conv2D(filters=32, kernel_size=7, activation='relu')(path8)
    path10 = Conv2D(filters=32, kernel_size=7, activation='relu')(path9)
    path11 = Conv2D(filters=32, kernel_size=1, activation='relu')(path10)

    # Define the fusion layer
    fusion_layer = Concatenate(axis=-1)([path1, path2, path3, path4, path5, path6, path7, path8, path9, path10, path11])

    # Define the fully connected layer
    fc_layer = Dense(units=10, activation='softmax')(fusion_layer)

    # Define the output shape
    output_shape = fc_layer.shape[1:]

    # Define the model
    model = Model(inputs=input_shape, outputs=fc_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model