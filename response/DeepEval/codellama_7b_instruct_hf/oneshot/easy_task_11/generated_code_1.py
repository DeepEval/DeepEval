from keras.layers import Input, AveragePooling2D, Conv2D, Flatten, Dense, Dropout
from keras.models import Model

，增加dl_model 和 return model
def dl_model():
    # Define the input shape and number of classes
    input_shape = (28, 28, 1)
    num_classes = 10

    # Define the model architecture
    input_layer = Input(shape=input_shape)
    avg_pool = AveragePooling2D(pool_size=(5, 5), strides=3)(input_layer)
    conv = Conv2D(32, (1, 1), activation='relu')(avg_pool)
    flatten = Flatten()(conv)
    dense1 = Dense(64, activation='relu')(flatten)
    dropout = Dropout(0.5)(dense1)
    dense2 = Dense(64, activation='relu')(dropout)
    output_layer = Dense(num_classes, activation='softmax')(dense2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model with a loss function and optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model