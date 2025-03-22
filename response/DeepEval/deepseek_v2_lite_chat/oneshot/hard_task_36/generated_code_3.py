import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():

    # Pathway 1
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    drop = Dropout(0.5)(max_pool)  # 50% dropout to mitigate overfitting

    # Pathway 2
    branch_input = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(drop)

    # Concatenate outputs from both pathways
    concat = Concatenate(axis=-1)([max_pool, branch_input])

    # Block sequence for main pathway
    block = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat)
    block = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block)
    block = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block)
    block = Dropout(0.5)(block)  # 50% dropout to mitigate overfitting

    # Global average pooling and fully connected layers for main pathway
    main_output = Flatten()(block)
    main_output = Dense(units=128, activation='relu')(main_output)
    main_output = Dense(units=10, activation='softmax')(main_output)

    # Branch pathway
    branch_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_input)

    # Concatenate outputs from both pathways
    fused_output = Add()([main_output, branch_output])

    # Flattening, fully connected layers for branch pathway
    branch_output = Flatten()(fused_output)
    branch_output = Dense(units=128, activation='relu')(branch_output)
    branch_output = Dense(units=10, activation='softmax')(branch_output)

    # Construct the model
    model = keras.Model(inputs=[input_layer, branch_input], outputs=[main_output, branch_output])

    return model

model = dl_model()