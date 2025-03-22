import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # First block
    split_input = tf.split(input_layer, 3, axis=-1)
    conv_1x1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(split_input[0])
    conv_3x3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(split_input[1])
    conv_5x5 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(split_input[2])
    concat_block1 = Concatenate()([conv_1x1, conv_3x3, conv_5x5])
    bn_block1 = BatchNormalization()(concat_block1)
    
    # Second block
    conv_1x1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(bn_block1)
    conv_3x3_1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(bn_block1)
    conv_1x7_7x1_3x3_1 = Conv2D(filters=32, kernel_size=(1, 7), padding='same', activation='relu')(bn_block1)
    avg_pool_1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(bn_block1)
    concat_block2 = Concatenate()([conv_1x1_1, conv_3x3_1, conv_1x7_7x1_3x3_1, avg_pool_1])
    
    # Fully connected layers
    flatten_layer = Flatten()(concat_block2)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
# model.fit(train_images, train_labels, epochs=10)

# # Evaluate the model
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print('Test accuracy:', test_acc)