import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Concatenate, Conv2DNormBatchActiv, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main pathway
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(1, 3), activation='relu')(input_layer)
    conv1_3 = Conv2D(filters=32, kernel_size=(3, 1), activation='relu')(input_layer)
    concat = Concatenate()([conv1_1, conv1_2, conv1_3])
    conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(concat)
    
    # Parallel branches
    conv2_2 = Conv2D(filters=64, kernel_size=(1, 3), activation='relu')(input_layer)
    conv2_3 = Conv2D(filters=64, kernel_size=(3, 1), activation='relu')(input_layer)
    conv2_4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    concat_branch = Concatenate()([conv2_2, conv2_3, conv2_4])
    
    # Addition
    add_layer = Add()([conv2_1, concat_branch])
    
    # Final 1x1 convolution
    conv3_1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(add_layer)
    
    # Classification head
    flatten = Flatten()(conv3_1)
    dense1 = Dense(units=512, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    
    model = Model(inputs=input_layer, outputs=dense2)
    
    return model

# Create the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Assuming you have a trainable validation dataset 'val_images' and 'val_labels'
# and a train dataset 'train_images' and 'train_labels'
model.fit(train_images, train_labels, validation_data=(val_images, val_labels), epochs=10)