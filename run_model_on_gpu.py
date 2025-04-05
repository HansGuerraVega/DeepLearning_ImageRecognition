
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def load_train(path):

    labels = pd.read_csv(f'{path}/labels.csv')
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen_flow = datagen.flow_from_dataframe(
        dataframe=labels,
        directory=f'{path}/final_files',
        x_col="file_name",
        y_col="real_age",
        target_size=(128, 128),
        batch_size=32,
        class_mode="raw",
        subset="training"
    )
    return train_gen_flow


def load_test(path):

    labels = pd.read_csv(f'{path}/labels.csv')
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    test_gen_flow = datagen.flow_from_dataframe(
        dataframe=labels,
        directory=f'{path}/final_files',
        x_col="file_name",
        y_col="real_age",
        target_size=(128, 128),
        batch_size=32,
        class_mode="raw",
        subset="validation"
    )
    return test_gen_flow


def create_model(input_shape=(128, 128, 3)):

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.2),
        Dense(256, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
    return model


def train_model(model, train_data, test_data, batch_size=32, epochs=20,
                steps_per_epoch=None, validation_steps=None):

    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)

    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1
    )
    return model, history


