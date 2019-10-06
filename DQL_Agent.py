import numpy as np
import pandas as pd
import pygame
import cv2
import tensorflow as tf

class DQL(object):
    def __init__(self):
        self.gamma = 0.9
        self.model = self.network()
        self.memory = list()

    def get_state(self, screen):
        arr = pygame.surfarray.array2d(screen)
        arr = np.array(arr)

        return arr

    def network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape = (28,28,1)),
            tf.keras.layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
            tf.keras.layers.MaxPool2D(pool_size = (2,2)),
            tf.keras.layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'),
            tf.keras.layers.Conv2D(128, kernel_size = (3,3), activation = 'relu'),
            tf.keras.layers.MaxPool2D(pool_size = (2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation = 'relu'),
            tf.keras.layers.Dense(3, activation = 'softmax')
        ])

        model.compile(optimizer = 'adam',
                      loss = 'sparse_categorical_crossentropy',
                      metrics = ['accuracy'])

        return model

    def replay_new(self, memory): pass # not done
    def remember(self, tuple):
        self.memory.append(tuple)

    def train_short_memory(self,state, action, reward, new_state, scratch):
        target = reward
        if(not scratch):
            target = target + self.gamma*self.model.predict(new_state)

    def set_reward(self): pass # note done

