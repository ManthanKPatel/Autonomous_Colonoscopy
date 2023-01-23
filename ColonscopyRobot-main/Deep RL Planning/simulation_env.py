import os
import copy
import numpy as np

import pygame
from pygame.locals import *

from utils import bresenham

# Environment Parameters
# Sensor Parameters
DRAW_SENSOR = True
SENSOR_RANGE = 100
SENSOR_WINDOW = (-np.pi/6, np.pi/6)
SENSOR_RESOLUTION = 0.1
# Robot Setting
WIDTH, HEIGHT = 700, 700
START_X, START_Y, START_ANG = 320.0, 500.0,-np.pi/2
SIZE_ROBOT = 10

# State Action
ACTION_DELTA_VELOCITY = 0.5
ACTION_DELTA_ANG_VEL = 0.05 

# Robot Maximum Speed
MAXIMUM_VELOCITY = 10
MAXIMUM_ANG_VELOCITY = 2.0

pygame.init()

class PathPlanningEnv:
    def __init__(self):
        # load PyGame resources
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        self.colon_img = pygame.image.load(os.path.join(os.getcwd(), 'resources/colon.png'))
        self.window.blit(self.colon_img, (0, 0))
        pygame.display.set_caption("Colon Path Planning")

        # Colonscopy State
        self.dt = 0.01
        self.u = np.array([0.0, 0.0])
        self.x = np.array([START_X, START_Y, START_ANG])
        self.x_prior = np.array([START_X, START_Y, START_ANG])

        # Public properties
        self.state_size = round(((-1*SENSOR_WINDOW[0] + SENSOR_WINDOW[1]) / SENSOR_RESOLUTION + 2)*2 + 3 + 4) - 3       # Sensor Window + Robot State + Control Actions
        self.action_size = 4                                                                                            # four actions [Left, Right, Up, Down] --> [0, 1, 2, 3]

    def render(self):
        self.window.blit(self.colon_img, (0, 0))
        self.__drawRobot()
        pygame.display.update()

    def step(self, action):
        if action == 0:
            if (self.u[1] > MAXIMUM_ANG_VELOCITY*-1.0):
                self.u[1] -= ACTION_DELTA_ANG_VEL  
                print("left button")
            else:
                print("Env: Maximum Left Speed Reached!")
        elif action == 1:
            if (self.u[1] < MAXIMUM_ANG_VELOCITY):
                self.u[1] += ACTION_DELTA_ANG_VEL
                print("right button")
            else:
                print("Env: Maximum Right Speed Reached!")
        elif action == 2:
            if (self.u[0] < MAXIMUM_VELOCITY):
                self.u[0] += ACTION_DELTA_VELOCITY
                print("up button")
            else:
                print("Env: Maximum Foward Speed Reached!")
        elif action == 3:
            if (self.u[0] > MAXIMUM_VELOCITY * -1.0):
                self.u[0] -= ACTION_DELTA_VELOCITY
                print("down button")
            else:
                print("Env: Maximum Reverse Speed Reached!")
        else:
            print("ERROR: Bad Action From Agent!")

        state = copy.deepcopy(np.concatenate((self.meas, self.x), axis=None))
        state = copy.deepcopy(np.concatenate((state, self.u), axis=None))
        self.__motionModel()
        self.__measurementModel()
        self.render()
        next_state = np.concatenate((self.meas, self.x), axis=None)
        next_state = np.concatenate((next_state, self.u), axis=None)
        reward = self.__reward()
        done = self.__collision()
        info = None
        return action, state, reward, next_state, done

    def reset(self):
        self.x = np.array([START_X, START_Y, START_ANG])
        self.u = np.array([0.0, 0.0])
        self.__motionModel()
        self.__measurementModel()
        self.render()
        state = np.concatenate((self.meas, self.x), axis=None)
        state = np.concatenate((state, self.u), axis=None)
        return state

    def close(self):
        pygame.quit()

    def __motionModel(self):
        self.x[0] = self.x[0] + self.dt * np.cos(self.x[2]) * self.u[0]
        self.x[1] = self.x[1] + self.dt * np.sin(self.x[2]) * self.u[0]
        self.x[2] = self.x[2] + self.dt * self.u[1]

    def __measurementModel(self):
        self.meas = np.array([[0.0, 0.0]])
        step_angle = SENSOR_WINDOW[0]
        while step_angle < 0.0:
            a = self.x[2] + step_angle
            r = self.__raySearch(a)
            self.meas = np.concatenate((self.meas, np.array([[a, r]])), axis=0)
            step_angle += SENSOR_RESOLUTION
        step_angle = 0.0
        while step_angle < SENSOR_WINDOW[1]:
            a = self.x[2] + step_angle
            r = self.__raySearch(a)
            self.meas = np.concatenate((self.meas, np.array([[a, r]])), axis=0)
            step_angle += SENSOR_RESOLUTION
        self.meas = self.meas[:][1:]

    def __reward(self):
        return -1.0

    def __drawRobot(self):
        if DRAW_SENSOR:
            for measurement in self.meas:
                x = self.x[0] + np.cos(measurement[0]) * measurement[1]
                y = self.x[1] + np.sin(measurement[0]) * measurement[1]
                pygame.draw.line(self.window, (255, 255, 0), (self.x[0], self.x[1]),(x, y), width = 2)
        radius = 30
        x = self.x[0] + np.cos(self.x[2]) * radius
        y = self.x[1] + np.sin(self.x[2]) * radius
        pygame.draw.circle(self.window, (255, 0, 0), (self.x[0], self.x[1]), radius = SIZE_ROBOT, width = 0)
        pygame.draw.line(self.window, (255, 0, 0), (self.x[0], self.x[1]), (x, y), width = 3)

    def __collision(self):
        ang = 0
        while ang <= np.pi * 2:
            x = self.x[0] + np.cos(ang) * SIZE_ROBOT
            y = self.x[1] + np.sin(ang) * SIZE_ROBOT
            if self.colon_img.get_at((int(x), int(y))) == (0, 0, 0, 255):
                return True
            ang += 0.1
        return False

    def __raySearch(self, angle, searchRadius=SENSOR_RANGE):
        x = self.x[0] + np.cos(angle) * searchRadius
        y = self.x[1] + np.sin(angle) * searchRadius
        x_coords, y_coords = bresenham(self.x[0], self.x[1], x, y)
        x_edge, y_edge = 0, 0 
        for i, x in enumerate(x_coords):
            x_edge, y_edge = x_coords[i], y_coords[i]
            if self.colon_img.get_at((int(x_edge), int(y_edge))) == (0, 0, 0, 255):
                break
        r = ((self.x[0] - x_edge)**2 + (self.x[1] - y_edge)**2)**0.5
        return r

    def __userControl(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.u[1] -= ACTION_DELTA_ANG_VEL
                if event.key == pygame.K_RIGHT:
                    self.u[1] += ACTION_DELTA_ANG_VEL
                if event.key == pygame.K_UP:
                    self.u[0] += ACTION_DELTA_VELOCITY
                if event.key == pygame.K_DOWN:
                    self.u[0] -= ACTION_DELTA_VELOCITY
                

    def test(self):
        while True:
            self.u = np.array([0.0, 0.0]) 
            for i in range(10000):
                self.__userControl()
                self.__motionModel()
                self.__measurementModel()
                if self.__collision():
                    break
                self.render()
            self.reset()

def test():
    env = PathPlanningEnv()
    env.test()

if __name__ == '__main__':
    test()
