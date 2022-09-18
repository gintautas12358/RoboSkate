# Here the remote control for the agent is defined!
# PyGame was used for this purpose.
# Here are also agents that play the level 1 completely automatically. For this, an optimal trajectoire is defined.

import math
import numpy as np
import pygame

class HardcodedAgents():
    """
    Hardcoded Agents for expert data collection
    """

    def __init__(self, obs, Teleoperation=False):
        super(HardcodedAgents, self).__init__()

        self.obs = obs
        self.info = {   "step": 0,
                        "board_pitch": 0,
                        "steering_angle": 0}
        self.reward = 0
        self.done = False
        self.step = 0
        self.stearing = 0
        self.keyoffset = 0
        self.level = 0
        self.backwards = False


        if Teleoperation:
            pygame.init()
            self.gameDisplay = pygame.display.set_mode((400, 400))
            pygame.display.set_caption('RoboSkate Teleoperation')


    # funktion to display information on the PyGame screen
    def RoboPos(self, x, y):

        self.gameDisplay.fill((255, 255, 255))
        font = pygame.font.SysFont('Arial', 25)

        pygame.draw.line(self.gameDisplay, (0,0,0), (200,0), (200,400), 1)
        pygame.draw.line(self.gameDisplay, (0, 0, 0), (0, 200), (400, 200), 1)
        pygame.draw.circle(self.gameDisplay, (255, 0, 0), (200-x*3,200-y*3), 15, 2)

        space = 25
        self.gameDisplay.blit(font.render("Arrow left = arm to the left", False, (0, 0, 0)), (10, space*0))
        self.gameDisplay.blit(font.render("Arrow right = arm to the right", False, (0, 0, 0)), (10 ,space*1))
        self.gameDisplay.blit(font.render("Arrow down = arm lower", False, (0, 0, 0)), (10 ,space*2))
        self.gameDisplay.blit(font.render("Arrow up = arm higher", False, (0, 0, 0)), (10 ,space*3))
        self.gameDisplay.blit(font.render("spacebar = control reset", False, (0, 0, 0)), (10 ,space*4))
        self.gameDisplay.blit(font.render("0-4 = set level", False, (0, 0, 0)), (10 ,space*5))
        self.gameDisplay.blit(font.render("x = reset", False, (0, 0, 0)), (10 ,space*6))
        self.gameDisplay.blit(font.render("r = backward", False, (0, 0, 0)), (10 ,space*7))

        pygame.display.update()

    # -----------------------------------------------------------------------------------------------------------------
    # follow a list of Checkpoints
    # -----------------------------------------------------------------------------------------------------------------

    def checkpoint_follower(self):

        # calculate steering angle
        self.steering = np.clip(self.info["steering_angle"], -70, 70)


        # ------------------------------- generate repetetiv movement ---------------------
        # scaled time to adjust movement speed
        time = self.step * 0.3

        # if the board is pitched correct the joint 2 angle
        offset_high = (self.info["board_pitch"]-0.5) * 200
        # continuous movement to move forward
        mean_angle_joint_2 = 90 - 30 + offset_high
        mean_angle_joint_3 = 125 - 70
        phase_shift = math.pi * 0.5
        range_of_motion_joint_2 = 30
        range_of_motion_joint_3 = 70

        joint2 = mean_angle_joint_2 - math.cos(time) * range_of_motion_joint_2
        joint3 = mean_angle_joint_3 - math.cos(time + phase_shift) * range_of_motion_joint_3

        # use Direction error directly for P controller of Joint 1
        joint1 = self.steering

        # P controler for actions
        # scale joint agnles to [-1, 1], calculate error, multiply with proportional parameter
        joint1_KP = 1
        joint2_KP = 1
        joint3_KP = 3  # as fast as possible to accelerate

        action1 = ((joint1 / 180) - self.obs[0]) * joint1_KP
        action2 = ((joint2 / 90) - self.obs[1]) * joint2_KP
        action3 = ((joint3 / 125) - self.obs[2]) * joint3_KP

        # limit actions to [-1, 1]
        action1 = 1 if (action1 > 1) else action1
        action1 = -1 if (action1 < -1) else action1

        action2 = 1 if (action2 > 1) else action2
        action2 = -1 if (action2 < -1) else action2

        action3 = 1 if (action3 > 1) else action3
        action3 = -1 if (action3 < -1) else action3

        # return actions
        return np.array([action1, action2, action3]).astype(np.float)



        # -----------------------------------------------------------------------------------------------------------------
        # Teleoperation
        # -----------------------------------------------------------------------------------------------------------------
    def Teleoperation(self):

        stop = False

        if pygame.key.get_pressed()[pygame.K_LEFT]:
            self.stearing += 1
        if pygame.key.get_pressed()[pygame.K_RIGHT]:
            self.stearing -= 1
        if pygame.key.get_pressed()[pygame.K_UP]:
            self.keyoffset += 1
        if pygame.key.get_pressed()[pygame.K_DOWN]:
            self.keyoffset -= 1


        # Cycles through all the events currently occuring
        for event in pygame.event.get():


            # Condition becomes true when keyboard is pressed
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.stearing = 0
                    self.keyoffset = 0
                    print("reset control")
                if event.key == pygame.K_x:
                    stop = True
                    print("reset game")
                if event.key == pygame.K_0:
                    self.level = 0
                    print("level 0 at next start")
                if event.key == pygame.K_1:
                    self.level = 1
                    print("level 1 at next start")
                if event.key == pygame.K_2:
                    self.level = 2
                    print("level 2 at next start")
                if event.key == pygame.K_3:
                    self.level = 3
                    print("level 3 at next start")
                if event.key == pygame.K_4:
                    self.level = 4
                    print("level 4 at next start")
                if event.key == pygame.K_r:
                    self.backwards = not self.backwards
                    print("backwards: " + str(self.backwards))

        self.RoboPos(self.stearing, self.keyoffset)


        # scaled time to adjust movement speed
        time = self.step * 0.3
        if self.backwards:
            time = time * (-1)

        # continuous movement to move forward
        mean_angle_joint_2 = 90 - 30 - self.keyoffset
        mean_angle_joint_3 = 125 - 70
        phase_shift = +math.pi * 0.5
        range_of_motion_joint_2 = 30
        range_of_motion_joint_3 = 70

        joint2 = mean_angle_joint_2 - math.cos(time) * range_of_motion_joint_2
        joint3 = mean_angle_joint_3 - math.cos(time + phase_shift) * range_of_motion_joint_3

        # use Direction error directly for P controller of Joint 1
        joint1 = self.stearing

        # P controler for actions
        # scale joint agnles to [-1, 1], calculate error, multiply with proportional parameter
        joint1_KP = 1
        joint2_KP = 1
        joint3_KP = 3  # as fast as possible to accelerate

        action1 = ((joint1 / 180) - self.obs[0]) * joint1_KP
        action2 = ((joint2 / 90) - self.obs[1]) * joint2_KP
        action3 = ((joint3 / 125) - self.obs[2]) * joint3_KP

        # limit actions to [-1, 1]
        action1 = 1 if (action1 > 1) else action1
        action1 = -1 if (action1 < -1) else action1

        action2 = 1 if (action2 > 1) else action2
        action2 = -1 if (action2 < -1) else action2

        action3 = 1 if (action3 > 1) else action3
        action3 = -1 if (action3 < -1) else action3

        # return actions
        return np.array([action1, action2, action3]).astype(np.float), stop, self.level