#Enjoy
`python enjoy.py --algo sac --env RoboSkateNumerical-v0 --folder ../trained_models/RoboSkate/RoboSkateNumerical-v0/ --load-best --env-kwargs headlessMode:False random_start_level:False startLevel:0 --exp-id 2 --n-timesteps 2000`


### sac/RoboSkateNumerical-v0_1
Was only trained to level 2. Can solve level 3 without having seen it.


### sac/RoboSkateNumerical-v0_2
sac/RoboSkateNumerical-v0_1 was used as a pre-trained model and it was always started in level 1 and trained through all levels.


### sac/RoboSkateNumerical-v0_3
Was trained from scratch 
with `max_Joint_force = 350.0`
without 
`        elif abs(board_roll-0.5) > 0.35:
            # Stop if board is tipped
            self.reward -= 10
            print("board tipped")
            done = True
        elif abs(self.state.boardCraneJointAngles[3] * max_Joint_vel) > 200:
            # Stop if turning the first joint to fast "Helicopter"
            self.reward -= 10
            print("Helicopter")
            done = True`

and only on level 1


### sac/RoboSkateNumerical-v0_4
Was trained from scratch \
use agent_without_limits = True \
with `max_Joint_force = 1000.0` \
without \
`        elif abs(board_roll-0.5) > 0.35:
            # Stop if board is tipped
            self.reward -= 10
            print("board tipped")
            done = True
        elif abs(self.state.boardCraneJointAngles[3] * max_Joint_vel) > 200:
            # Stop if turning the first joint to fast "Helicopter"
            self.reward -= 10
            print("Helicopter")
            done = True`

and only on level 1




### sac/RoboSkateSegmentation-v0_1
Trained on level 1 with image data latent space of VAE size 2


### sac/RoboSkateSegmentation-v0_2
Trained on level 2 with image data latent space of VAE size 2

### sac/RoboSkateSegmentation-v0_3
Trained on level 3 with image data latent space of VAE size 2

