# Reinforcement Learning for Multi-UAV System using PyBullet

## TODO:
[ ] Create wrapper for pybullet
[ ] Create environment for multi-agent
[ ] 



This repo is mostly a refactor of the works done in [gym-pybullet-drones
](https://github.com/utiasDSL/gym-pybullet-drones).

Manual on commanding the crayzlies:
https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/sensor-to-control/commanders_setpoints/
Controllers on the crazyflie:
https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/functional-areas/sensor-to-control/controllers/

The velocity control law is mostly derived from [here](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=791840409e73b26dcbe705bb3817df04f3fecfc7), Section 2.3.3

Similar Repos:
Quadswarm: https://imrclab.github.io/workshop-uav-sims-icra2023/papers/RS4UAVs_paper_16.pdf
https://github.com/Zhehui-Huang/quad-swarm-rl/tree/master


### 
Install pytorch: 
https://pytorch.org/get-started/locally/
install pytorch (CUDA):


Dynamics of the crazyflie:
https://arxiv.org/pdf/1608.05786.pdf

## Pybullet
install pybullet
pip install pybullet

pybullet quick start guide: 
https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit?tab=t.0#heading=h.2ye70wns7io3
## References
1. [Tutorial On robotis](https://www.youtube.com/watch?v=KaiznOkKkdA)
    [repo](https://github.com/Robotics-Club-IIT-BHU/Robotics-Club-x-NTU-MAERC-collab/tree/main)
2. [Tutorial on URDF](https://articulatedrobotics.xyz/ready-for-ros-7-urdf/)
3. [Bullet3 Repo](https://github.com/bulletphysics/bullet3)

    [The GRASP Micro-UVA testbed](https://ieeexplore.ieee.org/document/5569026)

    [Minimum snap trajectory generation and control for quadrotors](https://ieeexplore.ieee.org/document/5980409)