import numpy as np
from physics_sim import PhysicsSim



class MyTask():
       """
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):

        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.start_pos = self.sim.pose[:4]
        
        
        self.action_repeat = 3
        self.state_size = self.action_repeat * 10
        
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward.""" 
        reward = 0  
        penalty = 0 
        
        current_pos = self.sim.pose[:2]
        
        #penalty
        penalty += abs(self.sim.pose[3:6]).sum()                 
        penalty += abs(current_pos[0]-self.target_pos[0])**2
        penalty += abs(current_pos[1]-self.target_pos[1])**2 
        penalty += abs(current_pos[2]-self.target_pos[2])**2 
    
        #reward
        
          distance = np.sqrt((current_position[0]-self.target_pos[0])**2 + (current_position[1]-self.target_pos[1])**2 + (current_position[2]-self.target_pos[2])**2)
        
        if distance < 15:
            reward += 1150
            
        reward = 1. - .3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        

        return reward - penalty*0.0003
        

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self): 
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state