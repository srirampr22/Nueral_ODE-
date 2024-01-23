import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, OBSTACLE_HALFDIMS, OBSTACLE_CENTRE, BOX_SIZE
from torchdiffeq import odeint as odeint

TARGET_POSE_FREE_TENSOR = torch.as_tensor(TARGET_POSE_FREE, dtype=torch.float32)
TARGET_POSE_OBSTACLES_TENSOR = torch.as_tensor(TARGET_POSE_OBSTACLES, dtype=torch.float32)
OBSTACLE_CENTRE_TENSOR = torch.as_tensor(OBSTACLE_CENTRE, dtype=torch.float32)[:2]
OBSTACLE_HALFDIMS_TENSOR = torch.as_tensor(OBSTACLE_HALFDIMS, dtype=torch.float32)[:2]


def collect_data_random(env, num_trajectories=1000, trajectory_length=10):
    """
    Collect data from the provided environment using uniformly random exploration.
    :param env: Gym Environment instance.
    :param num_trajectories: <int> number of data to be collected.
    :param trajectory_length: <int> number of state transitions to be collected
    :return: collected data: List of dictionaries containing the state-action trajectories.
    Each trajectory dictionary should have the following structure:
        {'states': states,
        'actions': actions}
    where
        * states is a numpy array of shape (trajectory_length+1, state_size) containing the states [x_0, ...., x_T]
        * actions is a numpy array of shape (trajectory_length, actions_size) containing the actions [u_0, ...., u_{T-1}]
    Each trajectory is:
        x_0 -> u_0 -> x_1 -> u_1 -> .... -> x_{T-1} -> u_{T_1} -> x_{T}
        where x_0 is the state after resetting the environment with env.reset()
    All data elements must be encoded as np.float32.
    """
    collected_data = None
    # --- Your code here
    collected_data=[]
    
    for j in range (num_trajectories):
      state = env.reset()
      states = np.zeros((trajectory_length+1,env.observation_space.shape[0]),dtype = np.float32)
      actions = np.zeros((trajectory_length,env.action_space.shape[0]),dtype = np.float32)
      states[0] = state
      for i in range (trajectory_length):
        action = env.action_space.sample()
        next_state,_,_,_ = env.step(action)
        actions[i] = action
        state = next_state
        states[i+1] = state
      collected_data.append({'states':states,'actions':actions})
      # print(collected_data)

    # ---
    return collected_data


def process_data_single_step(collected_data, batch_size=500):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as {'state': x_t,
     'action': u_t,
     'next_state': x_{t+1},
    }
    where:
     x_t: torch.float32 tensor of shape (batch_size, state_size)
     u_t: torch.float32 tensor of shape (batch_size, action_size)
     x_{t+1}: torch.float32 tensor of shape (batch_size, state_size)

    The data should be split in a 80-20 training-validation split.
    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :return:

    Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement SingleStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    train_loader = None
    val_loader = None
    # --- Your code here
    train_loader=[]
    val_loader =[]
    
    dataset = SingleStepDynamicsDataset(collected_data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # ---
    return train_loader, val_loader


def process_data_multiple_step(collected_data, batch_size=500, num_steps=4):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as
    {'state': x_t,
     'action': u_t, ..., u_{t+num_steps-1},
     'next_state': x_{t+1}, ... , x_{t+num_steps}
    }
    where:
     state: torch.float32 tensor of shape (batch_size, state_size)
     next_state: torch.float32 tensor of shape (batch_size, num_steps, action_size)
     action: torch.float32 tensor of shape (batch_size, num_steps, state_size)

    Each DataLoader must load dictionary dat
    The data should be split in a 80-20 training-validation split.
    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :param num_steps: <int> number of steps to load the multistep data.
    :return:

    Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement MultiStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    train_loader = None
    val_loader = None
    # --- Your code here
    train_loader=[]
    val_loader =[]
    
    dataset = MultiStepDynamicsDataset(collected_data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


    # ---
    return train_loader, val_loader


class SingleStepDynamicsDataset(Dataset):
    """
    Each data sample is a dictionary containing (x_t, u_t, x_{t+1}) in the form:
    {'state': x_t,
     'action': u_t,
     'next_state': x_{t+1},
    }
    where:
     x_t: torch.float32 tensor of shape (state_size,)
     u_t: torch.float32 tensor of shape (action_size,)
     x_{t+1}: torch.float32 tensor of shape (state_size,)
    """

    def __init__(self, collected_data):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0]

    def __len__(self):
        return len(self.data) * self.trajectory_length

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (state, action, next_state).
        The class description has more details about the format of this data sample.
        """
        sample = {
            'state': None,
            'action': None,
            'next_state': None,
        }
        # --- Your code here
        traj_idx = item // self.trajectory_length
        step_idx = item % self.trajectory_length

        sample['state'] = self.data[traj_idx]['states'][step_idx]
        sample['action'] = self.data[traj_idx]['actions'][step_idx]
        sample['next_state'] = self.data[traj_idx]['states'][step_idx + 1]


        # ---
        return sample


class MultiStepDynamicsDataset(Dataset):
    """
    Dataset containing multi-step dynamics data.

    Each data sample is a dictionary containing (state, action, next_state) in the form:
    {'state': x_t, -- initial state of the multipstep torch.float32 tensor of shape (state_size,)
     'action': [u_t,..., u_{t+num_steps-1}] -- actions applied in the muli-step.
                torch.float32 tensor of shape (num_steps, action_size)
     'next_state': [x_{t+1},..., x_{t+num_steps} ] -- next multiple steps for the num_steps next steps.
                torch.float32 tensor of shape (num_steps, state_size)
    }
    """

    def __init__(self, collected_data, num_steps=4):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0] - num_steps + 1
        self.num_steps = num_steps

    def __len__(self):
        return len(self.data) * (self.trajectory_length)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (state, action, next_state).
        The class description has more details about the format of this data sample.
        """
        sample = {
            'state': None,
            'action': None,
            'next_state': None
        }
        # --- Your code here
        traj_idx = item // self.trajectory_length
        step_idx = item % self.trajectory_length
        sample['state'] = self.data[traj_idx]['states'][step_idx]

        actions = np.zeros((self.num_steps, self.data[traj_idx]['actions'].shape[1]), dtype=np.float32)
        for i in range(self.num_steps):
            actions[i] = self.data[traj_idx]['actions'][step_idx + i]
        sample['action'] = actions

        next_states = np.zeros((self.num_steps, self.data[traj_idx]['states'].shape[1]), dtype=np.float32)
        for i in range(self.num_steps):
            next_states[i] = self.data[traj_idx]['states'][step_idx + i + 1]
        sample['next_state'] = next_states

        # ---
        return sample


class SE2PoseLoss(nn.Module):
    """
    Compute the SE2 pose loss based on the object dimensions (block_width, block_length).
    Need to take into consideration the different dimensions of pose and orientation to aggregate them.

    Given a SE(2) pose [x, y, theta], the pose loss can be computed as:
        se2_pose_loss = MSE(x_hat, x) + MSE(y_hat, y) + rg * MSE(theta_hat, theta)
    where rg is the radious of gyration of the object.
    For a planar rectangular object of width w and length l, the radius of gyration is defined as:
        rg = ((l^2 + w^2)/12)^{1/2}

    """

    def __init__(self, block_width, block_length):
        super().__init__()
        self.w = block_width
        self.l = block_length

    def forward(self, pose_pred, pose_target):
        se2_pose_loss = None
        # --- Your code here
        x1,y1,t1 = pose_pred[:,0],pose_pred[:,1],pose_pred[:,2]
        x2,y2,t2 = pose_target[:,0],pose_target[:,1],pose_target[:,2]
        gyr = (((self.w**2+self.l**2)/12) ** 0.5)
        se2_pose_loss = nn.MSELoss()(x1,x2) + nn.MSELoss()(y1,y2) + gyr*nn.MSELoss()(t1,t2)

        # ---
        return se2_pose_loss


class SingleStepLoss(nn.Module):

    def __init__(self, loss_fn):
        super().__init__()
        self.loss = loss_fn

    def forward(self, model, state, action, target_state):
        """
        Compute the single step loss resultant of querying model with (state, action) and comparing the predictions with target_state.
        """
        single_step_loss = None
        # --- Your code here

        next_state = model(state,action)
        single_step_loss = self.loss(next_state,target_state)
        # ---
        return single_step_loss


class MultiStepLoss(nn.Module):

    def __init__(self, loss_fn, discount=0.99):
        super().__init__()
        self.loss = loss_fn
        self.discount = discount

    def forward(self, model, state, actions, target_states):
        """
        Compute the multi-step loss resultant of multi-querying the model from (state, action) and comparing the predictions with targets.
        """
        multi_step_loss = None
        # --- Your code here
        multi_step_loss = 0
        state_=state
        t = torch.linspace(0,1,steps=4)
        for i in range(actions.shape[1]):
          y0 = torch.cat([state_,actions[:,i,:]], dim=-1)
          next_state = odeint(model,y0,t)
          target = target_states
          next_state = next_state[:,:,:3].permute(1,0,2)
          multi_step_loss += (self.discount**i) * self.loss(next_state[:,-1,:],target[:,i])
          state_ = next_state[:,-1,:]
          
        # ---
        return multi_step_loss


class AbsoluteDynamicsModel(nn.Module):
    """
    Model the absolute dynamics x_{t+1} = f(x_{t},a_{t})
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # --- Your code here
        hidden =100
        self.l1 = nn.Linear(state_dim+action_dim,hidden)
        self.l2 = nn.Linear(hidden,hidden)
        self.l3 = nn.Linear(hidden,state_dim)
        self.activation = nn.ReLU()

        # ---

    def forward(self, state, action):
        """
        Compute next_state resultant of applying the provided action to provided state
        :param state: torch tensor of shape (..., state_dim)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., state_dim)
        """
        next_state = None
        # --- Your code here
        x = torch.cat([state,action],dim=1)
        x = self.activation(self.l1(x))
        x = self.activation(self.l2(x))
        next_state = self.l3(x)

        # ---
        return next_state


class ResidualDynamicsModel(nn.Module):
    """
    Model the residual dynamics s_{t+1} = s_{t} + f(s_{t}, u_{t})

    Observation: The network only needs to predict the state difference as a function of the state and action.
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # --- Your code here
        hidden =100
        self.l1 = nn.Linear(state_dim+action_dim,hidden)
        # nn.init.normal_(self.l1.weight, mean=0, std=0.1)
        # nn.init.constant_(self.l1.bias, val=0)
        self.l2 = nn.Linear(hidden,hidden)
        # nn.init.normal_(self.l2.weight, mean=0, std=0.1)
        # nn.init.constant_(self.l2.bias, val=0)
        self.l3 = nn.Linear(hidden,state_dim+action_dim)
        # nn.init.normal_(self.l3.weight, mean=0, std=0.1)
        # nn.init.constant_(self.l3.bias, val=0)
        self.activation = nn.ReLU()
        # self.net = nn.Sequential(
        #     nn.Linear(state_dim+action_dim,hidden),
        #     nn.Tanh(),
        #      nn.Linear(hidden,hidden),
        #      nn.Tanh,
        #      nn.Linear(hidden,state_dim+action_dim)
        # )

        # for m in self.net.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, mean=0, std=0.1)
        #         nn.init.constant_(m.bias, val=0)
        # ---

    def forward(self,t, state):
        """
        Compute next_state resultant of applying the provided action to provided state
        :param state: torch tensor of shape (..., state_dim)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., state_dim)
        """
        next_state = None
        # --- Your code here
        x = state
        # print('p',state.shape)
        x = self.activation(self.l1(x))
        x = self.activation(self.l2(x))
        next_state = self.l3(x) + state
        # ---
        return next_state


def free_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup without obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_FREE_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    # --- Your code here
    Q = torch.zeros((3,3))
    cost= []
    Q[0,0] = 1
    Q[1,1] = 1
    Q[2,2] = 0.1
    ssum_ = 0
    for t in range (state.shape[0]):
      sub = state[t] - target_pose
      cost.append(sub @ Q @ sub)
    cost = torch.tensor(cost)
    # print(state.shape,cost.shape)
    # ---
    return cost


def collision_detection(state):
    """
    Checks if the state is in collision with the obstacle.
    The obstacle geometry is known and provided in obstacle_centre and obstacle_halfdims.
    :param state: torch tensor of shape (B, state_size)
    :return: in_collision: torch tensor of shape (B,) containing 1 if the state is in collision and 0 if not.
    """
    obstacle_centre = OBSTACLE_CENTRE_TENSOR  # torch tensor of shape (2,) consisting of obstacle centre (x, y)
    obstacle_dims = 2 * OBSTACLE_HALFDIMS_TENSOR  # torch tensor of shape (2,) consisting of (w_obs, l_obs)
    box_size = BOX_SIZE  # scalar for parameter w
    in_collision = None
    # --- Your code here
    # print(obstacle_centre[0],obstacle_centre[1])
    xobs,yobs = obstacle_centre[0],obstacle_centre[1]
    in_collision = []
    w_obs,l_obs = obstacle_dims[0],obstacle_dims[1]
    for i in range (state.shape[0]):

      #x,y,theta
      x_block,y_block,theta_block = state[i,0],state[i,1],state[i,2]
      

      #corners of the block
      top_right_x,top_right_y = (x_block + box_size/2),(y_block + box_size/2)
      top_left_x,top_left_y = (x_block - box_size/2),(y_block + box_size/2)
      bottom_right_x,bottom_right_y = (x_block + box_size/2),(y_block - box_size/2)
      bottom_left_x,bottom_left_y =  (x_block - box_size/2),(y_block - box_size/2)

      coord = torch.tensor([[top_left_x,top_left_y],[top_right_x,top_right_y],[bottom_left_x,bottom_left_y],[bottom_right_x,bottom_right_y]])
      rotation_matrix = torch.tensor([[torch.cos(theta_block),-torch.sin(theta_block)],[torch.sin(theta_block),torch.cos(theta_block)]])
      coord = coord @ rotation_matrix

      top_left_x,top_left_y = coord[0][0],coord[0][1]
      top_right_x,top_right_y = coord[1][0],coord[1][1]
      bottom_left_x,bottom_left_y = coord[2][0],coord[2][1]
      bottom_right_x,bottom_right_y = coord[3][0],coord[3][1]

      # print(top_left_x,top_left_y,top_right_x,top_right_y,bottom_left_x,bottom_left_y,bottom_right_x,bottom_right_y)


      #diagonal calculation for obstacle
      x_right,y_top = xobs+w_obs/2,yobs+l_obs/2
      x_left,y_bottom = xobs-w_obs/2,yobs-l_obs/2
      diagonal_obs = ((x_left-x_right)**2 + (y_top - y_bottom)**2) ** 0.5

      #diagonal calculations for box
      diagonal_box = ((top_right_x - bottom_left_x)**2 + (top_right_y - bottom_left_y)**2) ** 0.5
      

      #condition for collision
      #diagonal
      dist_obs_box = ((xobs - x_block)**2 + (yobs - y_block)**2) ** 0.5
      if(dist_obs_box > ((diagonal_box/2) +(diagonal_obs/2))):
        in_collision.append(0.0)

      #width
      else:
        if (top_right_x <= x_right) and (top_right_x >= x_left) and (top_right_y <= y_top) and (top_right_y >= y_bottom):
          in_collision.append(1.0)

      #length
        elif (top_left_x <= x_right) and (top_left_x >= x_left) and (top_left_y <= y_top) and (top_left_y >= y_bottom):
          in_collision.append(1.0)


        elif (bottom_right_x <= x_right) and (bottom_right_x >= x_left) and (bottom_right_y <= y_top) and (bottom_right_y >= y_bottom):
          in_collision.append(1.0)
        
        elif (bottom_left_x <= x_right) and (bottom_left_x >= x_left) and (bottom_left_y <= y_top) and (bottom_left_y >= y_bottom):
          in_collision.append(1.0)
        
      #no collision
        else:
          in_collision.append(0.0)

    in_collision = torch.tensor(in_collision)
      
    # ---
    return in_collision


def obstacle_avoidance_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup with obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_OBSTACLES_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    # --- Your code here
    # cost = obstacle_avoidance_pushing_cost_function (state,action)
    Q = torch.zeros((3,3))
    cost= []
    Q[0,0] = 1
    Q[1,1] = 1
    Q[2,2] = 0.1
    ssum_ = 0
    incollision = collision_detection(state)
    # print(incollision)
    # print(len(incollision))
    for t in range (state.shape[0]):
      sub = state[t] - target_pose
      # print(state[t])
      # if (incollision[t] == 1):
      #   print(t)
      cost.append(sub.t() @ Q @ sub+ (100 * incollision[t]))
    cost = torch.tensor(cost)
    
    
    # cost = torch.tensor(cost)
    # print(state.shape,cost.shape)
    # ---
    return cost


class PushingController(object):
    """
    MPPI-based controller
    Since you implemented MPPI on HW2, here we will give you the MPPI for you.
    You will just need to implement the dynamics and tune the hyperparameters and cost functions.
    """

    def __init__(self, env, model, cost_function, num_samples=150, horizon=15):
        self.env = env
        self.model = model
        self.target_state = None
        # MPPI Hyperparameters:
        # --- You may need to tune them
        state_dim = env.observation_space.shape[0]
        u_min = torch.from_numpy(env.action_space.low)
        u_max = torch.from_numpy(env.action_space.high)
        noise_sigma = 0.5 * torch.eye(env.action_space.shape[0])
        lambda_value = 0.01
        # ---
        from mppi import MPPI
        self.mppi = MPPI(self._compute_dynamics,
                         cost_function,
                         nx=state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max)

    def _compute_dynamics(self, state, action):
        """
        Compute next_state using the dynamics model self.model and the provided state and action tensors
        :param state: torch tensor of shape (B, state_size)
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, state_size) containing the predicted states from the learned model.
        """
        next_state = None
        # --- Your code here
        # print(state.shape)
        # print(action.shape)
        # next_state = self.model(state,action)
        state_=state
        # t = [0,0.25,0.5,0.75]
        # t = torch.tensor(t)
        t = torch.linspace(0,1,steps=4)
        # print(state.shape)
        # print(action.shape)
        model = self.model
        # for i in range(action.shape[1]):
          # next_state = model(state_,actions[:,i,:])
          # state_=state[:,i]
          # print('l',state_.shape)
          # print(action[:,i].shape)
        y0 = torch.cat([state_,action], dim=-1)
        # print('y0',y0.shape)
        next_state = odeint(model,y0,t)
        
        
        # target = target_states
        next_state = next_state[:,:,:3].permute(1,0,2)
        next_state = next_state[:,-1,:]

        
          
        
        # ---
        return next_state

    def control(self, state):
        """
        Query MPPI and return the optimal action given the current state <state>
        :param state: numpy array of shape (state_size,) representing current state
        :return: action: numpy array of shape (action_size,) representing optimal action to be sent to the robot.
        TO DO:
         - Prepare the state so it can be send to the mppi controller. Note that MPPI works with torch tensors.
         - Unpack the mppi returned action to the desired format.
        """
        action = None
        state_tensor = None
        # --- Your code here
        state_tensor = torch.from_numpy(state)
        # print(state_tensor)
        # ---
        action_tensor = self.mppi.command(state_tensor)
        # action_tensor.requires_grad = False
        # --- Your code here
        action = action_tensor.detach().numpy()
        # action = action.numpy()
        # ---
        return action

# =========== AUXILIARY FUNCTIONS AND CLASSES HERE ===========
# --- Your code here



# ---
# ============================================================
