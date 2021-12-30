import gym
import cv2

import time
import json
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque

ENVIRONMENT = "PongDeterministic-v4"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_MODELS = False  
MODEL_PATH = "./models/pong-cnn-"  
SAVE_MODEL_INTERVAL = 10 
TRAIN_MODEL = False  

LOAD_MODEL_FROM_FILE = True  
LOAD_FILE_EPISODE = 900 

BATCH_SIZE = 64 
MAX_EPISODE = 100000  
MAX_STEP = 100000 

MAX_MEMORY_LEN = 50000 
MIN_MEMORY_LEN = 40000  

GAMMA = 0.97  
ALPHA = 0.00025 
EPSILON_DECAY = 0.99 

RENDER_GAME_WINDOW = True  

#Duel CNN
class DuelCNN(nn.Module):
    def __init__(self, h, w, output_size):
        super(DuelCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4,  out_channels=32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        convw, convh = self.conv2d_size_calc(w, h, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        convw, convh = self.conv2d_size_calc(convw, convh, kernel_size=3, stride=1)

        linear_input_size = convw * convh * 64 #Kích thước đầu vào của lớp state-value và lớp advantage

        #Lớp advantage
        self.Alinear1 = nn.Linear(in_features=linear_input_size, out_features=128)
        self.Alrelu = nn.LeakyReLU()
        self.Alinear2 = nn.Linear(in_features=128, out_features=output_size)

        #Lớp state-value
        self.Vlinear1 = nn.Linear(in_features=linear_input_size, out_features=128)
        self.Vlrelu = nn.LeakyReLU() 
        self.Vlinear2 = nn.Linear(in_features=128, out_features=1)

    def conv2d_size_calc(self, w, h, kernel_size=5, stride=2):
        next_w = (w - (kernel_size - 1) - 1) // stride + 1
        next_h = (h - (kernel_size - 1) - 1) // stride + 1
        return next_w, next_h

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.size(0), -1)

        Ax = self.Alrelu(self.Alinear1(x))
        Ax = self.Alinear2(Ax) 

        Vx = self.Vlrelu(self.Vlinear1(x))
        Vx = self.Vlinear2(Vx) 

        q = Vx + (Ax - Ax.mean())

        return q

#Tác tử
class Agent:
    def __init__(self, environment):
        #Kích thước của khung hình
        self.state_size_h = environment.observation_space.shape[0]
        self.state_size_w = environment.observation_space.shape[1]
        self.state_size_c = environment.observation_space.shape[2]

        #Kích thước của đầu ra là số lượng các hành động
        self.action_size = environment.action_space.n

        #Kích thước của khung hình sau khi xử lý
        self.target_h = 80  
        self.target_w = 64 
        self.crop_dim = [20, self.state_size_h, 0, self.state_size_w]

        self.gamma = GAMMA  #Discount factor
        self.alpha = ALPHA  #Learning Rate

        self.epsilon = 1 if TRAIN_MODEL else 0.0001 #Giá trị epsilon ban đầu
        self.epsilon_decay = EPSILON_DECAY  #Tốc độ giảm giá trị Epsilon
        self.epsilon_minimum = 0.05  #Giá trị epsilon cuối cùng

        #Replay memory
        self.memory = deque(maxlen=MAX_MEMORY_LEN)

        #Khởi tạo model
        self.online_model = DuelCNN(h=self.target_h, w=self.target_w, output_size=self.action_size).to(DEVICE)
        self.target_model = DuelCNN(h=self.target_h, w=self.target_w, output_size=self.action_size).to(DEVICE)
        #Gán giá trị các weight của online model cho target model
        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_model.eval()

        #Chọn optimizer
        self.optimizer = optim.Adam(self.online_model.parameters(), lr=self.alpha)

    def preProcess(self, image):
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Chuyển khung hình về dạng Gray
        frame = frame[self.crop_dim[0]:self.crop_dim[1], self.crop_dim[2]:self.crop_dim[3]]  #Cắt khung hình
        frame = cv2.resize(frame, (self.target_w, self.target_h))  #Chỉnh lại kích thước
        frame = frame.reshape(self.target_w, self.target_h) / 255  #Đưa các giá trị pixel về khoảng [0, 1]

        return frame

    def act(self, state):
        #Lựa chọn hành động
        #Chọn ngẫu nhiên nếu act_protocol <= epsilon
        #Không thì chọn hành động với giá trị Q cao nhất

        act_protocol = 'Explore' if random.uniform(0, 1) <= self.epsilon else 'Exploit'

        if act_protocol == 'Explore':
            action = random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float, device=DEVICE).unsqueeze(0)
                q_values = self.online_model.forward(state) 
                action = torch.argmax(q_values).item()

        return action

    def train(self):
        #Huấn luyện model bằng dữ liệu từ replay memory
        #Trả về loss và giá trị Q cực đại
        if len(agent.memory) < MIN_MEMORY_LEN:
            loss, max_q = [0, 0]
            return loss, max_q
        #Lấy mini batch từ replay memory
        state, action, reward, next_state, done = zip(*random.sample(self.memory, BATCH_SIZE))

        state = np.concatenate(state)
        next_state = np.concatenate(next_state)

        state = torch.tensor(state, dtype=torch.float, device=DEVICE)
        next_state = torch.tensor(next_state, dtype=torch.float, device=DEVICE)
        action = torch.tensor(action, dtype=torch.long, device=DEVICE)
        reward = torch.tensor(reward, dtype=torch.float, device=DEVICE)
        done = torch.tensor(done, dtype=torch.float, device=DEVICE)

        #Lấy các giá trị Q
        state_q_values = self.online_model(state)
        next_states_q_values = self.online_model(next_state)
        next_states_target_q_values = self.target_model(next_state)

        selected_q_value = state_q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        #Tìm Q(s', argmaxQ(s', a'))
        next_states_target_q_value = next_states_target_q_values.gather(1, next_states_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        #Sử dụng phường trình Bellman để tìm giá trị Q target
        expected_q_value = reward + self.gamma * next_states_target_q_value * (1 - done)

        #Tính loss
        loss = (selected_q_value - expected_q_value.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, torch.max(state_q_values).item()

    def storeResults(self, state, action, reward, nextState, done):
        #Lưu dữ liệu vào replay memory
        self.memory.append([state[None, :], action, reward, nextState[None, :], done])

    def adaptiveEpsilon(self):
        #Giám giá trị epsilon để huấn luyện càng nhiều, khám phá càng ít và tận dụng càng nhiều
        if self.epsilon > self.epsilon_minimum:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    #Khởi tạo môi trường
    environment = gym.make(ENVIRONMENT)
    #Khởi tạo tác tử 
    agent = Agent(environment)

    if LOAD_MODEL_FROM_FILE:
        agent.online_model.load_state_dict(torch.load(MODEL_PATH+str(LOAD_FILE_EPISODE)+".pkl", map_location=torch.device('cpu')))

        with open(MODEL_PATH+str(LOAD_FILE_EPISODE)+'.json') as outfile:
            param = json.load(outfile)
            agent.epsilon = param.get('epsilon')

        startEpisode = LOAD_FILE_EPISODE + 1

    else:
        startEpisode = 1

    last_100_ep_reward = deque(maxlen=100)
    total_step = 1 
    for episode in range(startEpisode, MAX_EPISODE):

        startTime = time.time()
        state = environment.reset()

        state = agent.preProcess(state)  #Xử lý khung hình ban đầu

        #Đầu vào ban đầu là 4 khung hình ban đầu
        state = np.stack((state, state, state, state))

        total_max_q_val = 0  
        total_reward = 0 
        total_loss = 0
        for step in range(MAX_STEP):

            if RENDER_GAME_WINDOW:
                environment.render()

            action = agent.act(state)  #Chọn hành động
            next_state, reward, done, info = environment.step(action)  #Quan sát

            next_state = agent.preProcess(next_state)  #Xử lý khung hình vừa quan sát

            #Đầu vào tiếp theo là 4 khung hình gần đây nhất
            next_state = np.stack((next_state, state[0], state[1], state[2]))

            #Lưu dữ liệu
            agent.storeResults(state, action, reward, next_state, done)

            state = next_state

            if TRAIN_MODEL:
                #Huấn luyện model
                loss, max_q_val = agent.train()
            else:
                loss, max_q_val = [0, 0]

            total_loss += loss
            total_max_q_val += max_q_val
            total_reward += reward
            total_step += 1
            if total_step % 20 == 0:
                agent.adaptiveEpsilon()  #Giảm Epsilon

            if done: #Episode kết thúc
                currentTime = time.time()  
                time_passed = currentTime - startTime 
                current_time_format = time.strftime("%H:%M:%S", time.gmtime())
                epsilonDict = {'epsilon': agent.epsilon} #Lưu giá trị epsilon để sau này sử dụng

                if SAVE_MODELS and episode % SAVE_MODEL_INTERVAL == 0:  #Lưu model dưới dạng file pkl và json
                    weightsPath = MODEL_PATH + str(episode) + '.pkl'
                    epsilonPath = MODEL_PATH + str(episode) + '.json'

                    torch.save(agent.online_model.state_dict(), weightsPath)
                    with open(epsilonPath, 'w') as outfile:
                        json.dump(epsilonDict, outfile)

                if TRAIN_MODEL:
                    agent.target_model.load_state_dict(agent.online_model.state_dict())  #Cập nhập target model

                last_100_ep_reward.append(total_reward)
                avg_max_q_val = total_max_q_val / step

                outStr = "Episode:{} |Time:{} |Reward:{:.2f} |Loss:{:.2f} |Last_100_Avg_Rew:{:.3f} |Avg_Max_Q:{:.3f} |Epsilon:{:.2f} |Duration:{:.2f} |Step:{} |CStep:{}".format(
                    episode, current_time_format, total_reward, total_loss, np.mean(last_100_ep_reward), avg_max_q_val, agent.epsilon, time_passed, step, total_step
                )

                print(outStr)
                break
