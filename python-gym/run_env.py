import argparse

from pyencoder.environment.av1_env import Av1Env
from stable_baselines3 import A2C

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)


env = Av1Env("Data/akiyo_qcif.y4m")

episodes = 5
for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        # print(rewards)
