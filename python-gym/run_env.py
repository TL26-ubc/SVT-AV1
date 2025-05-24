import argparse

import pyencoder
from pyencoder.environment.av1_env import Av1Env
from stable_baselines3 import A2C

env = Av1Env(
    "Data/akiyo_qcif.y4m",
    av1_runner=lambda x: pyencoder.run(
        input="../Data/akiyo_qcif.y4m", rc=True, enable_stat_report=True
    ),
)

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

episodes = 5
for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        # print(rewards)
