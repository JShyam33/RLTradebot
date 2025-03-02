import gymnasium
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ---------------------------
# Custom Feature Extractor for PPO
# ---------------------------
class CustomExtractor(BaseFeaturesExtractor):
    """
    A simple fully connected network to extract features from the environment observation.
    """

    def __init__(self, observation_space: gymnasium.spaces.Box, features_dim: int = 128):
        # The features_dim is the output size of the extractor.
        super(CustomExtractor, self).__init__(observation_space, features_dim)
        input_dim = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)


# ---------------------------
# Custom Portfolio Environment
# ---------------------------
class PortfolioEnv(gymnasium.Env):
    """
    A custom environment for portfolio management.

    - The bot trades a list of stocks (tickers) with a starting balance of $10,000.
    - The environment downloads historical data from yfinance, computes extra features,
      and the agent receives, for each day, a vector containing engineered features for each ticker,
      plus its current balance and current holdings.
    - The action space is a continuous Box (shape = [n_assets,]) with each value in [-1, 1]:
        * Positive values: buy that fraction of available cash (order size = action value)
        * Negative values: sell that fraction of held shares (order size = |action value|)
        * Near zero: hold.
    - The reward is the change in overall portfolio value. A penalty is applied if the total portfolio
      falls below the initial balance.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, tickers, start_date, end_date, initial_balance=10000):
        super(PortfolioEnv, self).__init__()
        self.tickers = tickers
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.start_date = start_date
        self.end_date = end_date
        self.data = self._download_data(tickers, start_date, end_date)
        # Use the length of one ticker's DataFrame as the number of time steps
        self.n_steps = len(next(iter(self.data.values())))
        self.n_assets = len(tickers)

        # Define observation space:
        # For each ticker we use 10 engineered features. We then append:
        #   - current balance (1 value)
        #   - current holdings for each ticker (n_assets values)
        obs_dim = self.n_assets * 10 + 1 + self.n_assets
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # Define action space as a continuous vector for each ticker (range [-1, 1]).
        # Action > 0: buy, Action < 0: sell, Action near 0: hold.
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_assets,), dtype=np.float32)

        # Initialize holdings (number of shares per ticker)
        self.holdings = {ticker: 0 for ticker in tickers}
        self.portfolio_history = []
        self.current_step = 0
        self.prev_value = self.initial_balance

    def _download_data(self, tickers, start, end):
        data = {}
        for ticker in tickers:
            df = yf.download(ticker, start=start, end=end)
            df = df.reset_index()
            df = self._feature_engineering(df)
            data[ticker] = df
        return data

    def _feature_engineering(self, df):
        # Add simple technical indicators
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['Return'] = df['Close'].pct_change().fillna(0)
        df['Spread'] = df['High'] - df['Low']
        df['NormVolume'] = df['Volume'] / df['Volume'].rolling(window=5).mean()
        df['Volatility'] = df['Return'].rolling(window=5).std()
        df.fillna(method='bfill', inplace=True)
        # Select 10 features: Open, High, Low, Close, Volume, MA5, MA10, Return, Spread, NormVolume.
        return df[['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA10', 'Return', 'Spread', 'NormVolume']]

    def _get_observation(self):
        """
        Construct the observation:
        - For each ticker, the 10 feature values of the current day.
        - Current portfolio state: balance and holdings per ticker.
        """
        obs_list = []
        for ticker in self.tickers:
            df = self.data[ticker]
            df.dropna(inplace=True)
            row = df.iloc[self.current_step].values.astype(np.float32)
            obs_list.extend(row)
        # Append portfolio state: balance and holdings
        balance_arr = np.array([self.balance], dtype=np.float32).flatten()  # Ensure it's 1D
        holdings_arr = np.array([self.holdings[ticker] for ticker in self.tickers], dtype=np.float32).flatten()
        obs_list_arr = np.array(obs_list, dtype=np.float32).flatten()

        # Concatenate properly
        obs = np.concatenate([balance_arr, holdings_arr, obs_list_arr])


        return obs

    def step(self, action):
        # Ensure action is a numpy array
        action = np.clip(action, -1, 1)

        # Get current prices (using the Close price) for each ticker
        prices = {}
        for ticker in self.tickers:
            df = self.data[ticker]
            price = df.iloc[self.current_step]['Close'].values
            prices[ticker] = price

        # Process the action for each ticker
        for i, ticker in enumerate(self.tickers):
            act = action[i]
            price = prices[ticker]
            if act > 0:  # Buy action
                # Use fraction 'act' of available cash to buy shares
                available_cash = self.balance
                order_value = ((available_cash // price) * act) * price
                shares_to_buy = int(order_value // price)
                cost = shares_to_buy * price
                if shares_to_buy > 0 and cost[0] <= self.balance:
                    self.holdings[ticker] += shares_to_buy
                    self.balance -= cost
            elif act < 0:  # Sell action
                # Sell a fraction equal to |act| of held shares
                shares_to_sell = int(self.holdings[ticker] * abs(act))
                revenue = shares_to_sell * price
                if shares_to_sell > 0:
                    self.holdings[ticker] -= shares_to_sell
                    self.balance += revenue
            # act near 0 means hold – do nothing

        # Calculate current portfolio value
        portfolio_value = self.balance
        for ticker in self.tickers:
            portfolio_value += self.holdings[ticker] * prices[ticker]

        # Reward is the change in portfolio value from the previous step.
        reward = portfolio_value - self.prev_value
        # Penalize if overall portfolio falls below the initial balance (risk management)
        if portfolio_value < self.initial_balance:
            penalty = (self.initial_balance - portfolio_value) * 0.1
            reward -= penalty

        self.prev_value = portfolio_value
        self.portfolio_history.append(portfolio_value)

        self.current_step += 1
        done = self.current_step >= self.n_steps

        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), done, done, {}

    def reset(self,seed:int=0):
        self.balance = self.initial_balance
        self.holdings = {ticker: 0 for ticker in self.tickers}
        self.current_step = 0
        self.portfolio_history = []
        self.prev_value = self.initial_balance
        info = {}
        return self._get_observation(),info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Holdings: {self.holdings}")


# ---------------------------
# Create the Environment and Check It
# ---------------------------
tickers = ['AAPL', 'MSFT', 'GOOGL']  # Example tickers; adjust as needed.
env = PortfolioEnv(tickers=tickers, start_date='2023-01-01', end_date='2023-06-01', initial_balance=100)
check_env(env, warn=True)

# ---------------------------
# Set up PPO with the Custom Actor-Critic Network
# ---------------------------
policy_kwargs = dict(
    features_extractor_class=CustomExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
# Train for a number of timesteps (adjust timesteps as necessary)
model.learn(total_timesteps=10000)

# ---------------------------
# Backtest the Trained Bot over the 1-Year Data
# ---------------------------
obs, _ = env.reset()  # Unpack the tuple (obs, info)
done = False

while not done:
    # Predict action using the trained model
    action, _states = model.predict(obs)
    obs, reward, done,truncated, info = env.step(action)

# Plot the portfolio value over time
plt.figure(figsize=(10, 6))
plt.plot(env.portfolio_history)
plt.title("Portfolio Value Over Time")
plt.xlabel("Time Step (Days)")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)
plt.show()
