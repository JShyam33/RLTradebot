import os

import gymnasium
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.env_checker import check_env
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
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


class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    Custom ActorCriticPolicy with separate actor and critic networks.

    Parameters:
        actor_net_arch (list): Hidden layer sizes for the actor network.
        critic_net_arch (list): Hidden layer sizes for the critic network.
    """

    def __init__(self, *args, actor_net_arch=[64, 64], critic_net_arch=[64, 64], **kwargs):
        # Call the parent constructor. It sets up the feature extractor and the action distribution.
        super(CustomActorCriticPolicy, self).__init__(*args, **kwargs)



        # Get the dimensionality of the features extracted from the observation.
        feature_dim = self.features_extractor.features_dim

        # Build the actor network from the extracted features.
        actor_layers = []
        input_dim = feature_dim
        for hidden_size in actor_net_arch:
            actor_layers.append(nn.Linear(input_dim, hidden_size))
            actor_layers.append(nn.ReLU())
            input_dim = hidden_size
        self.actor_net = nn.Sequential(*actor_layers)

        # Build the critic network from the extracted features.
        critic_layers = []
        input_dim = feature_dim
        for hidden_size in critic_net_arch:
            critic_layers.append(nn.Linear(input_dim, hidden_size))
            critic_layers.append(nn.ReLU())
            input_dim = hidden_size
        self.critic_net = nn.Sequential(*critic_layers)

        # Final layers: one that maps the actor network's output to the action space,
        # and one that maps the critic network's output to a single state value.
        self.action_net = nn.Linear(actor_net_arch[-1], self.action_space.shape[0])
        self.value_net = nn.Linear(critic_net_arch[-1], 1)

        # For continuous action spaces, register a learnable log standard deviation parameter.
        # (This is used by the DiagGaussianDistribution.)
        self.log_std = nn.Parameter(th.ones(self.action_space.shape[0]) * -0.5)

        # Initialize weights if desired (optional)
        self._initialize_weights()

        self.action_dist = DiagGaussianDistribution(self.action_space.shape[0])

    def _initialize_weights(self):
        # A simple weight initialization using orthogonal initialization.
        for m in [self.actor_net, self.critic_net, self.action_net, self.value_net]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight)
                    if layer.bias is not None:
                        layer.bias.data.fill_(0.0)



    def _predict(self, observation, deterministic=False):
        # The _predict method is used during inference.
        actions, _, _ = self.forward(observation, deterministic)
        return actions


# ---------------------------
# Custom Portfolio Environment
# ---------------------------
class PortfolioEnv(gymnasium.Env):
    """
    A custom environment for portfolio management.

    The environment simulates trading for a list of stocks (tickers) with a starting balance.
    Modifications for more realistic trading include:
      - Trade fees on each buy and sell.
      - Limits on shares traded per transaction.
      - Maximum position sizes per ticker.
      - A drawdown penalty if portfolio value falls too far below its peak.
      - Periodic rebalancing to a target allocation instead of immediate reinvestment.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, tickers, start_date, end_date, initial_balance=10000,
                 trade_fee_percentage=0.001, max_shares_per_trade=100, max_shares_per_ticker=1000,
                 drawdown_limit=0.2, drawdown_penalty_factor=0.5,reward_factor=1, rebalance_period=5):
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
        self.peak_value = self.initial_balance  # for tracking drawdowns

        # New parameters for realistic simulation
        self.trade_fee_percentage = trade_fee_percentage
        self.max_shares_per_trade = max_shares_per_trade
        self.max_shares_per_ticker = max_shares_per_ticker
        self.drawdown_limit = drawdown_limit
        self.reward_factor = reward_factor
        self.drawdown_penalty_factor = drawdown_penalty_factor
        self.rebalance_period = rebalance_period  # in time steps (days)

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
        df.bfill( inplace=True)
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

    def _rebalance_portfolio(self, prices):
        """
        Periodically rebalance the portfolio to a target allocation.
        Here, we rebalance to 80% in equities (evenly distributed across tickers)
        and 20% in cash. In this updated version, we ensure that buying
        transactions do not reduce the cash balance below the target cash level.
        """
        total_value = self.balance + sum(self.holdings[ticker] * prices[ticker] for ticker in self.tickers)
        target_cash_fraction = 0.2
        target_equity_fraction = 0.8
        target_cash = target_cash_fraction * total_value
        target_equity_value = target_equity_fraction * total_value
        target_value_per_asset = target_equity_value / self.n_assets

        for ticker in self.tickers:
            current_equity = self.holdings[ticker] * prices[ticker]
            # If current equity is above target, sell the excess
            if current_equity > target_value_per_asset:
                excess_value = current_equity - target_value_per_asset
                shares_to_sell = int(excess_value // prices[ticker])
                shares_to_sell = min(shares_to_sell, self.holdings[ticker], self.max_shares_per_trade)
                if shares_to_sell > 0:
                    revenue = shares_to_sell * prices[ticker]
                    fee = revenue * self.trade_fee_percentage
                    net_revenue = revenue - fee
                    self.holdings[ticker] -= shares_to_sell
                    self.balance += net_revenue
            # If current equity is below target, buy up to the target,
            # but only spend cash above the target cash reserve.
            elif current_equity < target_value_per_asset:
                deficit_value = target_value_per_asset - current_equity
                # Only use cash above the target cash reserve for buying
                available_cash_for_buying = max(self.balance - target_cash, 0)
                max_affordable_shares = int(available_cash_for_buying // prices[ticker])
                shares_to_buy = int(deficit_value // prices[ticker])
                shares_to_buy = min(shares_to_buy, max_affordable_shares, self.max_shares_per_trade,
                                    self.max_shares_per_ticker - self.holdings[ticker])
                if shares_to_buy > 0:
                    cost = shares_to_buy * prices[ticker]
                    fee = cost * self.trade_fee_percentage
                    total_cost = cost + fee
                    if total_cost <= available_cash_for_buying:
                        self.holdings[ticker] += shares_to_buy
                        self.balance -= total_cost


    def step(self, action):
        # Ensure action is a numpy array and clip it to [-1, 1]
        action = np.clip(action, -1, 1)

        # Get current prices (using the Close price) for each ticker
        prices = {}
        for ticker in self.tickers:
            df = self.data[ticker]
            # Get current price; note: no need for .values if it's a scalar.
            price = df.iloc[self.current_step]['Close'].values
            prices[ticker] = price

        # Process the action for each ticker
        for i, ticker in enumerate(self.tickers):
            act = action[i]
            price = prices[ticker]
            if act > 0:  # Buy action
                # Calculate desired number of shares based on available cash and action fraction
                max_affordable_shares = int(self.balance // price)
                desired_shares = int(max_affordable_shares * act)
                # Limit by maximum shares per trade and max allowed position per ticker
                desired_shares = min(desired_shares, self.max_shares_per_trade)
                allowed_shares = self.max_shares_per_ticker - self.holdings[ticker]
                shares_to_buy = min(desired_shares, allowed_shares)
                cost = shares_to_buy * price
                fee = cost * self.trade_fee_percentage
                total_cost = cost + fee
                if shares_to_buy > 0 and total_cost <= self.balance:
                    self.holdings[ticker] += shares_to_buy
                    self.balance -= total_cost

            elif act < 0:  # Sell action
                # Sell a fraction equal to |act| of held shares
                desired_shares = int(self.holdings[ticker] * abs(act))
                desired_shares = min(desired_shares, self.max_shares_per_trade)
                shares_to_sell = min(desired_shares, self.holdings[ticker])
                revenue = shares_to_sell * price
                fee = revenue * self.trade_fee_percentage
                net_revenue = revenue - fee
                if shares_to_sell > 0:
                    self.holdings[ticker] -= shares_to_sell
                    self.balance += net_revenue
            # act near 0 means hold – do nothing

        # Calculate current portfolio value
        portfolio_value = self.balance + sum(self.holdings[ticker] * prices[ticker] for ticker in self.tickers)

        # Update peak portfolio value for drawdown calculation
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

        # Compute drawdown and apply penalty if it exceeds the allowed limit
        drawdown = (self.peak_value - portfolio_value) / self.peak_value
        reward = (portfolio_value - self.prev_value) * self.reward_factor
        if drawdown > self.drawdown_limit:
            penalty = (drawdown - self.drawdown_limit) * portfolio_value * self.drawdown_penalty_factor
            reward -= penalty

        # Periodic rebalancing step (if enabled)
        if self.rebalance_period > 0 and self.current_step > 0 and self.current_step % self.rebalance_period == 0:
            self._rebalance_portfolio(prices)
            # Recalculate portfolio value after rebalancing
            portfolio_value = self.balance + sum(self.holdings[ticker] * prices[ticker] for ticker in self.tickers)

        self.prev_value = portfolio_value
        self.portfolio_history.append(portfolio_value)
        self.current_step += 1
        done = self.current_step >= self.n_steps

        obs = self._get_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), done, done, {}

    def reset(self, seed:int=0):
        self.balance = self.initial_balance
        self.holdings = {ticker: 0 for ticker in self.tickers}
        self.current_step = 0
        self.portfolio_history = []
        self.prev_value = self.initial_balance
        self.peak_value = self.initial_balance
        info = {}
        return self._get_observation(), info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Holdings: {self.holdings}")

# ---------------------------
# Create the Environment and Check It
# ---------------------------
tickers = ['TSLA', 'GME', 'NVDA']  # Example tickers; adjust as needed.
env = PortfolioEnv(
    tickers=tickers,
    start_date='2023-01-01',
    end_date='2024-01-01',
    initial_balance=10000,
    trade_fee_percentage=0.001,
    max_shares_per_trade=50,
    max_shares_per_ticker=200,
    drawdown_limit=0.2,
    drawdown_penalty_factor=0.3,
    reward_factor=5,
    rebalance_period=-1
)
check_env(env, warn=True)

# ---------------------------
# Set up PPO with the Custom Actor-Critic Network
# ---------------------------
policy_kwargs = dict(
actor_net_arch= [128,1024,512,256, 128],
    critic_net_arch = [256,512,256, 128],
    net_arch=[dict(pi=[128,1024,512,256, 128], vf=[256,512,256, 128])],
    features_extractor_class=CustomExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

MODEL_PATH = "ppo_trading_model.zip"

def get_model(env):
    """
    Load an existing model if available, otherwise create a new one.
    """
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        model = PPO.load(MODEL_PATH, env=env, device="cuda")
    else:
        print("No existing model found. Creating new model...")
        model = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1,n_epochs=30, device="cuda",batch_size=128)
        # Train for a number of timesteps (adjust timesteps as necessary)
        model.learn(total_timesteps=10000)

    return model


def save_model(model):
    """
    Save the trained model.
    """
    print("Saving model...")
    model.save(MODEL_PATH)



model = get_model(env)



# ---------------------------
# Define alternative strategies for backtesting:
# 1. Rule-Based Strategy (using moving average crossover)
# 2. Random Strategy (actions sampled uniformly)
# ---------------------------
def rule_based_policy(obs, n_assets):
    """
    Use MA5 and MA10 from the engineered features to decide:
      - If MA5 > MA10: Buy (action=1)
      - If MA5 < MA10: Sell (action=-1)
      - Otherwise: Hold (action=0)
    Note: The observation consists of [balance, holdings, features_for_each_ticker]
    where each ticker has 10 features and MA5, MA10 are at positions 5 and 6.
    """
    # Skip portfolio state (first 1+n_assets elements)
    features = obs[(1 + n_assets):]
    action = []
    for i in range(n_assets):
        ticker_features = features[i * 10:(i + 1) * 10]
        ma5 = ticker_features[5]
        ma10 = ticker_features[6]
        if ma5 > ma10:
            action.append(1.0)
        elif ma5 < ma10:
            action.append(-1.0)
        else:
            action.append(0.0)
    return np.array(action, dtype=np.float32)

def run_simulation(env, strategy, n_assets, model=None):
    """
    Run the simulation for the given environment using:
      - 'RL' strategy: use the PPO model to predict actions.
      - 'Rule-Based' strategy: use the rule_based_policy.
      - 'Random' strategy: sample actions randomly.
    """
    obs, _ = env.reset()
    done = False
    while not done:
        if strategy == "RL":
            action, _ = model.predict(obs)
        elif strategy == "Rule-Based":
            action = rule_based_policy(obs, n_assets)
        elif strategy == "Random":
            action = np.zeros(n_assets)  # Default action is 0 (no action)
            for i in range(n_assets):
                if np.random.randint(0, 100) > 20:  # 50% probability to take action
                    action[i] = np.random.uniform(-1, 1)  # Buy or hold randomly
        else:
            action = np.zeros(n_assets)
        obs, reward, done, truncated, info = env.step(action)
    return env.portfolio_history

# ---------------------------
# Create test environments for all strategies over the same period
# ---------------------------
test_tickers = ['TSLA', 'GME', 'NVDA']
start_date = '2024-01-01'
end_date = '2024-06-01'
env_test = PortfolioEnv(
    tickers=test_tickers,
    start_date=start_date,
    end_date=end_date,
    initial_balance=10000,
    trade_fee_percentage=0.001,
    max_shares_per_trade=50,
    max_shares_per_ticker=200,
    drawdown_limit=0.2,
    drawdown_penalty_factor=0.5,
    reward_factor=5,
    rebalance_period=-1
)


n_assets = len(test_tickers)
portfolio_history_rl = run_simulation(env_test, "RL", n_assets, model=model)
env_test.reset()
portfolio_history_rule = run_simulation(env_test, "Rule-Based", n_assets)
env_test.reset()
portfolio_history_random = run_simulation(env_test, "Random", n_assets)

# ---------------------------
# Plot the portfolio values over time for all three strategies
# ---------------------------


# Draw dashed horizontal lines at the final values
final_rl = portfolio_history_rl[-1].item()
final_rule = portfolio_history_rule[-1].item()
final_random = portfolio_history_random[-1].item()




plt.figure(figsize=(12, 8))

plt.plot(portfolio_history_rl, label="RL Strategy")
plt.plot(portfolio_history_rule, label="Rule-Based Strategy")
plt.plot(portfolio_history_random, label="Random Strategy")

x_pad , y_pad = 5,5

# Add final point and label for RL Strategy
x_rl = len(portfolio_history_rl) - 1
y_rl = portfolio_history_rl[-1].item()
plt.plot(x_rl, y_rl, marker='o', color='#1D66B6')
plt.text(x_rl + x_pad, y_rl + y_pad, f"{y_rl:.2f}", color='#1D66B6', va='bottom', ha='center')

# Add final point and label for Rule-Based Strategy
x_rule = len(portfolio_history_rule) - 1
y_rule = portfolio_history_rule[-1].item()
plt.plot(x_rule, y_rule, marker='o', color='orange')
plt.text(x_rule + x_pad, y_rule + y_pad, f"{y_rule:.2f}", color='orange', va='bottom', ha='center')

# Add final point and label for Random Strategy
x_random = len(portfolio_history_random) - 1
y_random = portfolio_history_random[-1].item()
plt.plot(x_random, y_random, marker='o', color='green')
plt.text(x_random + x_pad, y_random + y_pad, f"{y_random:.2f}", color='green', va='bottom', ha='center')


plt.title(
    "Comparison of Trading Strategies: Portfolio Value Over Time\n"
    f"Tickers: {test_tickers} | Date Range: {start_date} - {end_date}"
)

plt.xlabel("Time Step (Days)")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("portfolio.png")
#Save model

save_model(model)