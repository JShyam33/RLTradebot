# Reinforcement Learning for Portfolio Management

This project demonstrates the use of reinforcement learning (RL) to manage a stock portfolio with realistic trading constraints. The solution leverages a custom Gymnasium environment, a tailored feature extractor for PPO (Proximal Policy Optimization) using Stable Baselines3, and simulates trading with realistic elements such as trade fees, position limits, drawdown penalties, and periodic rebalancing.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Environment Parameters and Trading Constraints](#environment-parameters-and-trading-constraints)
- [Data Preparation and Feature Engineering](#data-preparation-and-feature-engineering)
- [Model Definition](#model-definition)
- [Training and Environment step function](#training-and-environment-step-function)
- [Backtesting and Visualization](#backtesting-and-visualization)
- [Usage](#usage)

## Overview

This repository contains a Python script that:
- **Simulates a trading environment:** A custom Gymnasium environment (`PortfolioEnv`) downloads historical stock data via *yfinance*, engineers technical indicators, and simulates trading with realistic constraints.
- **Defines a custom feature extractor:** A simple neural network built with PyTorch processes raw observations into a suitable representation for the PPO agent.
- **Trains an RL agent:** Using Stable Baselines3’s PPO algorithm, the agent learns optimal trading strategies over multiple timesteps.
- **Backtests performance:** The trained agent is evaluated on a separate test dataset, and portfolio performance is visualized.
The solution's constraints, such as trade fees and position limits, are designed to mirror real-world trading, ensuring that the model learns strategies that are both realistic and applicable to actual market conditions.

## Project Structure

- **Custom Feature Extractor:**  
  Implements a fully connected network (five linear layers with ReLU activations) that reduces the raw observation space into a 128-dimensional feature vector for the PPO model.

- **Custom Portfolio Environment (`PortfolioEnv`):**  
  - Downloads historical stock data for provided tickers within a specified date range.
  - Performs feature engineering to calculate technical indicators (e.g., moving averages, returns, spread, normalized volume, and volatility).
  - Simulates realistic trading by managing a cash balance, share holdings, trade fees, maximum shares per trade/ticker, drawdown penalties, and periodic portfolio rebalancing.

- **PPO Model Training:**  
  The PPO agent is set up with a custom policy that integrates the custom feature extractor. It is trained on the environment over 10,000 timesteps.

- **Backtesting Module:**  
  After training, the RL agent is evaluated on a test environment (with a different set of tickers and a shorter simulation period). The portfolio’s performance over time is recorded and plotted using *matplotlib*.

## Environment Setup

Ensure you have the following Python libraries installed:
- gymnasium: For creating and interacting with custom environments
- numpy : For numerical operations
- pandas : For handling and processing stock data
- yfinance : For downloading historical stock data
- matplotlib : For plotting and visualizing portfolio performance
- stable_baselines3 : For reinforcement learning algorithms like PPO
- torch : For defining and training neural networks (PyTorch)

Install the required packages mentioned in the requiremnets.txt using pip:

```bash
pip install gymnasium numpy pandas yfinance matplotlib stable-baselines3 torch
```

## Environment Parameters and Trading Constraints

The `PortfolioEnv` environment includes several configurable parameters to simulate realistic trading scenarios. Below are details on key variables and their impact:

**Trading Constraints**

1. **Trade Fees** (`trade_fee_percentage`)
   - A transaction fee is applied to both buy and sell actions.
   - Example: If `trade_fee_percentage = 0.001` (0.1%), buying 10,000 of shares incurs a 10 fee. Selling the same amount deducts $10 from the revenue.
   - Fees reduce the available cash balance and discourage excessive trading.

2. **Maximum shares per Trade** (`max_shares_per_trade`)
   - Limits the number of shares that can be bought or sold in a single transaction.
   - Example: If `max_shares_per_trade = 50`, the agent cannot trade more than 50 shares of any ticker per step.
   - Prevents large, market-impacting trades and encourages incremental position adjustments.

3. **Maximum Shares per Ticker** (`max_shares_per_ticker`)
   - Caps the total number of shares the agent can hold for any single ticker.
   - Example: If `max_shares_per_ticker = 200`, the agent cannot hold more than 200 shares of a single stock at any time.
   - Mitigates over-concentration risk in a single asset.

**Risk Management**

4. **Drawdown Penalty** (`drawdown_limit`, `drawdown_penalty_factor`)
   - **Drawdown Limit**: If the portfolio value falls below a specified percentage of its peak value (`drawdown_limit`), a penalty is applied. 

      Example: A `drawdown_limit = 0.2` triggers a penalty if the portfolio drops 20% below its peak.
   - **Penalty Factor**: The penalty scales with the severity of the drawdown. For instance, a `drawdown_penalty_factor = 0.5` reduces the reward by `(drawdown - drawdown_limit) * portfolio_value * 0.5`.
   - Encourages the agent to prioritize capital preservation during downturns.

**Portfolio Rebalancing**

5. **Rebalancing Period** (`rebalance_period`)
   - At fixed intervals (e.g., every 5 days if `rebalance_period = 5`), the portfolio is rebalanced to a target allocation of 80% equities (evenly distributed across tickers) and 20% cash.
   - **Process:**
     - **Selling Over-Allocated Assets**: If a ticker’s equity value exceeds its target allocation, excess shares are sold.
     - **Buying Under-Allocated Assets**: If a ticker’s equity value is below target, shares are purchased using cash reserves above the 20% target.
   - Ensures disciplined risk management and reduces drift from the target allocation.
   - Rebalancing respects trading constraints (fees, max shares per trade/ticker).

## Data Preparation and Feature Engineering

The environment downloads historical stock data from Yahoo Finance. For each stock ticker, the script computes and adds several engineered features:

- Moving Averages: 5-day (`MA5`) and 10-day (`MA10`) moving averages of the closing price.
- Daily Return: The percentage change of the closing price.
- Spread: The difference between the high and low prices.
- Normalized Volume: The ratio of the current volume to its 5-day moving average.
- Volatility: The rolling standard deviation of returns over a 5-day window.

These features are combined with the current portfolio state (balance and holdings) to form the observation space provided to the RL agent.

## Model Definition

The PPO model is defined using Stable Baselines3 with the following customizations:

- Custom Actor Critic Policy (`CustomActorCriticPolicy`):
 
    A PyTorch neural network for feature extraction with:
    - An input layer matching the observation space dimensions.
    - An output feature dimension of 128.    

    A PyTorch neural network for actor with:
    - An input layer matching the observation space dimensions.
    - three fully connected layers with ReLU activations.
    - An output feature dimension of 128.
  
    A PyTorch neural network for critic with:
    - An input layer matching the observation space dimensions.
    - two fully connected layers with ReLU activations.
    - An output feature dimension of 1.

- Policy Keyword Arguments:
    
    The PPO model is configured to use the custom feature extractor, which improves the processing of raw observations from the environment.

### Training and Environment step function

#### Proximal Policy Optimization (PPO) Overview

PPO is an on-policy actor-critic algorithm designed to provide stable and efficient learning by optimizing a clipped surrogate objective. Key details include:

- **Clipped Objective:** Prevents excessively large policy updates by clipping the probability ratio between the new and old policies.
- **Multiple Epochs:** Uses several passes over the same batch of experiences to improve sample efficiency.
- **Actor-Critic Structure:** Separates the policy network (actor) for decision-making from the value network (critic) for state-value estimation.
- **Entropy Bonus:** Encourages exploration by adding an entropy term to the loss function, which avoids premature convergence to suboptimal deterministic policies.

Below is an excerpt showing how PPO is configured and trained in this project:

```python
policy_kwargs = dict(
actor_net_arch= [128,1024,512,256, 128],
    critic_net_arch = [256,512,256, 128],
    net_arch=[dict(pi=[128,1024,512,256, 128], vf=[256,512,256, 128])],
    features_extractor_class=CustomExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)
model = PPO(CustomActorCriticPolicy, env, policy_kwargs=policy_kwargs, verbose=1,
            n_epochs=30, device="cuda", batch_size=128)
model.learn(total_timesteps=10000)
```

Environment step function

### Detailed Explanation of the Environment `step` Function

The `step` function is the core of the trading simulation. It processes the agent’s actions, updates the portfolio, computes rewards, and advances the simulation. Below is a detailed breakdown:

1. **Action Processing and Clipping**  
   - The input action (an array with one entry per ticker) is first clipped to the valid range \([-1, 1]\).  
   - Positive values indicate a buy action, negative values indicate a sell action, and values near zero imply holding the position.

2. **Price Retrieval**  
   - For each ticker, the current closing price is extracted from the historical data based on the current timestep.
   - These prices are used to determine the cost or revenue of the transactions.
```python
    # Retrieve current prices for each ticker using the Close price
    prices = {}
    for ticker in self.tickers:
        df = self.data[ticker]
        price = df.iloc[self.current_step]['Close'].values
        prices[ticker] = price
```

3. **Trade Execution Logic**  
   - **Buying:**  
     - **Desired Shares:** The number of shares to buy is calculated by taking a fraction (given by the action value) of the maximum affordable shares (based on the current balance and price).  
     - **Limits:** The desired number of shares is capped by both the maximum shares allowed per trade and the maximum shares allowed per ticker.  
     - **Transaction Cost:** The total cost includes both the purchase cost (price × shares) and the associated trading fee.  
     - **Portfolio Update:** If there is enough balance, the holdings are increased, and the balance is decreased accordingly.
     ```python
            max_affordable_shares = int(self.balance // price)
            desired_shares = int(max_affordable_shares * act)
            desired_shares = min(desired_shares, self.max_shares_per_trade)
            allowed_shares = self.max_shares_per_ticker - self.holdings[ticker]
            shares_to_buy = min(desired_shares, allowed_shares)
            cost = shares_to_buy * price
            fee = cost * self.trade_fee_percentage
            total_cost = cost + fee
            if shares_to_buy > 0 and total_cost <= self.balance:
                self.holdings[ticker] += shares_to_buy
                self.balance -= total_cost
        ```
   - **Selling:**  
     - **Desired Shares:** A fraction (absolute value of the action) of the currently held shares is considered for sale.  
     - **Limits:** The sale is capped by the maximum shares allowed per trade and the available holdings.  
     - **Transaction Revenue:** Revenue is computed, fees are deducted, and the net revenue is added back to the balance after reducing the holdings.
     ```python
            desired_shares = int(self.holdings[ticker] * abs(act))
            desired_shares = min(desired_shares, self.max_shares_per_trade)
            shares_to_sell = min(desired_shares, self.holdings[ticker])
            revenue = shares_to_sell * price
            fee = revenue * self.trade_fee_percentage
            net_revenue = revenue - fee
            if shares_to_sell > 0:
                self.holdings[ticker] -= shares_to_sell
                self.balance += net_revenue
        ```
   - **Hold:**  
     - When the action is near zero, no trading occurs.

4. **Portfolio Value Calculation and Reward Determination**  
   - **Portfolio Value:** Computed as the sum of the cash balance and the market value of all held shares.
   - **Peak Value Update:** The highest portfolio value reached so far is tracked to calculate drawdowns.
   - **Drawdown & Penalty:**  
     - Drawdown is the percentage drop from the peak value.  
     - If the drawdown exceeds a preset limit, a penalty is applied to the reward.
    ```python 
     # Compute drawdown and adjust reward if it exceeds the allowed limit
    drawdown = (self.peak_value - portfolio_value) / self.peak_value
    reward = (portfolio_value - self.prev_value) * self.reward_factor
    if drawdown > self.drawdown_limit:
        penalty = (drawdown - self.drawdown_limit) * portfolio_value * self.drawdown_penalty_factor
        reward -= penalty
    ```
   - **Reward:** The reward is the change in portfolio value (scaled by a reward factor) minus any drawdown penalty.

5. **Periodic Rebalancing (Optional)**  
   - If enabled (via `rebalance_period`), the portfolio is rebalanced at regular intervals to maintain a target allocation between equities and cash.
   - After rebalancing, the portfolio value is recalculated.

6. **State Update and Return**  
   - The portfolio history and current timestep are updated.
   - A new observation is generated (or a zeroed observation is returned if the simulation is done).
   - The function returns the new observation, the reward, a done flag (indicating the end of the simulation), and an empty info dictionary.

This function effectively simulates a day's trading by:

- Interpreting and executing the action,

- Updating the portfolio based on market prices and realistic trading constraints,

- Calculating the reward (with risk adjustments), and

- Advancing the simulation state.



## Backtesting and Visualization

The trained RL agent is evaluated against two benchmark strategies:

1. **Rule-Based Strategy (Moving Average Crossover):**
   - Uses technical indicators to make trading decisions
   - Trading rules:
     - If 5-day moving average (MA5) > 10-day moving average (MA10): Buy (action=1)
     - If MA5 < MA10: Sell (action=-1)
     - Otherwise: Hold (action=0)
   - This strategy represents a common technical analysis approach used by traders

2. **Random Strategy:**
   - Takes random actions with an 80% probability of trading (20% probability of holding)
   - When trading, actions are sampled uniformly from the range [-1, 1]
   - Serves as a baseline to demonstrate that the RL strategy performs better than random decisions

The performance of all three strategies (RL, Rule-Based, and Random) is visualized in a single plot showing portfolio value over time. This comparison demonstrates the effectiveness of the RL approach against traditional and random strategies.

Once the simulation loop ends, the script:
- Records the portfolio value over each timestep for all three strategies
- Plots the portfolio trajectories with final values labeled
- Visualizes the comparative performance to evaluate the RL agent's effectiveness

## Usage

### Basic Execution

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

### Configuring Trading Parameters

To customize the trading environment, modify the following parameters in `main.py`:

#### Training Environment
```python
tickers = ['TSLA', 'GME', 'NVDA']  # Stocks to include in portfolio
env = PortfolioEnv(
    tickers=tickers,
    start_date='2023-01-01',       # Training data start date
    end_date='2024-01-01',         # Training data end date
    initial_balance=10000,         # Starting cash
    trade_fee_percentage=0.001,    # 0.1% transaction fee
    max_shares_per_trade=50,       # Maximum shares per transaction
    max_shares_per_ticker=200,     # Maximum position size
    drawdown_limit=0.2,            # 20% drawdown limit
    drawdown_penalty_factor=0.3,   # Penalty scaling factor
    reward_factor=5,               # Reward scaling factor
    rebalance_period=-1            # Disable automatic rebalancing
)
```
