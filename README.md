# Reinforcement Learning for Portfolio Management

This project demonstrates the use of reinforcement learning (RL) to manage a stock portfolio with realistic trading constraints. The solution leverages a custom Gymnasium environment, a tailored feature extractor for PPO (Proximal Policy Optimization) using Stable Baselines3, and simulates trading with realistic elements such as trade fees, position limits, drawdown penalties, and periodic rebalancing.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Environment Parameters and Trading Constraints](#environment-parameters-and-trading-constraints)
- [Data Preparation and Feature Engineering](#data-preparation-and-feature-engineering)
- [Model Definition](#model-definition)
- [Training and RL Simulation Loop](#training-and-rl-simulation-loop)
- [Backtesting and Visualization](#backtesting-and-visualization)
- [Usage](#usage)

## Overview

This repository contains a Python script that:
- **Simulates a trading environment:** A custom Gymnasium environment (`PortfolioEnv`) downloads historical stock data via *yfinance*, engineers technical indicators, and simulates trading with realistic constraints.
- **Defines a custom feature extractor:** A simple neural network built with PyTorch processes raw observations into a suitable representation for the PPO agent.
- **Trains an RL agent:** Using Stable Baselines3’s PPO algorithm, the agent learns optimal trading strategies over multiple timesteps.
- **Backtests performance:** The trained agent is evaluated on a separate test dataset, and portfolio performance is visualized.

## Project Structure

- **Custom Feature Extractor:**  
  Implements a fully connected network (two linear layers with ReLU activations) that reduces the raw observation space into a 128-dimensional feature vector for the PPO model.

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
- gymnasium
- numpy
- pandas
- yfinance
- matplotlib
- stable_baselines3
- torch

Install the required packages using pip:

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

- Custom Feature Extractor (`CustomExtractor`):

    A PyTorch neural network with:
    - An input layer matching the observation space dimensions.
    - Two fully connected layers with ReLU activations.
    - An output feature dimension of 128.

- Policy Keyword Arguments:
    
    The PPO model is configured to use the custom feature extractor, which improves the processing of raw observations from the environment.

## Training and RL Simulation Loop

1. Environment Initialization:
    
    The `PortfolioEnv` is instantiated with parameters such as tickers, date range, initial balance, trading fee, maximum shares per trade/ticker, drawdown constraints, and a rebalance period.

2. Environment Validation:
    
    The environment is validated using `check_env` from Stable Baselines3 to ensure compliance with Gymnasium’s API.

3. Model Training:
    
    The PPO model is trained for 10,000 timesteps, during which the agent learns to balance its portfolio through buy, sell, and hold actions based on market observations.

4. RL Simulation Loop:
    
    After training, the simulation loop is executed in a test environment:
    - The test environment (with a new set of tickers and a different simulation period) is reset.
    - At each timestep, the trained model predicts an action based on the current observation.
    - The environment processes the action, updates the portfolio, and calculates rewards.
    - This loop continues until the simulation completes.

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