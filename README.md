# Reinforcement Learning for Portfolio Management

This project demonstrates the use of reinforcement learning (RL) to manage a stock portfolio with realistic trading constraints. The solution leverages a custom Gymnasium environment, a tailored feature extractor for PPO (Proximal Policy Optimization) using Stable Baselines3, and simulates trading with realistic elements such as trade fees, position limits, drawdown penalties, and periodic rebalancing.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
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

Once the simulation loop ends, the script:

- Records the portfolio value over each timestep.
- Uses matplotlib to plot the portfolio’s value trajectory, providing visual insight into the agent’s performance throughout the trading period.

## Usage
To run the script:

1. Configure the parameters (tickers, date ranges, trading constraints) as needed.
2. Execute the script in your Python environment.
3. The script will train the PPO model, perform backtesting, and display a plot of the portfolio value over time.
