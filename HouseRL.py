import random
from collections import defaultdict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ============================================================
# SEEDS
# ============================================================

random.seed(42)
np.random.seed(42)

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "HousePricesDataset/MLS.csv"
AREA_COL = "Location"
TYPE_COL = "PropertyType"
DATE_COL = "Date"
PRICE_COL = "BenchmarkPrice"

INITIAL_CASH = 1_000_000
TRANSACTION_COST = 5_000
HOLDING_COST_MONTHLY = 500

# Q-learning
EPISODES = 300
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05

# PPO
PPO_TIMESTEPS = 50_000

ACTIONS = {
    0: "BUY",
    1: "SELL",
    2: "HOLD",
}

# ============================================================
# LOAD + PREP DATA
# ============================================================

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])

    property_frames = []

    mapping = {
        "Composite": ("CompIndex", "CompBenchmark", "CompYoYChange"),
        "Detached": ("SFDetachIndex", "SFDetachBenchmark", "SFDetachYoYChange"),
        "Attached": ("SFAttachIndex", "SFAttachBenchmark", "SFAttachYoYChange"),
        "Townhouse": ("THouseIndex", "THouseBenchmark", "THouseYoYChange"),
        "Apartment": ("ApartIndex", "ApartBenchmark", "ApartYoYChange"),
    }

    for prop_type, (index_col, benchmark_col, yoy_col) in mapping.items():
        temp = df[["Location", "Date", index_col, benchmark_col, yoy_col]].copy()
        temp.columns = ["Location", "Date", "Index", "BenchmarkPrice", "YoYChange"]
        temp["PropertyType"] = prop_type
        property_frames.append(temp)

    df = pd.concat(property_frames, ignore_index=True)
    df = df.sort_values(["Location", "PropertyType", "Date"]).reset_index(drop=True)

    df["return_1m"] = (
        df.groupby(["Location", "PropertyType"])["BenchmarkPrice"]
        .pct_change(fill_method=None)
    )

    df["ma_3"] = (
        df.groupby(["Location", "PropertyType"])["BenchmarkPrice"]
        .rolling(3)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    df["ma_6"] = (
        df.groupby(["Location", "PropertyType"])["BenchmarkPrice"]
        .rolling(6)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    df["momentum_3"] = (df["BenchmarkPrice"] / df["ma_3"]) - 1
    df["momentum_6"] = (df["BenchmarkPrice"] / df["ma_6"]) - 1
    df["YoYChange"] = df["YoYChange"] / 100.0

    df = df.dropna().reset_index(drop=True)
    return df

# ============================================================
# DISCRETIZATION FOR Q-LEARNING
# ============================================================

def bucket_return(x: float) -> str:
    if x < -0.02:
        return "DOWN_BIG"
    if x < 0:
        return "DOWN_SMALL"
    if x < 0.02:
        return "UP_SMALL"
    return "UP_BIG"


def bucket_momentum(x: float) -> str:
    if x < -0.02:
        return "WEAK"
    if x < 0.02:
        return "NEUTRAL"
    return "STRONG"

# ============================================================
# TABULAR ENV FOR Q-LEARNING
# ============================================================

class HousingEnv:
    def __init__(self, df: pd.DataFrame, selected_area: str, selected_type: str):
        self.df = df[
            (df[AREA_COL] == selected_area) &
            (df[TYPE_COL] == selected_type)
        ].copy().reset_index(drop=True)

        if len(self.df) < 12:
            raise ValueError("Not enough data for this area/property type.")

        self.selected_area = selected_area
        self.selected_type = selected_type
        self.reset()

    def reset(self):
        self.idx = 0
        self.cash = INITIAL_CASH
        self.holding = 0
        self.buy_price = 0.0
        self.done = False
        return self._get_state()

    def _get_state(self):
        row = self.df.iloc[self.idx]
        return (
            bucket_return(float(row["return_1m"])),
            bucket_momentum(float(row["momentum_3"])),
            bucket_momentum(float(row["momentum_6"])),
            self.holding,
        )

    def _current_price(self) -> float:
        return float(self.df.iloc[self.idx][PRICE_COL])

    def _portfolio_value(self, price: float) -> float:
        return float(self.cash + (self.holding * price))

    def step(self, action: int):
        if self.done:
            raise ValueError("Episode already finished.")

        current_price = self._current_price()
        old_value = self._portfolio_value(current_price)

        if action == 0:  # BUY
            if self.holding == 0 and self.cash >= current_price + TRANSACTION_COST:
                self.cash -= (current_price + TRANSACTION_COST)
                self.holding = 1
                self.buy_price = current_price
            else:
                reward = -2000.0
                self.idx += 1
                if self.idx >= len(self.df) - 1:
                    self.done = True
                return self._get_state(), reward, self.done

        elif action == 1:  # SELL
            if self.holding == 1:
                self.cash += (current_price - TRANSACTION_COST)
                self.holding = 0
                self.buy_price = 0.0
            else:
                reward = -2000.0
                self.idx += 1
                if self.idx >= len(self.df) - 1:
                    self.done = True
                return self._get_state(), reward, self.done

        elif action == 2:  # HOLD
            pass

        if self.holding == 1:
            self.cash -= HOLDING_COST_MONTHLY

        self.idx += 1
        if self.idx >= len(self.df) - 1:
            self.done = True

        next_price = self._current_price()
        new_value = self._portfolio_value(next_price)

        reward = float(new_value - old_value)
        next_state = self._get_state()
        return next_state, reward, self.done

# ============================================================
# Q-LEARNING AGENT
# ============================================================

class QLearningAgent:
    def __init__(self):
        self.q_table = defaultdict(lambda: np.zeros(len(ACTIONS), dtype=np.float64))

    def choose_action(self, state, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.choice(list(ACTIONS.keys()))
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action: int, reward: float, next_state):
        best_next = np.max(self.q_table[next_state])
        td_target = reward + GAMMA * best_next
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += ALPHA * td_error


def train_q_learning(df: pd.DataFrame, selected_area: str, selected_type: str):
    env = HousingEnv(df, selected_area, selected_type)
    agent = QLearningAgent()

    epsilon = EPSILON
    rewards_per_episode = []

    for _ in range(EPISODES):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.choose_action(state, epsilon)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            total_reward += reward

        rewards_per_episode.append(total_reward)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

    return agent, rewards_per_episode


def evaluate_q_learning(agent: QLearningAgent, df: pd.DataFrame, selected_area: str, selected_type: str):
    env = HousingEnv(df, selected_area, selected_type)
    state = env.reset()

    dates = [env.df.iloc[env.idx][DATE_COL]]
    values = [env._portfolio_value(env._current_price())]

    done = False
    while not done:
        action = int(np.argmax(agent.q_table[state]))
        next_state, reward, done = env.step(action)
        state = next_state

        dates.append(env.df.iloc[env.idx][DATE_COL])
        values.append(env._portfolio_value(env._current_price()))

    return pd.DataFrame({
        "Date": dates,
        "PortfolioValue": values,
        "Model": "Q-Learning",
    })

# ============================================================
# GYM ENV FOR PPO
# ============================================================

class HousingGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, selected_area: str, selected_type: str):
        super().__init__()

        self.df = df[
            (df[AREA_COL] == selected_area) &
            (df[TYPE_COL] == selected_type)
        ].copy().reset_index(drop=True)

        if len(self.df) < 12:
            raise ValueError("Not enough data for this area/property type.")

        self.action_space = spaces.Discrete(3)

        # [return_1m, momentum_3, momentum_6, yoy_change, price_rel,
        #  holding, cash_ratio, unrealized_pnl, can_buy, can_sell]
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(10,),
            dtype=np.float32,
        )

        self.reset()

    def _current_price(self) -> float:
        return float(self.df.iloc[self.idx][PRICE_COL])

    def _portfolio_value(self, price: float) -> float:
        return float(self.cash + (self.holding * price))

    def _can_buy(self) -> bool:
        return self.holding == 0 and self.cash >= self._current_price() + TRANSACTION_COST

    def _can_sell(self) -> bool:
        return self.holding == 1

    def _get_obs(self) -> np.ndarray:
        row = self.df.iloc[self.idx]
        current_price = float(row[PRICE_COL])
        initial_price = float(self.df.iloc[0][PRICE_COL])

        price_rel = (current_price / initial_price) - 1.0

        if self.holding == 1 and self.buy_price > 0:
            unrealized = (current_price / self.buy_price) - 1.0
        else:
            unrealized = 0.0

        obs = np.array([
            float(np.clip(row["return_1m"], -1.0, 1.0)),
            float(np.clip(row["momentum_3"], -1.0, 1.0)),
            float(np.clip(row["momentum_6"], -1.0, 1.0)),
            float(np.clip(row["YoYChange"], -1.0, 1.0)),
            float(np.clip(price_rel, -2.0, 2.0)),
            float(self.holding),
            float(np.clip(self.cash / INITIAL_CASH, 0.0, 5.0)),
            float(np.clip(unrealized, -1.0, 1.0)),
            float(self._can_buy()),
            float(self._can_sell()),
        ], dtype=np.float32)

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        self.cash = INITIAL_CASH
        self.holding = 0
        self.buy_price = 0.0
        return self._get_obs(), {}

    def step(self, action: int):
        current_price = self._current_price()
        old_value = self._portfolio_value(current_price)

        invalid_penalty = 0.0

        if action == 0:  # BUY
            if self._can_buy():
                self.cash -= (current_price + TRANSACTION_COST)
                self.holding = 1
                self.buy_price = current_price
            else:
                invalid_penalty = -1.0

        elif action == 1:  # SELL
            if self._can_sell():
                self.cash += (current_price - TRANSACTION_COST)
                self.holding = 0
                self.buy_price = 0.0
            else:
                invalid_penalty = -1.0

        elif action == 2:  # HOLD
            pass

        if self.holding == 1:
            self.cash -= HOLDING_COST_MONTHLY

        self.idx += 1
        terminated = self.idx >= len(self.df) - 1
        truncated = False

        next_price = self._current_price()
        new_value = self._portfolio_value(next_price)

        # PPO learns better from scaled returns than raw dollars
        reward = 100.0 * ((new_value - old_value) / max(old_value, 1.0))
        reward += invalid_penalty

        if terminated:
            return np.zeros(self.observation_space.shape, dtype=np.float32), float(reward), terminated, truncated, {}

        return self._get_obs(), float(reward), terminated, truncated, {}

# ============================================================
# PPO TRAIN / EVAL
# ============================================================

def make_ppo_env(df: pd.DataFrame, selected_area: str, selected_type: str):
    def _init():
        return HousingGymEnv(df, selected_area, selected_type)
    return _init


def train_ppo(df: pd.DataFrame, selected_area: str, selected_type: str):
    vec_env = DummyVecEnv([make_ppo_env(df, selected_area, selected_type)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=32,
        batch_size=32,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,
        clip_range=0.2,
        n_epochs=10,
        verbose=0,
        seed=42,
    )

    model.learn(total_timesteps=PPO_TIMESTEPS)
    return model, vec_env


def evaluate_ppo(model: PPO, vec_norm: VecNormalize, df: pd.DataFrame, selected_area: str, selected_type: str):
    raw_env = HousingGymEnv(df, selected_area, selected_type)
    obs, _ = raw_env.reset()

    dates = [raw_env.df.iloc[raw_env.idx][DATE_COL]]
    values = [raw_env._portfolio_value(raw_env._current_price())]

    done = False
    while not done:
        norm_obs = vec_norm.normalize_obs(obs.reshape(1, -1))
        action, _ = model.predict(norm_obs, deterministic=True)

        obs, reward, terminated, truncated, _ = raw_env.step(int(action[0]))
        done = terminated or truncated

        dates.append(raw_env.df.iloc[raw_env.idx][DATE_COL])
        values.append(raw_env._portfolio_value(raw_env._current_price()))

    return pd.DataFrame({
        "Date": dates,
        "PortfolioValue": values,
        "Model": "PPO",
    })

# ============================================================
# BUY-AND-HOLD BASELINE
# ============================================================

def evaluate_buy_and_hold(df: pd.DataFrame, selected_area: str, selected_type: str):
    subset = df[
        (df[AREA_COL] == selected_area) &
        (df[TYPE_COL] == selected_type)
    ].copy().reset_index(drop=True)

    first_price = float(subset.iloc[0][PRICE_COL])

    if INITIAL_CASH < first_price + TRANSACTION_COST:
        raise ValueError("Initial cash is too low to buy one unit for buy-and-hold.")

    cash = INITIAL_CASH - first_price - TRANSACTION_COST
    holding = 1

    dates = []
    values = []

    for _, row in subset.iterrows():
        price = float(row[PRICE_COL])
        cash -= HOLDING_COST_MONTHLY
        portfolio_value = cash + (holding * price)

        dates.append(row[DATE_COL])
        values.append(portfolio_value)

    return pd.DataFrame({
        "Date": dates,
        "PortfolioValue": values,
        "Model": "Buy-and-Hold",
    })

# ============================================================
# PLOT
# ============================================================

def plot_final_comparison(q_df: pd.DataFrame, ppo_df: pd.DataFrame, bh_df: pd.DataFrame, area: str, property_type: str):
    plt.figure(figsize=(11, 6))
    plt.plot(q_df["Date"], q_df["PortfolioValue"], label="Q-Learning")
    plt.plot(ppo_df["Date"], ppo_df["PortfolioValue"], label="PPO")
    plt.plot(bh_df["Date"], bh_df["PortfolioValue"], label="Buy-and-Hold")

    plt.title(f"Portfolio Value Comparison: {property_type} in {area}")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("final_comparison.png", dpi=200)
    plt.show()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    df = load_data(CSV_PATH)

    area = "City of Toronto"
    property_type = "Detached"

    print(f"Training Q-Learning for {property_type} in {area}...")
    q_agent, q_rewards = train_q_learning(df, area, property_type)

    print(f"Training PPO for {property_type} in {area}...")
    ppo_model, ppo_vec_norm = train_ppo(df, area, property_type)

    print("Evaluating models...")
    q_eval = evaluate_q_learning(q_agent, df, area, property_type)
    ppo_eval = evaluate_ppo(ppo_model, ppo_vec_norm, df, area, property_type)
    bh_eval = evaluate_buy_and_hold(df, area, property_type)

    print("\nFinal portfolio values:")
    print(f"Q-Learning:   ${q_eval['PortfolioValue'].iloc[-1]:,.2f}")
    print(f"PPO:          ${ppo_eval['PortfolioValue'].iloc[-1]:,.2f}")
    print(f"Buy-and-Hold: ${bh_eval['PortfolioValue'].iloc[-1]:,.2f}")

    plot_final_comparison(q_eval, ppo_eval, bh_eval, area, property_type)