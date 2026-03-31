import random
from collections import defaultdict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO, A2C
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
INVALID_ACTION_PENALTY = 1.0

# Q-learning
EPISODES = 300
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05

# PPO / A2C
PPO_TIMESTEPS = 50_000
A2C_TIMESTEPS = 150_000

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

    def _current_price(self) -> float:
        return float(self.df.iloc[self.idx][PRICE_COL])

    def _portfolio_value(self, price: float) -> float:
        return float(self.cash + (self.holding * price))

    def _can_buy(self) -> bool:
        return self.holding == 0 and self.cash >= self._current_price() + TRANSACTION_COST

    def _can_sell(self) -> bool:
        return self.holding == 1

    def valid_actions(self):
        actions = [2]  # HOLD always valid
        if self._can_buy():
            actions.append(0)
        if self._can_sell():
            actions.append(1)
        return actions

    def _get_state(self):
        row = self.df.iloc[self.idx]
        return (
            bucket_return(float(row["return_1m"])),
            bucket_momentum(float(row["momentum_3"])),
            bucket_momentum(float(row["momentum_6"])),
            self.holding,
            int(self._can_buy()),
            int(self._can_sell()),
        )

    def step(self, action: int):
        if self.done:
            raise ValueError("Episode already finished.")

        current_price = self._current_price()
        old_value = self._portfolio_value(current_price)
        invalid_penalty = 0.0

        if action == 0:  # BUY
            if self._can_buy():
                self.cash -= (current_price + TRANSACTION_COST)
                self.holding = 1
                self.buy_price = current_price
            else:
                invalid_penalty = -INVALID_ACTION_PENALTY

        elif action == 1:  # SELL
            if self._can_sell():
                self.cash += (current_price - TRANSACTION_COST)
                self.holding = 0
                self.buy_price = 0.0
            else:
                invalid_penalty = -INVALID_ACTION_PENALTY

        elif action == 2:  # HOLD
            pass

        if self.holding == 1:
            self.cash -= HOLDING_COST_MONTHLY

        self.idx += 1
        if self.idx >= len(self.df) - 1:
            self.done = True

        next_price = self._current_price()
        new_value = self._portfolio_value(next_price)

        reward = 100.0 * ((new_value - old_value) / max(old_value, 1.0)) + invalid_penalty
        next_state = self._get_state()
        return next_state, reward, self.done

# ============================================================
# Q-LEARNING AGENT
# ============================================================

class QLearningAgent:
    def __init__(self):
        self.q_table = defaultdict(lambda: np.zeros(len(ACTIONS), dtype=np.float64))

    def choose_action(self, state, epsilon: float, valid_actions):
        if random.random() < epsilon:
            return random.choice(valid_actions)

        q_values = self.q_table[state]
        best_action = valid_actions[0]
        best_value = q_values[best_action]

        for action in valid_actions[1:]:
            if q_values[action] > best_value:
                best_value = q_values[action]
                best_action = action

        return int(best_action)

    def update(self, state, action: int, reward: float, next_state, next_valid_actions, done: bool):
        if done:
            td_target = reward
        else:
            best_next = max(self.q_table[next_state][a] for a in next_valid_actions)
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
            valid_actions = env.valid_actions()
            action = agent.choose_action(state, epsilon, valid_actions)

            next_state, reward, done = env.step(action)
            next_valid_actions = env.valid_actions() if not done else [2]

            agent.update(state, action, reward, next_state, next_valid_actions, done)

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
    actions = ["START"]

    action_counts = {a: 0 for a in ACTIONS.values()}

    done = False
    while not done:
        action = int(np.argmax(agent.q_table[state]))
        action_name = ACTIONS[action]

        valid_action = (
            (action == 0 and env.holding == 0 and env.cash >= env._current_price() + TRANSACTION_COST) or
            (action == 1 and env.holding == 1) or
            (action == 2)
        )

        next_state, reward, done = env.step(action)
        state = next_state

        if valid_action:
            action_counts[action_name] += 1
            logged_action = action_name
        else:
            logged_action = f"INVALID_{action_name}"

        dates.append(env.df.iloc[env.idx][DATE_COL])
        values.append(env._portfolio_value(env._current_price()))
        actions.append(logged_action)

    total = sum(action_counts.values())
    action_percent = {k: (v / total) * 100 for k, v in action_counts.items()} if total > 0 else {k: 0.0 for k in action_counts}

    print("\nQ-Learning Action Distribution (valid only):")
    print(action_counts)
    print({k: f"{v:.2f}%" for k, v in action_percent.items()})

    return pd.DataFrame({
        "Date": dates,
        "PortfolioValue": values,
        "Action": actions,
        "Model": "Q-Learning",
    })

# ============================================================
# GYM ENV FOR PPO / A2C
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
                invalid_penalty = -INVALID_ACTION_PENALTY

        elif action == 1:  # SELL
            if self._can_sell():
                self.cash += (current_price - TRANSACTION_COST)
                self.holding = 0
                self.buy_price = 0.0
            else:
                invalid_penalty = -INVALID_ACTION_PENALTY

        elif action == 2:  # HOLD
            pass

        if self.holding == 1:
            self.cash -= HOLDING_COST_MONTHLY

        self.idx += 1
        terminated = self.idx >= len(self.df) - 1
        truncated = False

        next_price = self._current_price()
        new_value = self._portfolio_value(next_price)

        reward = 100.0 * ((new_value - old_value) / max(old_value, 1.0))
        reward += invalid_penalty

        if terminated:
            zero_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return zero_obs, float(reward), terminated, truncated, {}

        return self._get_obs(), float(reward), terminated, truncated, {}

# ============================================================
# PPO / A2C TRAINING HELPERS
# ============================================================

def make_rl_env(df: pd.DataFrame, selected_area: str, selected_type: str):
    def _init():
        return HousingGymEnv(df, selected_area, selected_type)
    return _init

def train_ppo(df: pd.DataFrame, selected_area: str, selected_type: str):
    vec_env = DummyVecEnv([make_rl_env(df, selected_area, selected_type)])
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
        device="cpu"
    )

    model.learn(total_timesteps=PPO_TIMESTEPS)
    return model, vec_env

def train_a2c(df: pd.DataFrame, selected_area: str, selected_type: str):
    vec_env = DummyVecEnv([make_rl_env(df, selected_area, selected_type)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = A2C(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=8,
        gamma=0.95,
        gae_lambda=0.95,
        ent_coef=0.05,
        vf_coef=0.25,
        verbose=0,
        seed=42,
    )

    model.learn(total_timesteps=A2C_TIMESTEPS)
    return model, vec_env

def evaluate_policy_model(model, vec_norm: VecNormalize, df: pd.DataFrame, selected_area: str, selected_type: str, label: str):
    raw_env = HousingGymEnv(df, selected_area, selected_type)
    obs, _ = raw_env.reset()

    vec_norm.training = False
    vec_norm.norm_reward = False

    dates = [raw_env.df.iloc[raw_env.idx][DATE_COL]]
    values = [raw_env._portfolio_value(raw_env._current_price())]
    actions = ["START"]

    action_counts = {a: 0 for a in ACTIONS.values()}

    done = False
    while not done:
        norm_obs = vec_norm.normalize_obs(obs.reshape(1, -1))
        action, _ = model.predict(norm_obs, deterministic=True)

        action_int = int(action[0])
        action_name = ACTIONS[action_int]

        valid_action = (
            (action_int == 0 and raw_env._can_buy()) or
            (action_int == 1 and raw_env._can_sell()) or
            (action_int == 2)
        )

        obs, reward, terminated, truncated, _ = raw_env.step(action_int)
        done = terminated or truncated

        if valid_action:
            action_counts[action_name] += 1

        dates.append(raw_env.df.iloc[raw_env.idx][DATE_COL])
        values.append(raw_env._portfolio_value(raw_env._current_price()))
        actions.append(action_name if valid_action else f"INVALID_{action_name}")

    total = sum(action_counts.values())
    if total > 0:
        action_percent = {k: (v / total) * 100 for k, v in action_counts.items()}
    else:
        action_percent = {k: 0.0 for k in action_counts}

    print(f"\n{label} Action Distribution (valid only):")
    print(action_counts)
    print({k: f"{v:.2f}%" for k, v in action_percent.items()})

    return pd.DataFrame({
        "Date": dates,
        "PortfolioValue": values,
        "Action": actions,
        "Model": label,
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

def plot_final_comparison(
    q_df: pd.DataFrame,
    ppo_df: pd.DataFrame,
    a2c_df: pd.DataFrame,
    bh_df: pd.DataFrame,
    area: str,
    property_type: str
):
    plt.figure(figsize=(11, 6))
    plt.plot(q_df["Date"], q_df["PortfolioValue"], label="Q-Learning")
    plt.plot(ppo_df["Date"], ppo_df["PortfolioValue"], label="PPO")
    plt.plot(a2c_df["Date"], a2c_df["PortfolioValue"], label="A2C")
    plt.plot(bh_df["Date"], bh_df["PortfolioValue"], label="Buy-and-Hold")

    plt.title(f"Portfolio Value Comparison: {property_type} in {area}")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("final_comparison.png", dpi=200)
    plt.show()

def plot_with_actions(df: pd.DataFrame, title: str, filename: str):
    plt.figure(figsize=(11, 6))

    plt.plot(df["Date"], df["PortfolioValue"], label=title)

    buy_points = df[df["Action"] == "BUY"]
    sell_points = df[df["Action"] == "SELL"]
    invalid_buy_points = df[df["Action"] == "INVALID_BUY"]
    invalid_sell_points = df[df["Action"] == "INVALID_SELL"]

    plt.scatter(buy_points["Date"], buy_points["PortfolioValue"], marker="^", label="BUY")
    plt.scatter(sell_points["Date"], sell_points["PortfolioValue"], marker="v", label="SELL")

    if not invalid_buy_points.empty:
        plt.scatter(
            invalid_buy_points["Date"],
            invalid_buy_points["PortfolioValue"],
            marker="x",
            label="INVALID_BUY"
        )

    if not invalid_sell_points.empty:
        plt.scatter(
            invalid_sell_points["Date"],
            invalid_sell_points["PortfolioValue"],
            marker="x",
            label="INVALID_SELL"
        )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()

    plt.savefig(filename, dpi=200)
    plt.show()

def plot_action_distribution(df: pd.DataFrame, title: str, filename: str):
    action_counts = df["Action"].value_counts()

    if "START" in action_counts:
        action_counts = action_counts.drop("START")

    total = action_counts.sum()
    percentages = (action_counts / total) * 100 if total > 0 else action_counts.astype(float)

    plt.figure(figsize=(6, 4))
    plt.bar(percentages.index, percentages.values)

    plt.title(f"{title} Action Distribution")
    plt.xlabel("Action")
    plt.ylabel("Percentage (%)")

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    df = load_data(CSV_PATH)

    area = "City of Toronto"
    property_type_list = ["Composite", "Detached", "Attached", "Townhouse", "Apartment"]

    for property_type in property_type_list:
        print(f"\n=== Training Q-Learning for {property_type} in {area} ===")
        q_agent, q_rewards = train_q_learning(df, area, property_type)

        print(f"=== Training PPO for {property_type} in {area} ===")
        ppo_model, ppo_vec_norm = train_ppo(df, area, property_type)

        print(f"=== Training A2C for {property_type} in {area} ===")
        a2c_model, a2c_vec_norm = train_a2c(df, area, property_type)

        print(f"=== Evaluating models for {property_type} in {area} ===")
        q_eval = evaluate_q_learning(q_agent, df, area, property_type)
        ppo_eval = evaluate_policy_model(ppo_model, ppo_vec_norm, df, area, property_type, "PPO")
        a2c_eval = evaluate_policy_model(a2c_model, a2c_vec_norm, df, area, property_type, "A2C")
        bh_eval = evaluate_buy_and_hold(df, area, property_type)

        print("\nFinal portfolio values:")
        print(f"Q-Learning:   ${q_eval['PortfolioValue'].iloc[-1]:,.2f}")
        print(f"PPO:          ${ppo_eval['PortfolioValue'].iloc[-1]:,.2f}")
        print(f"A2C:          ${a2c_eval['PortfolioValue'].iloc[-1]:,.2f}")
        print(f"Buy-and-Hold: ${bh_eval['PortfolioValue'].iloc[-1]:,.2f}")

        plot_final_comparison(q_eval, ppo_eval, a2c_eval, bh_eval, area, property_type)

        plot_with_actions(q_eval, f"Q-Learning Actions: {property_type}", f"q_actions_{property_type}.png")
        plot_with_actions(ppo_eval, f"PPO Actions: {property_type}", f"ppo_actions_{property_type}.png")
        plot_with_actions(a2c_eval, f"A2C Actions: {property_type}", f"a2c_actions_{property_type}.png")

        plot_action_distribution(q_eval, f"Q-Learning ({property_type})", f"q_dist_{property_type}.png")
        plot_action_distribution(ppo_eval, f"PPO ({property_type})", f"ppo_dist_{property_type}.png")
        plot_action_distribution(a2c_eval, f"A2C ({property_type})", f"a2c_dist_{property_type}.png")