# ====================== 1. 导入依赖库 ======================
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# 字体配置（英文标签避免乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#e0e0e0'

# ====================== 2. 数据加载与预处理（轻量化） ======================
def load_and_preprocess_data(file_path, start_date='2015-01-01', end_date='2025-06-30'):
    df = pd.read_csv("btc-usd-max.csv")
    # 只保留必要列，减少内存
    df = df[['snapped_at', 'price']].copy()
    df['snapped_at'] = pd.to_datetime(df['snapped_at']) + pd.Timedelta(hours=8)
    df.rename(columns={'snapped_at': 'date'}, inplace=True)
    # 计算收益率（避免浮点精度问题）
    df['log_return'] = np.log(df['price'] / df['price'].shift(1)).astype(np.float32)
    df = df.dropna(subset=['log_return']).reset_index(drop=True)
    # 筛选时间范围
    df['date'] = pd.to_datetime(df['date'])
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df_target = df.loc[mask].copy().reset_index(drop=True)
    # 二阶AR去自相关（轻量化）
    returns = df_target['log_return'].values
    ar_model = ARIMA(returns, order=(2, 0, 0)).fit()
    residuals = ar_model.resid.astype(np.float32)  # 用float32减少内存
    print(f"✅ 数据预处理完成！有效样本量：{len(df_target)} 条")
    return df_target, residuals

# ====================== 3. SV-MCMC模型（内存优化版） ======================
class SVModelMCMC:
    def __init__(self, y, alpha_init=-6, beta_init=0.2, sigma_w_init=3, burn_in=1000, n_iter=5000):
        self.y = y.astype(np.float32)  # 降精度减少内存
        self.T = len(y)
        self.burn_in = burn_in
        self.n_iter = n_iter
        # 参数初始值
        self.alpha = alpha_init
        self.beta = beta_init
        self.sigma_w = sigma_w_init
        # 只存储参数样本（不存储全量波动率样本）
        self.alpha_samples = []
        self.beta_samples = []
        self.sigma_w_samples = []
        # 波动率统计量（实时累加，不存储全量）
        self.vol_sum = np.zeros(self.T, dtype=np.float32)  # 波动率和
        self.vol_sq_sum = np.zeros(self.T, dtype=np.float32)  # 波动率平方和
        self.vol_count = 0  # 有效抽样次数

    def _log_likelihood(self, alpha, beta, sigma_w, h):
        """轻量化似然函数计算"""
        sigma_t = np.exp(h / 2)
        log_lik = -0.5 * np.sum(np.log(sigma_t**2) + (self.y**2) / sigma_t**2)
        return log_lik

    def _sample_h(self, alpha, beta, sigma_w):
        """轻量化波动率抽样"""
        h = np.zeros(self.T, dtype=np.float32)
        h[0] = np.random.normal(alpha / (1 - beta), sigma_w / np.sqrt(1 - beta**2))
        for t in range(1, self.T):
            mean_h = alpha + beta * h[t-1]
            var_h = sigma_w**2
            h[t] = np.random.normal(mean_h, var_h)
        return h

    def _metropolis_step(self):
        """单次Metropolis抽样（轻量化）"""
        # 抽样波动率h
        h = self._sample_h(self.alpha, self.beta, self.sigma_w)
        # 提议参数（缩小方差减少计算量）
        alpha_prop = np.random.normal(self.alpha, 0.05)
        beta_prop = np.random.normal(self.beta, 0.005)
        sigma_w_prop = np.random.normal(self.sigma_w, 0.05)
        # 参数约束
        beta_prop = np.clip(beta_prop, -0.99, 0.99)
        sigma_w_prop = max(sigma_w_prop, 0.01)
        # 计算接受概率（简化计算）
        log_lik_current = self._log_likelihood(self.alpha, self.beta, self.sigma_w, h)
        log_lik_prop = self._log_likelihood(alpha_prop, beta_prop, sigma_w_prop, h)
        # 先验分布（简化）
        prior_current = stats.norm.logpdf(self.alpha, 0, 10) + stats.norm.logpdf(self.beta, 0, 10)
        prior_prop = stats.norm.logpdf(alpha_prop, 0, 10) + stats.norm.logpdf(beta_prop, 0, 10)
        # 接受概率
        log_accept = (log_lik_prop + prior_prop) - (log_lik_current + prior_current)
        accept_prob = min(1, np.exp(log_accept))
        # 接受/拒绝
        if np.random.uniform(0, 1) < accept_prob:
            self.alpha = alpha_prop
            self.beta = beta_prop
            self.sigma_w = sigma_w_prop
        return self.alpha, self.beta, self.sigma_w, h

    def run_mcmc(self):
        print(f"\n🚀 MCMC抽样（总迭代：{self.n_iter}，燃烧期：{self.burn_in}）")
        for i in tqdm(range(self.n_iter)):
            alpha, beta, sigma_w, h = self._metropolis_step()
            # 燃烧期后：只存储参数+累加波动率统计量（不存储全量h）
            if i >= self.burn_in:
                self.alpha_samples.append(alpha)
                self.beta_samples.append(beta)
                self.sigma_w_samples.append(sigma_w)
                # 累加波动率（实时计算均值/方差，不存储全量）
                self.vol_sum += h
                self.vol_sq_sum += h**2
                self.vol_count += 1

        # 计算参数后验统计量
        self.alpha_mean = np.mean(self.alpha_samples)
        self.beta_mean = np.mean(self.beta_samples)
        self.sigma_w_mean = np.mean(self.sigma_w_samples)
        self.alpha_rmse = np.sqrt(np.mean((np.array(self.alpha_samples) - self.alpha_mean)**2))

        # 计算波动率后验均值和95%置信区间（轻量化）
        self.vol_mean = self.vol_sum / self.vol_count  # 均值
        vol_var = (self.vol_sq_sum / self.vol_count) - (self.vol_mean**2)  # 方差
        self.vol_std = np.sqrt(vol_var)  # 标准差
        # 95%置信区间（正态近似，替代分位数，减少内存）
        self.vol_ci = [
            self.vol_mean - 1.96 * self.vol_std,
            self.vol_mean + 1.96 * self.vol_std
        ]

        print("\n✅ 参数后验估计：")
        print(f"α = {self.alpha_mean:.4f} | RMSE = {self.alpha_rmse:.4f}")
        print(f"β = {self.beta_mean:.4f}")
        print(f"σ_w = {self.sigma_w_mean:.4f}")
        return self

# ====================== 4. 绘图（轻量化） ======================
def plot_price(df, save_path="btc_price_2015_2025.png"):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['date'], df['price'], color='#2c7fb8', linewidth=1.2)
    ax.set_title('Bitcoin Price (2015-2025)', fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (USD)', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Price plot saved: {save_path}")

def plot_return(df, save_path="btc_return_2015_2025.png"):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['date'], df['log_return'], color='#ff7f0e', linewidth=0.8)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_title('Bitcoin Daily Log Returns (2015-2025)', fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Log Return', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Return plot saved: {save_path}")

def plot_volatility(df, sv_model, save_path="btc_volatility_sv_mcmc.png"):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['date'], sv_model.vol_mean, color='#2ca02c', linewidth=1.2, label='Posterior Mean of Volatility')
    ax.fill_between(df['date'], sv_model.vol_ci[0], sv_model.vol_ci[1], 
                    color='#2ca02c', alpha=0.2, label='95% Confidence Interval')
    ax.set_title('Bitcoin Volatility Estimation (SV-MCMC Model)', fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('ln(σ_t²)', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Volatility plot saved: {save_path}")

# ====================== 5. 主程序（低内存配置） ======================
if __name__ == "__main__":
    # 低内存配置：减少迭代次数，避免内存溢出
    DATA_FILE = "你的比特币数据.csv"  # 替换为你的数据路径
    BURN_IN = 5000    # 减少燃烧期
    N_ITER = 20000    # 减少总迭代次数（平衡内存和精度）

    # 数据预处理
    df_processed, residuals = load_and_preprocess_data(DATA_FILE)
    # 运行SV-MCMC（内存优化版）
    sv_model = SVModelMCMC(y=residuals, burn_in=BURN_IN, n_iter=N_ITER).run_mcmc()

    # 绘图
    plot_price(df_processed)
    plot_return(df_processed)
    plot_volatility(df_processed, sv_model)

    # 保存参数结果（无乱码）
    param_results = pd.DataFrame({
        'Parameter': ['α', 'β', 'σ_w'],
        'MCMC Estimate': [sv_model.alpha_mean, sv_model.beta_mean, sv_model.sigma_w_mean],
        'RMSE': [sv_model.alpha_rmse, '-', '-']
    })
    param_results.to_csv('sv_model_param_results.csv', index=False, encoding='gbk')  # Windows编码
    print("\n📋 Parameter results saved: sv_model_param_results.csv")