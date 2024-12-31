import pandas as pd
import numpy as np
from scipy.stats import norm
import streamlit as st
import plotly.graph_objs as go

class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma):
        self.S = S  # Current stock price
        self.K = K  # Strike price
        self.T = T  # Time to maturity in years
        self.r = r  # Risk-free interest rate
        self.sigma = sigma  # Volatility

    def calculate_d1_d2(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return d1, d2

    def price(self, option_type):
        d1, d2 = self.calculate_d1_d2()
        if option_type == 'Call':
            return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        elif option_type == 'Put':
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        else:
            raise ValueError('Option type not recognized')

    def greeks(self, option_type):
        d1, d2 = self.calculate_d1_d2()
        delta = norm.cdf(d1) if option_type == 'Call' else norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        theta = (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T))
                 - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)) if option_type == 'Call' else (
                -self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T))
                + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2))
        vega = self.S * norm.pdf(d1) * np.sqrt(self.T)
        rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2) if option_type == 'Call' else (
              -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2))
        return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}
    

def save_configuration(config, filename="config.csv"):
    """Save the current configuration to a CSV file."""
    df = pd.DataFrame([config])
    df.to_csv(filename, index=False)

def load_configuration(filename="config.csv"):
    """Load configuration from a CSV file."""
    try:
        return pd.read_csv(filename).iloc[0].to_dict()
    except FileNotFoundError:
        st.error("Configuration file not found.")
        return None

def export_results_to_csv(results, filename="results.csv"):
    """Export option pricing results to a CSV file."""
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    st.success(f"Results exported to {filename}")

def plot_option_prices(S_range, S, K, T, r, sigma, option_type):
    values = np.linspace(*S_range, 100)
    prices = [BlackScholesModel(val, K, T, r, sigma).price(option_type) for val in values]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=values, y=prices, mode='lines', name=f'{option_type.capitalize()} Option Price'))
    fig.update_layout(title=f'{option_type.capitalize()} Option Prices vs. Stock Price',
                      xaxis_title='Stock Price (S)',
                      yaxis_title='Option Price')
    return fig

def plot_greeks(S_range, S, K, T, r, sigma, option_type):
    greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
    values = np.linspace(*S_range, 100)
    fig = go.Figure()

    for greek in greeks:
        greek_values = [BlackScholesModel(val, K, T, r, sigma).greeks(option_type)[greek] for val in values]
        fig.add_trace(go.Scatter(x=values, y=greek_values, mode='lines', name=f'{greek.capitalize()}'))

    fig.update_layout(title=f'{option_type.capitalize()} Option Greeks vs. Stock Price',
                      xaxis_title='Stock Price (S)',
                      yaxis_title='Value')
    return fig

def plot_heatmap(S_range, K_range, T, r, sigma, option_type):
    S_values = np.linspace(*S_range, 50)
    K_values = np.linspace(*K_range, 50)
    S_grid, K_grid = np.meshgrid(S_values, K_values)
    prices = np.array([[BlackScholesModel(S, K, T, r, sigma).price(option_type) for S in S_values] for K in K_values])

    fig = go.Figure(data=go.Heatmap(
        z=prices,
        x=S_values,
        y=K_values,
        colorscale='Viridis',
        colorbar=dict(title="Option Price")
    ))
    fig.update_layout(
        title=f"Option Price Heatmap ({option_type})",
        xaxis_title="Stock Price (S)",
        yaxis_title="Strike Price (K)"
    )
    return fig

def plot_3d_surface(S_range, K_range, T, r, sigma, option_type):
    S_values = np.linspace(*S_range, 50)
    K_values = np.linspace(*K_range, 50)
    S_grid, K_grid = np.meshgrid(S_values, K_values)
    prices = np.array([[BlackScholesModel(S, K, T, r, sigma).price(option_type) for S in S_values] for K in K_values])

    fig = go.Figure(data=[go.Surface(z=prices, x=S_grid, y=K_grid, colorscale="Viridis")])
    fig.update_layout(
        title=f"Option Price 3D Surface ({option_type})",
        scene=dict(
            xaxis_title="Stock Price (S)",
            yaxis_title="Strike Price (K)",
            zaxis_title="Option Price"
        )
    )
    return fig

def main():
    st.set_page_config(page_title="Black-Scholes Option Pricing Model", layout="wide")

    st.markdown("<h1 style='text-align: center;'>Massimo Capuano BS Option Pricing Model</h1>", unsafe_allow_html=True)

    st.markdown("<p style='text-align: center;'>This app calculates the price and greeks of a European call or put option using the Black-Scholes model, with heatmaps and 3D visualizations for enhanced analysis.</p>", unsafe_allow_html=True)

    st.sidebar.subheader("Option Parameters")
    option_type = st.sidebar.selectbox("Option Type", ["Call", "Put"])
    S = st.sidebar.slider("Current Stock Price (S)", 50.0, 150.0, 100.0, 1.0)
    K = st.sidebar.slider("Strike Price (K)", 50.0, 150.0, 100.0, 1.0)
    T = st.sidebar.slider("Time to Maturity (T)", 0.1, 2.0, 1.0, 0.1)
    r = st.sidebar.slider("Risk-Free Interest Rate (r)", 0.0, 0.1, 0.05, 0.01)
    sigma = st.sidebar.slider("Volatility (σ)", 0.1, 0.5, 0.2, 0.01)

    config = {"option_type": option_type, "S": S, "K": K, "T": T, "r": r, "sigma": sigma}

    if st.sidebar.button("Save Configuration"):
        save_configuration(config)

    if st.sidebar.button("Load Configuration"):
        loaded_config = load_configuration()
        if loaded_config:
            option_type = loaded_config["option_type"]
            S = loaded_config["S"]
            K = loaded_config["K"]
            T = loaded_config["T"]
            r = loaded_config["r"]
            sigma = loaded_config["sigma"]

    model = BlackScholesModel(S, K, T, r, sigma)
    price = model.price(option_type)
    greeks = model.greeks(option_type)

    st.markdown(f"<h2 style='text-align: left; color: Green;'>{option_type.capitalize()} Option Price: {price:.2f}</h2>", unsafe_allow_html=True)
    st.subheader("**Greeks**")

    st.markdown(f"**Delta:** {greeks['delta']:.2f}")
    st.markdown(f"**Gamma:** {greeks['gamma']:.2f}")
    st.markdown(f"**Theta:** {greeks['theta']:.2f}")
    st.markdown(f"**Vega:** {greeks['vega']:.2f}")
    st.markdown(f"**Rho:** {greeks['rho']:.2f}")

    S_range = (50, 150)
    K_range = (50, 150)

    st.plotly_chart(plot_option_prices(S_range, S, K, T, r, sigma, option_type))
    st.plotly_chart(plot_greeks(S_range, S, K, T, r, sigma, option_type))
    st.plotly_chart(plot_heatmap(S_range, K_range, T, r, sigma, option_type))
    st.plotly_chart(plot_3d_surface(S_range, K_range, T, r, sigma, option_type))

    st.markdown("""
    <div style='text-align: center; margin-top: 20px;'>
        <a href='https://www.linkedin.com/in/mascapuano' style='text-decoration: none; padding: 8px 15px; background-color: #0077B5; color: white; border-radius: 5px;'>
            Connect with me on LinkedIn
        </a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()