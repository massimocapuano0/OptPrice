# OptPrice
This project is an interactive Streamlit-based application that employs the Black-Scholes model to analyze European call and put options. The app computes option prices and Greeks (Delta, Gamma, Theta, Vega, and Rho) based on user-defined parameters such as stock price, strike price, time to maturity, risk-free interest rate, and volatility. It provides advanced visualization tools, including line plots, heatmaps, and 3D surfaces, to offer a comprehensive understanding of the option pricing landscape. Additionally, users can save and load configurations and export results for further analysis. This project is ideal for financial analysts, traders, and educators aiming to explore option pricing and sensitivities interactively.

The Black-Scholes model (Black-Scholes-Merton model) is a cornerstone of modern financial theory. It offers a mathematical formula to determine the theoretical price of options by factoring in time and various risk components. The Black-Scholes formula depends on five primary inputs: **Volatility**, **Current price**, **Strike price**, **Time to expiration**, **Risk-free interest rate**.

The project can be found here: [Link to Project](https://bsmassimocapuano.streamlit.app/)

---

## VenV Workspace Setup
To ensure all dependencies are properly installed and the environment is correctly set up, it is recommended to use a **virtual environment**. Follow the steps below to set up the project environment:

1. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   ```
2. **Activate the virtual environment:**
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
3. **Install the required packages using `requirements.txt`:**
   First, ensure that the `requirements.txt` file is in the project directory.
   Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---
