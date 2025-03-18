import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    return mo, np, pd, plt


@app.cell
def _(mo):
    mo.md("""# TESS expriment with BOPTEST result for typical weekday and linear HVAC bidding function""")
    return


@app.cell
def _(mo):
    mo.md(
        """
        # HVAC Pricing Function

        ### General Adjustments
        - If the expected standard deviation (`expected_stdev`) is zero, it is set to a minimal non-zero value to prevent computational issues:
          ```python
          if expected_stdev == 0:
              expected_stdev = expected_price * 0.001
          ```

        ### Heating Mode Logic
        Triggered if:
        - `mode` is `'heating'`, or
        - `mode` is `'auto'` and the last temperature change was negative (implying recent heating).

        Bid logic:
        - If `current_temp < min_temp`: HVAC is critically cold; bids at **maximum price** to urgently obtain energy.
        - If `current_temp > max_temp`: HVAC is too warm; bids at **minimum price**.
        - For intermediate temperatures, the bid price is adjusted based on the temperature difference from `desired_temp`:
          ```python
          p_order = expected_price + K_hvac * expected_stdev * ((desired_temp - current_temp) / abs(min_temp - desired_temp))
          ```
          The result is constrained between `min_price` and `max_price`.

        ### Cooling Mode Logic
        Triggered if:
        - `mode` is `'cooling'`, or
        - `mode` is `'auto'` and the last temperature change was positive (implying recent cooling).

        Bid logic:
        - If `current_temp < min_temp`: HVAC is too cool; bids at **minimum price**.
        - If `current_temp > max_temp`: HVAC is critically hot; bids at **maximum price** to urgently obtain cooling energy.
        - For intermediate temperatures, the bid price is adjusted based on the temperature difference from `desired_temp`:
          ```python
          p_order = expected_price + K_hvac * expected_stdev * ((current_temp - desired_temp) / abs(max_temp - desired_temp))
          ```
          Again, constrained between `min_price` and `max_price`.

        ## Example Usage
        ```python
        # Example parameters
        mode = 'auto'
        last_change_temp = -0.5
        current_temp = 19.0
        desired_temp = 21.0
        min_temp = 18.0
        max_temp = 24.0
        expected_price = 50.0
        expected_stdev = 5.0
        K_hvac = 0.8
        min_price = 10.0
        max_price = 100.0

        price_bid = get_hvac_bid_price(mode, last_change_temp, current_temp, desired_temp,
                                       min_temp, max_temp, expected_price, expected_stdev,
                                       K_hvac, min_price, max_price)
        print(f"HVAC bid price: {price_bid}")
        ```

        ## Purpose and Utility
        This function allows HVAC systems in a transactive energy framework to dynamically and intelligently set energy purchase prices. It optimizes energy consumption economically, responding appropriately to real-time conditions and market volatility.

        ## KPI vs K values
        """
    )
    return


@app.cell
def _(pd):
    kpi_values = pd.read_csv("data/kpi_values.csv")
    return (kpi_values,)


@app.cell
def _(kpi_values, plt):
    plt.figure(figsize=(15,5))
    plt.plot(kpi_values["K_hvac"], kpi_values["Energy Consumption (kWh)"])
    plt.xlabel("K Value")
    plt.ylabel("Energy Consumption (kWh)")
    return


@app.cell
def _(kpi_values, plt):
    plt.figure(figsize=(15,5))
    plt.plot(kpi_values["K_hvac"], kpi_values["Peak Power (kW)"])
    plt.xlabel("K Value")
    plt.ylabel("Peak Power (kW)")
    return


@app.cell
def _(kpi_values, plt):
    plt.figure(figsize=(15,5))
    plt.plot(kpi_values["K_hvac"], kpi_values["Temperature Discomfort"])
    plt.xlabel("K Value")
    plt.ylabel("Temperature Discomfort")
    return


@app.cell
def _(kpi_values, plt):
    plt.figure(figsize=(15,5))
    plt.plot(kpi_values["K_hvac"], kpi_values["tdis_tot"])
    plt.xlabel("K Value")
    plt.ylabel("BOPTEST Temp Discomfort")
    return


@app.cell
def _(kpi_values, plt):
    plt.figure(figsize=(15,5))
    plt.plot(kpi_values["K_hvac"], kpi_values["Average Heating Cycles per hour"])
    plt.xlabel("K Value")
    plt.ylabel("Average Heating Cycles per hour")
    return


@app.cell
def _(kpi_values, plt):
    plt.figure(figsize=(15,5))
    plt.plot(kpi_values["K_hvac"], kpi_values["ener_tot"])
    plt.xlabel("K Value")
    plt.ylabel("BOPTEST Energy Total")
    return


@app.cell
def _(kpi_values, plt):
    plt.figure(figsize=(15,5))
    plt.plot(kpi_values["K_hvac"], kpi_values["emis_tot"])
    # plt.xlabel("K Value")
    plt.ylabel("BOPTEST Emisssion")
    return


@app.cell
def _():
    # kpi_values.columns
    return


if __name__ == "__main__":
    app.run()
