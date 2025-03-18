import marimo

__generated_with = "0.9.20"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import marimo as mo
    import requests
    import json
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta, date
    import numpy as np

    from hvac_simulation.boptest.boptest_suite import BOPTESTClient as bt
    from hvac_simulation.boptest import boptest_suite as bs
    from hvac_simulation.bidding_strategy.heuristic_order import HVACOrder
    from hvac_simulation.kpi import HVAC_KPI
    from hvac_simulation.tess_control import TESSControl

    BOPTEST_URL = 'http://127.0.0.1'
    return (
        BOPTEST_URL,
        HVACOrder,
        HVAC_KPI,
        TESSControl,
        bs,
        bt,
        date,
        datetime,
        json,
        mo,
        np,
        pd,
        plt,
        requests,
        timedelta,
    )


@app.cell
def _(mo):
    mo.md(r"""# HVAC TESS Simulation using BOPTEST""")
    return


@app.cell
def _(BOPTEST_URL, date, mo, requests):
    all_testcases = requests.get(f"{BOPTEST_URL}/testcases")
    test_cases_list = [i["testcaseid"] for i in all_testcases.json()]
    test_case_dropdown = mo.ui.dropdown(
        test_cases_list,
        value="bestest_hydronic_heat_pump",
    )
    start_date_ui = mo.ui.date(
        start=date(2023, 1, 7), stop=date(2023, 12, 24), value=date(2023, 1, 7),
    )
    warmup_days_ui = mo.ui.number(value=1)
    number_of_days_ui = mo.ui.number(value=1)
    temperature_unit_ui = mo.ui.dropdown(
        ["C", "F"],
        value="F",
    )
    control_step_ui = mo.ui.number(value=900, step=300)
    return (
        all_testcases,
        control_step_ui,
        number_of_days_ui,
        start_date_ui,
        temperature_unit_ui,
        test_case_dropdown,
        test_cases_list,
        warmup_days_ui,
    )


@app.cell
def _(
    control_step_ui,
    mo,
    number_of_days_ui,
    start_date_ui,
    temperature_unit_ui,
    test_case_dropdown,
    warmup_days_ui,
):
    test_param_form = (
    mo.md(
    """
    ## Simulation Parameters
    - **Test Case:** {test_case_dropdown}
    - **Temperature Unit:** {temperature_unit}
    - **Control Step in seconds:** {control_step}
    - **Warmup Days:** {warmup_days}
    - **Start Date:** {start_date}
    - **Number of Days:** {number_of_days}
    """).batch(
        test_case_dropdown=test_case_dropdown,
        start_date=start_date_ui,
        number_of_days=number_of_days_ui,
        temperature_unit=temperature_unit_ui,
        control_step=control_step_ui,
        warmup_days=warmup_days_ui)
    ).form(show_clear_button=True, bordered=True)
    test_param_form
    return (test_param_form,)


@app.cell
def _(datetime, mo, test_param_form):
    mo.stop(test_param_form.value is None, mo.md("**Submit the form to continue.**"))
    test_case = test_param_form.value["test_case_dropdown"]
    temperature_unit = test_param_form.value["temperature_unit"]
    control_step = test_param_form.value["control_step"]
    simul_start_date = datetime(test_param_form.value["start_date"].year, test_param_form.value["start_date"].month, test_param_form.value["start_date"].day, 0, 0, 0)
    simul_days = test_param_form.value["number_of_days"]
    warmup_days = test_param_form.value["warmup_days"]
    return (
        control_step,
        simul_days,
        simul_start_date,
        temperature_unit,
        test_case,
        warmup_days,
    )


@app.cell
def _(BOPTEST_URL, bt, control_step, temperature_unit, test_case):
    bt_instance = bt(
        base_url=BOPTEST_URL,
        testcase=test_case,
        control_step=control_step,
        temp_unit=temperature_unit,
        timeout=300
    )
    return (bt_instance,)


@app.cell
def _(bt_instance, mo):
    inputs_available = bt_instance.inputs
    measurements_available = bt_instance.measurements
    forecast_points = bt_instance.forecast_points

    inputs_md = mo.md(f"""
    {inputs_available.to_markdown()}
    """)
    measurements_md = mo.md(f"""
    {measurements_available.to_markdown()}
    """)
    forecast_points_md = mo.md(f"""
    {forecast_points.to_markdown()}
    """)
    mo.ui.tabs(
        {
            "Available Inputs": inputs_md,
            "Available Measurements": measurements_md,
            "Forecast Points": forecast_points_md
        }

    )
    return (
        forecast_points,
        forecast_points_md,
        inputs_available,
        inputs_md,
        measurements_available,
        measurements_md,
    )


@app.cell
def _(
    BOPTEST_URL,
    TESSControl,
    bt_instance,
    control_step,
    simul_days,
    simul_start_date,
    temperature_unit,
    test_case,
    warmup_days,
):
    customer_parameters = {
            "K_hvac": 0.1, #K_hvac.value,
            "desired_temp": 70 #desired_soc.value
    }
    market_parameters  = {
            "min_price": 0.001*1000,
            "max_price": 2*1000,
            "interval": 300
    }
    boptest_param = {
        "base_url": BOPTEST_URL,
        "testcase": test_case,
        "control_step": control_step,
        "temp_unit": temperature_unit,
        "timeout": 300
    }
    simul_hours = simul_days*24
    tess_simul = TESSControl(
        bt_instance=bt_instance,
        start_time=simul_start_date,
        duration_hours=simul_hours,
        warmup_period=warmup_days*24,
        market_interval=control_step,
        avaialble_control_inputs=["oveHeaPumY_activate", "oveHeaPumY_u", "oveTSet_u", "oveTSet_activate"],
        control_default_values=[0, None, 294.261, 1],
        control_dr_values=[1, 0, 0, None]
    )
    tess_simul.set_hvac_customer_parameters(customer_parameters)
    tess_simul.set_hvac_market_parameters(market_parameters)
    return (
        boptest_param,
        customer_parameters,
        market_parameters,
        simul_hours,
        tess_simul,
    )


@app.cell
def _(control_step, np, simul_hours, tess_simul):
    timesteps = int((simul_hours * 3600) / control_step)
    one_array = np.ones(timesteps)
    outdoor_temp = tess_simul.forecasted_data["TDryBul"]
    temp_deviations = (np.mean(outdoor_temp) - outdoor_temp)

    market_cleared_price =one_array*50 + temp_deviations

    market_expected_mean_price = market_cleared_price
    market_expected_std_price = temp_deviations/np.linalg.norm(temp_deviations)*1
    market_expected_std_price = market_expected_mean_price*0.01
    return (
        market_cleared_price,
        market_expected_mean_price,
        market_expected_std_price,
        one_array,
        outdoor_temp,
        temp_deviations,
        timesteps,
    )


@app.cell
def _(
    market_cleared_price,
    market_expected_mean_price,
    market_expected_std_price,
    plt,
    timesteps,
):
    # Plotting the values
    plt.figure(figsize=(18, 15/3))
    plt.plot(market_expected_mean_price, label="Market Expected Mean Price", color='blue')
    plt.fill_between(range(timesteps), market_expected_mean_price - market_expected_std_price, market_expected_mean_price + market_expected_std_price, color='blue', alpha=0.2, label="Price Range (Std Dev)")
    plt.plot(market_cleared_price, label="Market Cleared Price", color='red')

    # Labels and Title
    plt.title("Market Prices with Expected Mean and Cleared Price")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(
    market_cleared_price,
    market_expected_mean_price,
    market_expected_std_price,
    tess_simul,
):
    simul_result = tess_simul.run_tess_simulation(
        market_expected_mean_price,
        market_expected_std_price,
        market_cleared_price
    )
    return (simul_result,)


@app.cell
def _(
    customer_parameters,
    market_cleared_price,
    np,
    plt,
    simul_result,
    tess_simul,
):
    simul_result
    tess_simul.control_list["on"] = tess_simul.control_list["mode"] != "off"
    subset = 100
    plt.rcParams.update(
        {
            "font.size": 22,
            "xtick.major.size": 10,
            "xtick.major.width": 3,
            "ytick.major.size": 10,
            "ytick.major.width": 3,
            "axes.linewidth": 2.5,
            "axes.edgecolor": "black",
        }
    )
    legend_loc="lower right"
    fig, ax = plt.subplots(3, 1, figsize=(18, 15), sharex=True)
    ax[0].plot(
        tess_simul.simulation_results["datetime"][:subset],
        ((tess_simul.simulation_results["reaTZon_y"] - 273.15)*9/5 + 32)[:subset],
        label="Air Temp"
    )
    for i in range(1, len(tess_simul.control_list["datetime"][:subset])):
        if tess_simul.control_list['mode'].iloc[i] == "heating":
            ax[0].axvspan(tess_simul.control_list['datetime'].iloc[i-1], tess_simul.control_list['datetime'].iloc[i], color='orange', alpha=0.3, ec=None)
        if tess_simul.control_list['mode'].iloc[i] == "cooling":
            ax[0].axvspan(tess_simul.control_list['datetime'].iloc[i-1], tess_simul.control_list['datetime'].iloc[i], color='lightblue', alpha=0.3, ec=None)
    ax[0].axhline(y=customer_parameters["desired_temp"], color='r', linestyle='--', label="Desired Temp")
    ax[0].legend(loc=legend_loc)
    ax[0].set_ylabel("Zone Temperature (F)")
    ax[0].set_xlabel("Time")

    # plot the power variables
    ax[1].plot(
        tess_simul.simulation_results["datetime"][:subset],
        tess_simul.simulation_results["reaPHeaPum_y"][:subset],
        label="Heat Pump Power"
    )
    ax[1].plot(
        tess_simul.simulation_results["datetime"][:subset],
        tess_simul.simulation_results["reaPFan_y"][:subset],
        label="Fan Power"
    )
    ax[1].legend(loc=legend_loc)
    ax[1].set_ylabel("Power (W)")
    ax[1].set_xlabel("Time")

    # plot market clearing price
    ax[2].plot(
        tess_simul.simulation_results["datetime"][:subset],
        market_cleared_price[:subset],
        label="Cleared Price"
    )
    ax[2].plot(
        tess_simul.tess_results["datetime"][:subset],
        tess_simul.tess_results["price"][:subset],    
        label="Bid Price"
    )
    ax[2].fill_between(
        tess_simul.tess_results["datetime"][:subset],
        market_cleared_price[:subset],
        np.maximum(market_cleared_price[:subset], tess_simul.tess_results["price"][:subset]),
        color="orange",
        alpha=0.5
    )
    ax[2].set_ylabel("Cleared Price ($/MWh)")
    ax[2].set_xlabel("Time")
    ax[2].legend(loc=legend_loc)
    return ax, fig, i, legend_loc, subset


@app.cell
def _(
    HVAC_KPI,
    control_step,
    customer_parameters,
    simul_result,
    tess_simul,
):
    simul_result
    kpi = HVAC_KPI(tess_simul.simulation_results,
                   control_step=control_step,
                   heating_power_variable="reaPHeaPum_y",
                   temperature_variable="reaTZon_y",
                   setpoint_temperature=(customer_parameters["desired_temp"] - 32) * 5/9 + 273.15,
                   temperature_unit="K")
    return (kpi,)


@app.cell
def _(kpi, mo, pd):
    kpi_values = {
        'Energy Consumption (kWh)': kpi.calculate_energy_consumption()/1000,
        'Peak Power (kW)': kpi.calculate_peak_power()/1000,
        'Temperature Discomfort': kpi.calculate_temperature_discomfort(),
        'Average Heating Cycles per hour': kpi.calculate_cycles(kpi.heating_power_variable)['average_cycles_per_hour']
    }
    kpi_df = pd.DataFrame.from_dict(kpi_values, orient='index', columns=['Value'])
    kpi_df.index.name = 'KPI'

    mo.md(
    f"""
    ### HVAC KPI values
    - **Energy Consumption (kWh):** {kpi_values['Energy Consumption (kWh)']:.2f}
    - **Peak Power (kW):** {kpi_values['Peak Power (kW)']:.2f}
    - **Temperature Discomfort:** {kpi_values['Temperature Discomfort']:.3f}
    - **Average Heating Cycles per hour:** {kpi_values['Average Heating Cycles per hour']:.2f}
    """
    )
    return kpi_df, kpi_values


@app.cell
def _(mo):
    stop_simulation = mo.ui.run_button(
        kind="danger",
        label="End Simulation",    
    )
    stop_simulation
    return (stop_simulation,)


@app.cell
def _(bt_instance, mo, stop_simulation):
    output = mo.md("")
    if stop_simulation.value:
        bt_instance.stop()
        output = mo.md(f"""
        ## Simulation Stopped, You can Restart
        """)
    output
    return (output,)


@app.cell
def _():
    # rerun_simulation = mo.ui.run_button(
    #     kind="neutral",
    #     label="Re Run Simulation",
    #     disabled=not(stop_simulation.value)
    # )
    # rerun_simulation
    return


if __name__ == "__main__":
    app.run()
