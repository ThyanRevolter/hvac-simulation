import marimo

__generated_with = "0.10.17"
app = marimo.App(width="full")


@app.cell
def _():
    import pandas as pd
    import marimo as mo
    import requests
    import json
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta, date

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
        pd,
        plt,
        requests,
        timedelta,
    )


@app.cell
def _(mo):
    mo.md("""# HVAC Simulation using BOPTEST""")
    return


@app.cell
def _(BOPTEST_URL, date, mo, requests):
    all_testcases = requests.get(f"{BOPTEST_URL}/testcases")
    test_cases_list = [i["testcaseid"] for i in all_testcases.json()]
    test_case_dropdown = mo.ui.dropdown(
        test_cases_list,
        value="bestest_air",
    )
    start_date_ui = mo.ui.date(
        start=date(2023, 1, 7), stop=date(2023, 12, 24), value=date(2023, 1, 7),
    )
    warmup_days_ui = mo.ui.number(value=5)
    number_of_days_ui = mo.ui.number(value=7)
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
def _(mo):
    dr_event_check = mo.ui.checkbox(
        label="Add DR Event",
        value=False
    )
    dr_event_check
    return (dr_event_check,)


@app.cell
def _(mo):
    turn_off_heating = mo.ui.checkbox(
        label="Turn off Heating",
        value=False
    )
    turn_off_heating
    return (turn_off_heating,)


@app.cell
def _(datetime, mo):
    dr_event = [
        {
            "start_time": datetime(2023, 1, 8, 16, 0, 0),
            "duration": 2,
        },
        {
            "start_time": datetime(2023, 1, 9, 16, 0, 0),
            "duration": 2,
        }
    ]
    mo.md(f"""
    ## DR Event Details
    - **DR Events:** {[event["start_time"].strftime("%Y-%m-%d %I:%M:%S %p")+" for "+str(event["duration"]) + " hours" for event in dr_event]}
    """)
    return (dr_event,)


@app.cell
def _(
    bs,
    control_step,
    dr_event,
    dr_event_check,
    simul_days,
    simul_start_date,
    turn_off_heating,
):
    control_inputs = bs.create_dr_control_inputs(
        simul_start_date=simul_start_date,
        simul_days=simul_days,
        control_step=control_step,
        dr_events=dr_event  
    )
    if turn_off_heating.value:
        control_inputs["con_oveTSetHea_activate"] = 1
        control_inputs["con_oveTSetHea_u"] = 278.15
    else:
        if not dr_event_check.value:
            control_inputs = None
    return (control_inputs,)


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
def _(bt_instance, mo, simul_days, simul_start_date, warmup_days):
    with mo.status.spinner(subtitle=f"Initializing Simulation at {simul_start_date.strftime('%Y-%m-%d')} with {warmup_days} warmup days") as _spinner:
        initialized_output = bt_instance.initialize(simul_start_date, warmup_period=warmup_days*24*3600)
        forecast_data = bt_instance.get_forecast(simul_days*24)
    return forecast_data, initialized_output


@app.cell
def _(
    bt_instance,
    control_inputs,
    control_step,
    initialized_output,
    mo,
    simul_days,
    simul_start_date,
    timedelta,
):
    initialized_output
    with mo.status.progress_bar(
        title=f"Running Simulation from {simul_start_date.strftime('%Y-%m-%d')} to {(simul_start_date + timedelta(days=simul_days)).strftime('%Y-%m-%d')}",
        completion_title="Simulation Complete",
        show_eta=True,
        total=int(simul_days*24*3600/control_step)
    ) as bar:
        bt_instance.run_simulation(simul_days*24, control_inputs=control_inputs, prog_bar=bar)
    return (bar,)


@app.cell
def _(forecast_points, mo):
    forecast_plot_select = mo.ui.dropdown(
        dict(zip(forecast_points["Description"], forecast_points.index)),
        value=forecast_points["Description"][0],
        label="Forecast Data Plot"
    )
    forecast_plot_select
    return (forecast_plot_select,)


@app.cell
def _(forecast_data, forecast_plot_select, forecast_points, plt):
    plt.figure(figsize=(20,5))
    plt.plot(forecast_data["datetime"], forecast_data[forecast_plot_select.value])
    plt.ylabel(forecast_points.loc[forecast_plot_select.value, "Description"])
    return


@app.cell
def _(bt_instance):
    bt_instance.simulation_results
    return


@app.cell
def _(HVAC_KPI, bt_instance):
    kpi = HVAC_KPI(bt_instance.simulation_results)
    return (kpi,)


@app.cell
def _():
    # kpi.calculate_energy_consumption(), kpi.calculate_peak_power(), kpi.calculate_temperature_discomfort(), kpi.calculate_cycles(kpi.heating_power_variable), kpi.calculate_cycles(kpi.cooling_power_variable)
    return


@app.cell
def _():
    # kpi.calculate_cycles(kpi.heating_power_variable)
    return


@app.cell
def _(inputs_available, measurements_available, mo):
    temperature_variables_list = list(inputs_available[inputs_available["Unit"] == "K"].index) + list(measurements_available[measurements_available["Unit"] == "K"].index)
    temperature_variables_desc = list(inputs_available[inputs_available["Unit"] == "K"]["Description"]) + list(measurements_available[measurements_available["Unit"] == "K"]["Description"])

    power_variables_list = list(inputs_available[inputs_available["Unit"] == "W"].index) + list(measurements_available[measurements_available["Unit"] == "W"].index)
    power_variables_desc = list(inputs_available[inputs_available["Unit"] == "W"]["Description"]) + list(measurements_available[measurements_available["Unit"] == "W"]["Description"])

    control_variables_list = list(inputs_available[(inputs_available["Unit"] == 1) | (inputs_available["Unit"].isnull())].index)
    control_variables_desc = list(inputs_available[(inputs_available["Unit"] == 1) | (inputs_available["Unit"].isnull())]["Description"])

    temperature_variables_y1_select = mo.ui.multiselect(
        dict(zip(temperature_variables_desc, temperature_variables_list)),
        full_width=True
    )
    temperature_variables_y2_select = mo.ui.multiselect(
        dict(zip(temperature_variables_desc, temperature_variables_list)),
        full_width=True
    )
    power_variables_select = mo.ui.multiselect(
        dict(zip(power_variables_desc, power_variables_list)),
        full_width=True
    )
    control_variables_select = mo.ui.multiselect(
        dict(zip(control_variables_desc, control_variables_list)),
        full_width=True
    )
    temperature_variables_y1_select, temperature_variables_y2_select, power_variables_select, control_variables_select
    mo.md(f"""
    ## Plotting Simulation Results

    - **Temperature y1-axis:** {temperature_variables_y1_select}
    - **Temperature y2-axis:** {temperature_variables_y2_select}
    - **Power variables:** {power_variables_select}
    - **Control variables:** {control_variables_select}
    """)
    return (
        control_variables_desc,
        control_variables_list,
        control_variables_select,
        power_variables_desc,
        power_variables_list,
        power_variables_select,
        temperature_variables_desc,
        temperature_variables_list,
        temperature_variables_y1_select,
        temperature_variables_y2_select,
    )


@app.cell
def _(
    bt_instance,
    control_variables_select,
    power_variables_select,
    temperature_variables_y1_select,
    temperature_variables_y2_select,
):
    bt_instance.plot_simulation_results(
        temperature_variables_y1_select.value,
        temperature_variables_y2_select.value,
        power_variables_select.value,
        control_variables_select.value
    )
    return


@app.cell
def _():
    import numpy as np
    return (np,)


@app.cell
def _(bt_instance):
    bt_instance.get_kpis().to_dict()["Value"]
    return


@app.cell
def _(bt_instance, mo):
    mo.md(f"""
    ## Key Performance Indicators
    {bt_instance.get_kpis().to_markdown()}
    """
    )
    return


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
        ## Simulation Stopped
        """)
    output
    return (output,)


if __name__ == "__main__":
    app.run()
