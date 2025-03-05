import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date
import numpy as np


from hvac_simulation.boptest_suite import BOPTESTClient as bt
from hvac_simulation.kpi import HVAC_KPI
from hvac_simulation.tess_control import TESSControl

BOPTEST_URL = 'http://127.0.0.1'

test_case_parameters = {
  "test_case": "bestest_hydronic_heat_pump",
  "start_date": "2023-01-07",
  "number_of_days": 5,
  "temperature_unit": "F",
  "control_step": 900,
  "warmup_days": 1
}

test_case = test_case_parameters['test_case']
control_step = test_case_parameters['control_step']
temperature_unit = test_case_parameters['temperature_unit']
simul_start_date = datetime.strptime(test_case_parameters['start_date'], '%Y-%m-%d')
simul_days = test_case_parameters['number_of_days']
simul_hours = simul_days*24
warmup_days = test_case_parameters['warmup_days']

k_value_range = np.arange(0.01, 15, 0.01)
kpi_values_list = []

for k_value in k_value_range:
    bt_instance = bt(
        base_url=BOPTEST_URL,
        testcase=test_case_parameters['test_case'],
        control_step=test_case_parameters['control_step'],
        temp_unit=test_case_parameters['temperature_unit'],
        timeout=300
    )

    customer_parameters = {
            "K_hvac": k_value,
            "desired_temp": 70
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

    # the market mean price is dependant on the temperature and a random noise
    market_expected_mean_price = (
        np.ones(int((simul_hours * 3600) / control_step))*50 # the mean price is 50
        + (
            np.mean(tess_simul.forecasted_data["TDryBul"])
            - tess_simul.forecasted_data["TDryBul"] # we are subtracting the temperature from the mean temperature to get the deviation
        )[0:-1] # the last value is not used because it is the last control step
        + np.random.normal(0, 0.5, int((simul_hours * 3600) / control_step))
    )

    # the market std price is dependant on the temperature
    market_expected_std_price = (
        np.ones(int((simul_hours * 3600) / control_step))*10 # the std price is 10
        + (
            (np.mean(tess_simul.forecasted_data["TDryBul"])
            - tess_simul.forecasted_data["TDryBul"])[0:-1] # we are subtracting the temperature from the mean temperature to get the deviation
        )/5 # we are dividing by 5 to get the std price
    )

    # the market cleared price is dependant on the temperature
    market_cleared_price = (
        np.ones(int((simul_hours * 3600) / control_step))*50 # the mean price is 50
        - (
            np.mean(tess_simul.forecasted_data["TDryBul"])
            - tess_simul.forecasted_data["TDryBul"] # we are subtracting the temperature from the mean temperature to get the deviation
        )[0:-1] # the last value is not used because it is the last control step
    )

    simul_result = tess_simul.run_tess_simulation(
        market_expected_mean_price,
        market_expected_std_price,
        market_cleared_price
    )

    kpi = HVAC_KPI(tess_simul.simulation_results,
                control_step=control_step,
                heating_power_variable="reaPHeaPum_y",
                temperature_variable="reaTZon_y",
                setpoint_temperature=(customer_parameters["desired_temp"] - 32) * 5/9 + 273.15,
                temperature_unit="K")
    
    # save simulation results in data folder
    pd.DataFrame(tess_simul.simulation_results).to_csv(f"data/tess_simulation_results_{k_value}.csv", index=False)
    
    kpi_values = {
        "K_hvac": k_value,
        "Energy Consumption (kWh)": kpi.calculate_energy_consumption()/1000,
        "Peak Power (kW)": kpi.calculate_peak_power()/1000,
        "Temperature Discomfort": kpi.calculate_temperature_discomfort(),
        "Average Heating Cycles per hour": (kpi.calculate_cycles(kpi.heating_power_variable))['average_cycles_per_hour']
    }

    boptest_kpi = bt_instance.get_kpis().to_dict()["Value"] # dict of kpis from boptest
    # merge boptest kpis with tess kpis
    kpi_values = {**kpi_values, **boptest_kpi}


    kpi_values_list.append(kpi_values)
    

    bt_instance.stop()

kpi_df = pd.DataFrame(kpi_values_list)
kpi_df.to_csv("data/kpi_values.csv", index=False)
    