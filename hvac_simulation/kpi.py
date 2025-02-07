"""
This module contains the class to calculate KPIs for HVAC system performance over a period of time
"""
import numpy as np
import pandas as pd


class HVAC_KPI:
    """
    Class to calculate KPIs for HVAC system performance over a period of time
    """

    def __init__(
            self,
            hvac_operation_data,
            heating_power_variable = "fcu_reaPHea_y",
            cooling_power_variable = "fcu_reaPCoo_y",
            temperature_variable = "zon_reaTRooAir_y",
            setpoint_temperature = 22.0,
            temperature_unit = "C"
        ):
        """
        Initialize the KPI class with the data

        Args:   
        hvac_operation_data (pandas.DataFrame): Dataframe containing the HVAC operation data
        """
        self.hvac_operation_data = hvac_operation_data
        self.heating_power_variable = heating_power_variable
        self.cooling_power_variable = cooling_power_variable
        self.temperature_variable = temperature_variable
        self.setpoint_temperature = setpoint_temperature
        self.temperature_unit = temperature_unit


    def calculate_energy_consumption(self):
        """
        Calculate the total energy consumption of the HVAC system

        Returns:
        float: Total energy consumption in kWh
        """
        return (
            self.hvac_operation_data[self.heating_power_variable].sum()
            + self.hvac_operation_data[self.cooling_power_variable].sum()
        )
    
    def calculate_peak_power(self):
        """
        Calculate the peak power consumption of the HVAC system

        Returns:
        float: Peak power consumption in kW
        """
        return max(
            self.hvac_operation_data[self.heating_power_variable].max(),
            self.hvac_operation_data[self.cooling_power_variable].max()
        )
    
    def calculate_cycles(self, power_variable):
        """
        Calculate the total number of cycles of the HVAC system

        Returns:
        float: Average number of cycles per hour
        """
        self.hvac_operation_data["state"] = (
            self.hvac_operation_data[power_variable] > 0
        ).astype(int)

        self.hvac_operation_data["stage_change"] = self.hvac_operation_data["state"].diff()

        # indices where state changes from 0->1 and 1->0
        cycle_start = self.hvac_operation_data[self.hvac_operation_data["stage_change"] == 1].index
        cycle_end = self.hvac_operation_data[self.hvac_operation_data["stage_change"] == -1].index

        # edge cases
        if len(cycle_start) == 0 or len(cycle_end) == 0:
            return 0
        
        # if HVAC starts in heating mode, remove the first cycle start
        if cycle_end[0] < cycle_start[0]:
            cycle_end = cycle_end[1:]
        
        # if HVAC ends in heating mode, remove the last cycle end
        if cycle_start[-1] > cycle_end[-1]:
            cycle_start = cycle_start[:-1]

        # ensure equal number of cycle starts and ends
        n_cycles = min(len(cycle_start), len(cycle_end))
        cycle_start = cycle_start[:n_cycles]
        cycle_end = cycle_end[:n_cycles]

        cycles = pd.DataFrame({
            "start_time": self.hvac_operation_data.loc[cycle_start, "datetime"].values,
            "end_time": self.hvac_operation_data.loc[cycle_end, "datetime"].values
        })

        # calculate cycle durations
        cycles["duration_minutes"] = (cycles["end_time"] - cycles["start_time"]).dt.total_seconds() / 60

        # calculate average number of cycles per hour
        total_hours = (
            self.hvac_operation_data["datetime"].iloc[-1] - self.hvac_operation_data["datetime"].iloc[0]
        ).total_seconds() / 3600
        average_cycles_per_hour = n_cycles / total_hours if total_hours > 0 else 0

        return {
            "total_cycles": n_cycles,
            "average_cycles_per_hour": average_cycles_per_hour,
            "cycles": cycles
        }
    
    def calculate_temperature_discomfort(self):
        """
        Calculate the temperature discomfort of the HVAC system

        Returns:
        float: Average temperature discomfort in degrees Celsius
        """
        return np.linalg.norm(
            self.hvac_operation_data[self.temperature_variable] - self.setpoint_temperature
        ) / np.sqrt(len(self.hvac_operation_data))
