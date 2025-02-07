import requests
import json
import pandas as pd
from typing import List, Dict, Union, Optional
from datetime import datetime, timedelta
import marimo as mo
import matplotlib.pyplot as plt
import numpy as np

KPI_LABELS = {
    "cost_tot": "HVAC energy cost in $/m2 or Euro/m2",
    "emis_tot": "HVAC energy emissions in kgCO2e/m2",
    "ener_tot": "HVAC energy total in kWh/m2",
    "pele_tot": "HVAC peak electrical demand in kW/m2",
    "pgas_tot": "HVAC peak gas demand in kW/m2",
    "pdih_tot": "HVAC peak district heating demand in kW/m2",
    "idis_tot": "Indoor air quality discomfort in ppmh/zone",
    "tdis_tot": "Thermal discomfort in Kh/zone",
    "time_rat": "Computational time ratio in s/ss"
}

def seconds_to_datetime(seconds: float) -> datetime:
        """Convert seconds to date:hour format.

        Args:
            seconds: Number of seconds since start of year

        Returns:
            String in format 'MM-DD:HH' representing the date and hour

        Example:
            >>> BOPTESTClient.seconds_to_datetime(86400)  # 24 hours
            '01-02:00'  # January 2nd, midnight
        """
        # Start from January 1st
        base_date = datetime(2023, 1, 1, 0, 0, 0)  # Using 2024 as it's a leap year
        # Add the seconds
        target_date = base_date + timedelta(seconds=seconds)

        return target_date

def calculate_thermal_discomfort(
    air_temp: Union[np.ndarray, List[float], pd.Series],
    setpoint: Union[np.ndarray, List[float], pd.Series],
    time_step: float = 900.0  # Default 15 minutes in seconds
) -> float:
    """
    Calculate thermal discomfort (Kh) based on temperature violations.
    
    Args:
        air_temp: Zone air temperature time series [K or °C]
        cooling_setpoint: Cooling setpoint time series [K or °C]
        heating_setpoint: Heating setpoint time series [K or °C]
        time_step: Time step between measurements in seconds
        
    Returns:
        float: Thermal discomfort in Kelvin-hours (Kh)
        
    Note:
        All temperature inputs must be in the same unit (either all Kelvin or all Celsius)
        The output will be in Kelvin-hours regardless of input units since we're measuring differences
    """
    # Convert inputs to numpy arrays if they aren't already
    air_temp = np.array(air_temp)
    setpoint = np.array(setpoint)
    
    # Verify all arrays have the same length
    if not (len(air_temp) == len(setpoint)):
        raise ValueError("All input arrays must have the same length")
    
    t_discomfort = np.linalg.norm(air_temp - setpoint, ord=2) * time_step
    
    return t_discomfort

def create_dr_control_inputs(
    simul_start_date: datetime,
    simul_days: int,
    control_step: int,
    dr_events: List[Dict],
    dr_cooling_setpoint: float = 308.15,      # 35°C
) -> pd.DataFrame:
    """
    Create control inputs DataFrame with demand response events.
    
    Args:
        simul_start_date: Start date of simulation
        simul_days: Number of days to simulate
        control_step: Control step in seconds
        inputs_available: DataFrame of available inputs
        dr_events: List of DR events with start_time, duration, and notification_delta
        dr_cooling_setpoint: DR event cooling setpoint in Kelvin
        default_heating_setpoint: Heating setpoint in Kelvin
        
    Returns:
        DataFrame with control inputs for entire simulation period
    """
    # Create time series for entire simulation period
    simulation_time_series = pd.date_range(
        start=simul_start_date,
        end=simul_start_date + timedelta(days=simul_days),
        freq=f"{control_step}s"
    )
    
    # Initialize control inputs DataFrame
    control_inputs = pd.DataFrame(simulation_time_series, columns=["datetime"])
    
    # # Set default temperature setpoints
    control_inputs["con_oveTSetCoo_u"] = None
    control_inputs["con_oveTSetCoo_activate"] = 0
    control_inputs["con_oveTSetHea_u"] = 278.15
    control_inputs["con_oveTSetHea_activate"] = 1
    
    # Process each DR event
    for event in dr_events:
        # Convert start_time to datetime if it's a string
        if isinstance(event['start_time'], str):
            start_time = pd.to_datetime(event['start_time'])
        else:
            start_time = event['start_time']
            
        # Calculate end time
        end_time = start_time + timedelta(hours=event['duration'])
        
        # Create mask for DR event period
        dr_mask = (control_inputs['datetime'] >= start_time) & \
                 (control_inputs['datetime'] < end_time)
        
        # Apply DR settings during event
        control_inputs.loc[dr_mask, "con_oveTSetCoo_u"] = dr_cooling_setpoint
        control_inputs.loc[dr_mask, "con_oveTSetCoo_activate"] = 1
        
    return control_inputs

class BOPTESTClient:
    """
    Client for BOPTEST API.

    Parameters
    ----------

    base_url : str
        Base URL for BOPTEST API.

    testcase : str
        Test case name.

    Attributes
    ----------
    base_url : str
        Base URL for BOPTEST API.

    testcase : str
        Test case name.

    test_id : str
        Test ID for selected test case.

    timeout : int
        Timeout for API requests in seconds.

    Methods
    -------
    get_version()
        Get BOPTEST version.

    get_name()
        Get test case name.

    get_measurements()
        Get available sensor measurements.

    get_inputs()
        Get available control inputs.

    get_step()
        Get current control step.

    set_step(step: float)
        Set control step in seconds.

    initialize(start_time: float, warmup_period: float)
        Initialize simulation with start time and warmup period.

    get_scenario()
        Get current test scenario.

    set_scenario(electricity_price: Optional[str] = None, time_period: Optional[str] = None)
        Set test scenario.

    get_forecast_points()
        Get available forecast points.

    get_forecast(point_names: List[str], horizon: float, interval: float)
        Get forecasts for specified points.

    advance(inputs: Optional[Dict[str, float]] = None)
        Advance simulation one control step.

    get_results(point_names: List[str], start_time: float, final_time: float)
        Get simulation results for specified points and time period.

    get_kpis()
        Get KPI values.

    submit_results(api_key: str, tags: Optional[List[str]] = None)
        Submit results to dashboard.

    run_simulation(duration_hours: float = 24, timestep: float = 900, control_inputs: Optional[Dict] = None)
        Run simulation for specified duration with optional control inputs.
    """

    def __init__(self, base_url: str, testcase: str, timeout: int = 10, control_step: float = 900, temp_unit='C', start_date: datetime = datetime(2023, 1, 1, 0, 0, 0)):
        """Initialize BOPTEST client with base URL and testcase."""
        self.timeout = timeout
        self.base_url = base_url.rstrip('/')
        self.testcase = testcase
        self.test_id = self._select_testcase()
        self.control_step = control_step
        self.temp_unit = temp_unit
        self.start_date = start_date
        self.set_input_output_parameters()
 
    def __repr__(self):
        """Return string representation of BOPTEST client."""
        return f"BOPTESTClient(base_url='{self.base_url}', testcase='{self.testcase}') test_id='{self.test_id}'"
    
    def __str__(self):
        """Return string representation of BOPTEST client."""
        return f"BOPTESTClient(base_url='{self.base_url}', testcase='{self.testcase}') test_id='{self.test_id}'"

    def _select_testcase(self) -> str:
        """Select the testcase and return test ID."""
        response = requests.post(f"{self.base_url}/testcases/{self.testcase}/select", timeout=self.timeout)
        return response.json()['testid']
    
    def _handle_response(self, response: requests.Response) -> Union[Dict, List]:
        """Handle API response and return payload."""
        if response.status_code == 200:
            try:
                result = response.json()['payload']
                return result
            except json.JSONDecodeError:
                return response.text
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
    
    def get_version(self) -> str:
        """Get BOPTEST version."""
        response = requests.get(f"{self.base_url}/version/{self.test_id}", timeout=self.timeout)
        return self._handle_response(response)['version']
    
    def get_name(self) -> str:
        """Get test case name."""
        response = requests.get(f"{self.base_url}/name/{self.test_id}", timeout=self.timeout)
        return self._handle_response(response)['name']
    
    def set_input_output_parameters(self) -> None:
        """Set input/output parameters."""
        self.measurements = self.get_measurements()
        self.inputs = self.get_inputs()
        self.forecast_points = self.get_forecast_points()
        self.set_control_step(self.control_step)

    def stop(self) -> None:
        """Stop the simulation."""
        response = requests.put(f"{self.base_url}/stop/{self.test_id}", timeout=self.timeout)
        return self._handle_response(response)

    def get_measurements(self) -> Dict:
        """Get available sensor measurements."""
        response = requests.get(f"{self.base_url}/measurements/{self.test_id}", timeout=self.timeout)
        response = pd.DataFrame(self._handle_response(response)).T
        response.index.name = "Variable Name"
        return response
    
    def get_inputs(self) -> Dict:
        """Get available control inputs."""
        response = requests.get(f"{self.base_url}/inputs/{self.test_id}", timeout=self.timeout)
        response = self._handle_response(response)
        response = pd.DataFrame(response).T
        response.index.name = "Variable Name"
        return response
    
    def get_control_step(self) -> float:
        """Get current control step."""
        response = requests.get(f"{self.base_url}/step/{self.test_id}", timeout=self.timeout)
        return self._handle_response(response)
    
    def set_control_step(self, step: float) -> float:
        """Set control step in seconds."""
        self.control_step = step
        response = requests.put(f"{self.base_url}/step/{self.test_id}", data={'step': step}, timeout=self.timeout)
        return self._handle_response(response)
    
    def initialize(self, start_time: datetime, warmup_period: float = 7*24*3600) -> Dict:
        """
        Initialize simulation with start time and warmup period.
        
        Args:
            start_time: Start time.
            warmup_period: Warmup period in seconds.

        Returns:
            Dict: Response payload.
        """
        start_time_sec = int((start_time - self.start_date).total_seconds()) - 3600
        self.simulation_start_time = start_time
        response = requests.put(
            f"{self.base_url}/initialize/{self.test_id}",
            data={
                'start_time': start_time_sec,
                'warmup_period': warmup_period
            },
            timeout=self.timeout
        )
        return self._handle_response(response)
    
    def get_scenario(self) -> Dict:
        """Get current test scenario."""
        response = requests.get(f"{self.base_url}/scenario/{self.test_id}", timeout=self.timeout)
        return self._handle_response(response)
    
    def set_scenario(self, electricity_price: Optional[str] = None, 
                    time_period: Optional[str] = None) -> Dict:
        """Set test scenario."""
        data = {}
        if electricity_price:
            data['electricity_price'] = electricity_price
        if time_period:
            data['time_period'] = time_period
            
        response = requests.put(f"{self.base_url}/scenario/{self.test_id}", data=data, timeout=self.timeout)
        return self._handle_response(response)
    
    def get_forecast_points(self) -> Dict:
        """Get available forecast points."""
        response = requests.get(f"{self.base_url}/forecast_points/{self.test_id}", timeout=self.timeout)
        response = self._handle_response(response)
        response = pd.DataFrame(response).T
        response.index.name = "Variable Name"
        return response
    
    def get_forecast(self, horizon_hours: float, interval: float = None) -> Dict:
        """Get forecasts for specified points."""
        interval = interval if interval else self.control_step
        response = requests.put(
            f"{self.base_url}/forecast/{self.test_id}",
            data={
                'point_names': self.forecast_points.index.tolist(),
                'horizon': horizon_hours*3600,
                'interval': interval
            },
            timeout=self.timeout
        )
        response_data = self._handle_response(response)
        response_data = pd.DataFrame(response_data)
        response_data["datetime"] = response_data["time"].apply(seconds_to_datetime)
        return response_data
    
    def advance(self, inputs: Optional[Dict[str, float]] = None) -> Dict:
        """Advance simulation one control step."""
        data_inputs = {k: v for k, v in inputs.items() if v is not None} if inputs else {}
        response = requests.post(f"{self.base_url}/advance/{self.test_id}", data=data_inputs, timeout=self.timeout)
        return self._handle_response(response)
    
    def get_results(self, point_names: List[str], start_time: float,
                   final_time: float) -> pd.DataFrame:
        """Get simulation results for specified points and time period."""
        response = requests.put(
            f"{self.base_url}/results/{self.test_id}",
            data={
                'point_names': point_names,
                'start_time': start_time,
                'final_time': final_time
            },
            timeout=self.timeout
        )
        return pd.DataFrame(self._handle_response(response))
    
    def get_kpis(self) -> Dict:
        """Get KPI values."""
        response = requests.get(f"{self.base_url}/kpi/{self.test_id}", timeout=self.timeout)
        response = self._handle_response(response)
        kpi_measures = pd.DataFrame(response, index=[0]).T
        kpi_measures.columns = ["Value"]
        kpi_measures.index.name = "KPI"
        kpi_measures["Description"] = kpi_measures.index.map(KPI_LABELS)
        # rounding off the values
        kpi_measures["Value"] = kpi_measures["Value"]
        kpi_measures  = kpi_measures[["Description", "Value"]]
        return kpi_measures
    
    def submit_results(self, api_key: str, tags: Optional[List[str]] = None) -> str:
        """Submit results to dashboard."""
        data = {'api_key': api_key}
        if tags:
            for i, tag in enumerate(tags[:10], 1):
                data[f'tag{i}'] = tag
                
        response = requests.get(f"{self.base_url}/submit/{self.test_id}", params=data, timeout=self.timeout)
        return self._handle_response(response)['identifier']

    def convert_temperature_variables(self, sim_results: pd.DataFrame) -> pd.DataFrame:
        """Convert temperature from Kelvin to Celsius."""
        temp_variables = (
            list(self.measurements[self.measurements["Unit"] == "K"].index)
            + list(self.inputs[self.inputs["Unit"] == "K"].index)
        )
        for temp_var in temp_variables:
            if temp_var in sim_results.columns:
                sim_results[temp_var] = sim_results[temp_var].apply(self.convert_temperature_value)
        return sim_results

    def convert_temperature_value(self, temparature: float) -> float:
        """Convert temperature value from Kelvin to self.temp_units."""
        if self.temp_unit == 'C':
            return temparature - 273.15
        elif self.temp_unit == 'F':
            return (temparature - 273.15) * 9/5 + 32

    def get_control_input(
            self,
            control_input: pd.DataFrame,
            current_time: datetime
        ) -> Optional[Dict[str, float]]:
        """
        Get control input for current time
        """
        if control_input is None:
            return None
        # find the control input for current time
        filter_idx = (
            (control_input["datetime"].dt.year == current_time.year) &
            (control_input["datetime"].dt.month == current_time.month) &
            (control_input["datetime"].dt.day == current_time.day) &
            (control_input["datetime"].dt.hour == current_time.hour) &
            (control_input["datetime"].dt.minute == current_time.minute) &
            (control_input["datetime"].dt.second == current_time.second)
        )
        if filter_idx.sum() == 0:
            print(f"No control input found for {current_time}")
            return None
        control_input = control_input[filter_idx].to_dict(orient='records')[0]
        # del datetime key and keys where value is None
        control_input.pop("datetime")
        control_input = {k: v for k, v in control_input.items() if v is not None}
        return control_input if control_input else None

    def run_simulation(
            self,
            duration_hours: float = 24,
            control_inputs: pd.DataFrame = None,
            prog_bar: mo.status.progress_bar = None
        ) -> pd.DataFrame:
        """
        Run simulation for specified duration with optional control inputs.
        
        Args:
            duration_hours: Duration of simulation in hours.
            control_inputs: Optional control inputs with keys as timestep to apply control and values as control values.

        Returns:
            pd.DataFrame: Simulation results.
        """
        num_steps = int(((duration_hours) * 3600) / self.control_step)
        simulation_results = []
        for i in range(num_steps):
            # datetime of current step
            current_time = self.simulation_start_time + timedelta(seconds=i * self.control_step)
            control_input = self.get_control_input(control_inputs, current_time)
            print(f"Current time: {current_time}, control_input: {control_input}")
            y = self.advance(control_input)
            simulation_results.append(y)
            
            if prog_bar:
                prog_bar.update(subtitle=f"Current step time: {seconds_to_datetime(y['time'])}")
        
        simulation_results = pd.DataFrame(simulation_results)
        simulation_results["datetime"] = simulation_results["time"].apply(seconds_to_datetime)
        simulation_results = self.convert_temperature_variables(simulation_results)
        self.simulation_results = simulation_results
        return simulation_results

    def plot_simulation_results(
            self,
            temp_y1_vars: List[str] = [],
            temp_y2_vars: List[str] = [],
            power_vars: List[str] = [],
            control_vars: List[str] = [],
    ):
        """
        Plot simulation results

        Args:
            temp_y1_vars: List of temperature variables to plot on y1 axis.
            temp_y2_vars: List of temperature variables to plot on y2 axis.
            power_vars: List of power variables to plot.
            control_vars: List of control variables to plot.

        Returns:
            fig: matplotlib figure object.
        """
        
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

        fig, ax = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

        ax_0_0 = ax[0].twinx()

        for temp_var in temp_y1_vars:
            ax[0].plot(self.simulation_results["datetime"], self.simulation_results[temp_var], label=temp_var)
        ax[0].set_ylabel(f"Temperature ({self.temp_unit})")
        ax[0].set_title("Temperature Variables")
        ax[0].legend(loc=(1.1, 0))

        for temp_var in temp_y2_vars:
            ax_0_0.plot(self.simulation_results["datetime"], self.simulation_results[temp_var], label=temp_var, color='r')
        ax_0_0.set_ylabel(f"Temperature {self.temp_unit}", color='r')
        ax_0_0.legend(loc=(1.1, 0.5))

        for power_var in power_vars:
            ax[1].plot(self.simulation_results["datetime"], self.simulation_results[power_var], label=power_var)
        ax[1].set_ylabel("Power (W)")
        ax[1].set_title("Power Variables")
        ax[1].legend(loc=(1.1, 0))

        for control_var in control_vars:
            ax[2].plot(self.simulation_results["datetime"], self.simulation_results[control_var], label=control_var)
        ax[2].set_ylabel("Control Value")
        ax[2].set_title("Control Variables")
        ax[2].legend(loc=(1.1, 0))

        ax[2].set_xlabel("Time")

        # set x-axis major locator to 10 ticks total and minor locator to 2 ticks per major tick
        ax[2].xaxis.set_major_locator(plt.MaxNLocator(10))
        ax[2].xaxis.set_minor_locator(plt.MaxNLocator(20))
        # rotate x-axis labels for better readability
        plt.setp(ax[2].xaxis.get_majorticklabels(), rotation=45)

        return fig
