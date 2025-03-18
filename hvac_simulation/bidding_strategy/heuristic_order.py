""" 
This module contains the heuristic order class for the devices.

"""
from abc import ABC, abstractmethod
from enum import Enum
from statistics import NormalDist
import json
import numpy as np
import datetime as dt

class DeviceType(Enum):
    """
    The DeviceType class is an enumeration class that contains the device types.

    Attributes
    ----------
    HVAC : str
        The HVAC device type
    WATER_HEATER : str
        The water heater device type
    BATTERY : str
        The battery device type
    """
    HVAC = 'hvac'
    WATER_HEATER = 'water_heater'
    BATTERY = 'battery'

class Order(ABC):
    """
    The Order class is an abstract class that is inherited by the specific device order classes. The Order class is used to calculate the power price and quantity for the devices. The Order class contains the following methods:

        - set_market_parameters: This method is used to set the market parameters for the device order.
        - set_customer_parameters: This method is used to set the customer parameters for the device order.
        - set_device_parameters: This method is used to set the device parameters for the device order.
        - get_power_price: This method is used to calculate the power price for the device order.
        - get_power_quantity: This method is used to calculate the power quantity for the device order.

    Attributes
    ----------
    device_type : enum
        The type of the device (HVAC, WATER_HEATER, BATTERY)
    device_id : str
        The device id
    market_parameters : dict
        The market parameters for the device order
    customer_parameters : dict
        The customer parameters for the device order
    device_parameters : dict
        The device parameters for the device order
    bidding_strategy : str
        The bidding strategy for the device order
    creation_timestamp : str
        The timestamp for the device order in ISO8601 format

    Methods
    -------
    set_market_parameters(market_parameters)
        Set the market parameters for the device order
    set_customer_parameters(customer_parameters)
        Set the customer parameters for the device order
    set_device_parameters(device_parameters)
        Set the device parameters for the device order
    get_power_price()
        Calculate the power price for the device order
    get_power_quantity()
        Calculate the power quantity for the device order
    """

    def __init__(self, device_type):
        self.device_type = device_type
        self.device_id = None
        self.market_parameters = None
        self.customer_parameters = None
        self.device_parameters = None
        self.bidding_strategy = None
        self.creation_timestamp = dt.datetime.utcnow().isoformat()+"Z"

    def __str__(self):
        return f"Device Type: {self.device_type}, Device ID: {self.device_id}, Market Parameters: {self.market_parameters}, Customer Parameters: {self.customer_parameters}, Device Parameters: {self.device_parameters}, Bidding Strategy: {self.bidding_strategy}, Creation Timestamp: {self.creation_timestamp}"

    def __repr__(self):
        return f"Device Type: {self.device_type}, Device ID: {self.device_id}, Market Parameters: {self.market_parameters}, Customer Parameters: {self.customer_parameters}, Device Parameters: {self.device_parameters}, Bidding Strategy: {self.bidding_strategy}, Creation Timestamp: {self.creation_timestamp}"
                                                                                                                                                                                                                                                                                          
    def set_market_parameters(self, market_parameters):
        """
        Set the market parameters for the device order.

        Parameters
        ----------
        market_parameters : dict
            The market parameters for the device order

        Returns
        -------
        None
        """
        self.market_parameters = market_parameters

    def set_customer_parameters(self, customer_parameters):
        """
        Set the customer parameters for the device order.

        Parameters
        ----------
        customer_parameters : dict
            The customer parameters for the device order

        Returns
        -------
        None
        """
        self.customer_parameters = customer_parameters
    
    def set_device_parameters(self, device_parameters):
        """
        Set the device parameters for the device order.

        Parameters
        ----------
        device_parameters : dict
            The device parameters for the device order

        Returns
        -------
        None
        """
        self.device_parameters = device_parameters

    @abstractmethod
    def get_power_price(self):
        """
        Calculate the power price for the device order. This method is implemented in the subclass.
        """

    @abstractmethod
    def get_power_quantity(self):
        """
        Calculate the power quantity for the device order. This method is implemented in the subclass.
        """

class WaterHeaterOrder(Order):
    """
    The WaterHeaterOrder class is a subclass of the Order class. 
    This class is created for each the water heater device and based on the three parameters, the power price and quantity are calculated.

    Attributes
    ----------
    device_type : str
        The type of the device
    device_id : str
        The device id
    market_parameters : dict
        The market parameters for the device order
    customer_parameters : dict
        The customer parameters for the device order
    device_parameters : dict
        The device parameters for the device order
    bidding_strategy : str
        The bidding strategy for the device order

    Methods
    -------
    get_power_price()
        Calculate the power price for the water heater heuristic strategy
    get_power_quantity()
        Calculate the power quantity for the water heater heuristic strategy
    get_power_price_with_temperture()
        Calculate the power price for the water heater heuristic strategy based on the temperature
    get_power_price_with_states(weights=None)
        Calculate the power price for the water heater heuristic strategy based on the states
    get_power_price_cdf(cdf_value)
        Calculate the power price for the water heater heuristic strategy based on the cumulative distribution function
    """

    def __init__(self):
        super().__init__(DeviceType.WATER_HEATER)

    def get_power_price(self):
        """
        The hotwater heater strategy depends on the device, customer and market parameters. The strategy is based on the following heuristics:

        If the current temperature of the water heater is known: temperature_heuristic
            - If the current temperature is below the minimum temperature, the price is set to the maximum price
            - If the current temperature is above the maximum temperature, the price is set to the minimum price
            - Else the price is calculated based on the following formula:
                .. math:: \text{price} = \text{expected\_{price}} + 3 \times \text{K\_{wh}} \times \text{expected\_{stdev}} \times \left( \frac{\text{current\_{temp}} - \text{desired\_{temp}}}{\left| \text{min\_{temp}} - \text{desired\_{temp}} \right|} \right)

        If the current capacity of the water heater is known through skycentrics data: capacity_heuristic
            - The cumulative distribution function (CDF) is calculated as:
                .. math::\text{CDF} = \min \left( \max \left( 1 - \frac{\text{present\_{energy\_{storage\_{capacity}}}}}{\text{total\_{energy\_{storage\_{capacity}}}}}, 0 \right), 1 \right)
            - The price is calculated based on the CDF value
        
        If the previous states of the water heater are known through skycentrics data: states_heuristic
            - The price is calculated based on the previous states of the water heater
                .. math:: \text{CDF} = \min \left ( \max \left( \frac{\sum_{i=1}^{T} \text{S}_{i} \times i}{T}, 0 \right ), 1 \right)
                where :math:`\text{S}_{i}` is the present energy storage capacity at time :math:`i` and :math:`T` is time period
            - The price is calculated based on the CDF value

        If none of the above heuristics are applicable: random_heuristic
            - The price is calculated based on a random CDF value
        
        CDF based price calculation:
            .. math:: \text{price} = \text{expected\_{price}} + 3 \times \text{K\_{wh}} \times \text{expected\_{stdev}} \times \left( 0.5 - \text{NormalDist}(0, 1).\text{cdf}(\text{CDF}) \right), \text{min\_{price}} \right), \text{max\_{price}} \right)

        Parameters
        ----------
        None
        
        Returns
        -------
        p_order : float
            The price at which the participant/water heater will buy power from the auction for the current time period

        """
        # Device parameters
        current_temp = 0
        previous_states = None
        previous_power_consumption = None
        present_capacity = 0
        total_capacity = 0

        if "current_temp" in self.device_parameters:
            current_temp = self.device_parameters['current_temp']

        if "skycentrics_data" in self.device_parameters:
            if isinstance(self.device_parameters["skycentrics_data"], str):
                self.device_parameters["skycentrics_data"] = json.loads(self.device_parameters["skycentrics_data"])
            if "present_energy_storage_capacity" in self.device_parameters["skycentrics_data"]:
                present_capacity = self.device_parameters["skycentrics_data"]["present_energy_storage_capacity"][-1]
            if "total_energy_storage_capacity" in self.device_parameters["skycentrics_data"]:
                total_capacity = self.device_parameters["skycentrics_data"]["total_energy_storage_capacity"][-1]
            if "power" in self.device_parameters["skycentrics_data"]:
                previous_power_consumption = self.device_parameters["skycentrics_data"]["power"]
            if "state" in self.device_parameters["skycentrics_data"]:
                previous_states = self.device_parameters["skycentrics_data"]["state"]

        if current_temp != 0:
            p_order = self.get_power_price_with_temperture()
            self.bidding_strategy = "temperature_heuristic"
        else:
            if present_capacity != 0 and total_capacity != 0:
                cdf = min(max((1 - present_capacity/total_capacity), 0), 1)
                p_order = self.get_power_price_cdf(cdf)
                self.bidding_strategy = "capacity_heuristic"
            else:
                if previous_states is not None:
                    p_order = self.get_power_price_with_states()
                    self.bidding_strategy = "states_heuristic"
                else:
                    p_order = self.get_power_price_cdf(np.random.rand())
                    self.bidding_strategy = "random_heuristic"
        return p_order
    
    def get_power_price_with_temperture(self):
        """
        This method calculates the power price for the water heater heuristic strategy based on the temperature, if the data is available.

        Parameters
        ----------
        None

        Returns
        -------
        p_order : float
            The price at which the participant will buy power from the auction for the current time period
        """
        expected_stdev = self.market_parameters['expected_stdev']
        expected_price = self.market_parameters['expected_price']
        expected_stdev = self.market_parameters['expected_stdev'] if self.market_parameters['expected_stdev'] != 0 else expected_price*0.001
        min_price = self.market_parameters['min_price']
        max_price = self.market_parameters['max_price']
        # Customer parameters
        k_waterheater = self.customer_parameters['K_waterheater']
        desired_temp = self.customer_parameters['desired_temp']
        # Device parameters
        min_temp = self.device_parameters['min_temp']
        max_temp = self.device_parameters['max_temp']
        current_temp = self.device_parameters['current_temp']
        if current_temp < min_temp:
            p_order = max_price
        elif current_temp > max_temp:
            p_order = min_price
        else:
            p_order = min(
                max(
                    expected_price + 3 * k_waterheater * expected_stdev * ((current_temp - desired_temp)/abs(min_temp - desired_temp)), 
                    min_price),
                max_price)
        return p_order

    def get_power_price_with_states(self, weights=None):
        """
        This method calculates the power price for the water heater heuristic strategy based on the states, if the data is available.

        Parameters
        ----------
        weights : list
            The weights for the previous states of the water heater. This has to be multiplied with the previous states to get the cumulative distribution function (CDF) value.
            The final CDF value should be between 0 and 1.

        Returns
        -------
        p_order : float
            The price at which the participant will buy power from the auction for the current time period
        """
        previous_states = self.device_parameters["skycentrics_data"]["state"]

        if weights is None:
            weights = ((np.arange(len(previous_states))+1)/len(previous_states))
        cdf_states = np.average(
            previous_states,
            weights=weights
        )
        return self.get_power_price_cdf(cdf_states)

    def get_power_price_cdf(self, cdf_value)->float:
        """
        The hotwater heater strategy is similar to that of the HVAC system in heating mode

        Parameters
        ----------
        None
        
        Returns
        -------
        p_order : float
            The price at which the participant will buy power from the 
            auction for the current time period
        """
        # Market parameters
        expected_stdev = self.market_parameters['expected_stdev']
        expected_price = self.market_parameters['expected_price']
        min_price = self.market_parameters['min_price']
        max_price = self.market_parameters['max_price']
        # Customer parameters
        k_waterheater = self.customer_parameters['K_waterheater']
        current_state = 0.5 - NormalDist(0, 1).cdf(cdf_value)
        p_order = min(
            max(
                expected_price + 3 * k_waterheater * expected_stdev * current_state,
                min_price),
            max_price)
        return p_order

    def get_power_quantity(self)->float:
        """
        The hotwater heater quantity

        Parameters
        ----------
        None

        Returns
        -------
        q_order : float
            The quantity of power that the participant will buy from the auction for the current time period
        """
        current_temp = self.device_parameters['current_temp']
        max_temp = self.device_parameters['max_temp']
        power_rating = self.device_parameters['power_rating']
        q_order = 0
        if current_temp < max_temp:
            q_order = power_rating
        return q_order

class HVACOrder(Order):

    def __init__(self):
        super().__init__('hvac')

    def get_power_price(self) -> float:
        """
        Calculate the power price for the HVAC system through heuristic strategy. 

        Parameters
        ----------
        None

        Returns
        -------
        p_order : float
            The price at which the participant will buy power from the auction for the current time period
        """
        # Market parameters
        expected_stdev = self.market_parameters['expected_stdev']
        expected_price = self.market_parameters['expected_price']
        min_price = self.market_parameters['min_price']
        max_price = self.market_parameters['max_price']
        # Customer parameters
        K_hvac = self.customer_parameters['K_hvac']
        desired_temp = self.customer_parameters['desired_temp']
        # Device parameters
        current_temp = self.device_parameters['current_temp']
        last_change_temp = self.device_parameters['last_change_temp']
        max_temp = self.device_parameters['max_temp']
        min_temp = self.device_parameters['min_temp']
        mode = self.device_parameters['mode']
        state = self.device_parameters['state']
        fan_mode = self.device_parameters['fan_mode']
        fan_state = self.device_parameters['fan_state']
        power_rating = self.device_parameters['power_rating']
        
        # seconds_since_on_change = self.device_parameters['seconds_since_on_change']

        
        # # make sure the device is on for at least 15 minutes before bidding
        # if state!="off" and seconds_since_on_change < 20*60:
        #     return max_price
        # print("\n" + "="*50)
        # print("BID PRICE CALCULATION DEBUG")
        # print("="*50)
        # print(f"Mode: {mode}")
        # print(f"State: {state}")
        # print(f"Current Temperature: {current_temp:.2f}°F")
        # print(f"Desired Temperature: {desired_temp:.2f}°F")
        # print(f"Temperature Range: {min_temp:.2f}°F to {max_temp:.2f}°F")
        # print(f"Last Temperature Change: {last_change_temp:.2f}°F")
        # print(f"Market Parameters:")
        # print(f"  - Expected Price: ${expected_price:.2f}")
        # print(f"  - Expected Std Dev: ${expected_stdev:.2f}")
        # print(f"  - Min Price: ${min_price:.2f}")
        # print(f"  - Max Price: ${max_price:.2f}")
        # print(f"Customer Parameter:")
        # print(f"  - K_hvac: {K_hvac:.2f}")
        
        if expected_stdev == 0:
            expected_stdev = expected_price*0.001
            print(f"Warning: Expected std dev was 0, adjusted to: ${expected_stdev:.2f}")
        
        p_order = 0
        if mode == 'heating' or (mode == 'auto' and last_change_temp < 0):
            # print("\nCalculating heating mode bid price...")
            if current_temp < min_temp:
                p_order = max_price
                # print(f"Temperature below minimum ({min_temp:.2f}°F), using max price: ${p_order:.2f}")
            elif current_temp > max_temp:
                p_order = min_price
                # print(f"Temperature above maximum ({max_temp:.2f}°F), using min price: ${p_order:.2f}")
            elif (current_temp != desired_temp):
                temp_delta = desired_temp - current_temp
                temp_range = abs(min_temp - desired_temp)
                price_adjustment = K_hvac * expected_stdev * (temp_delta/temp_range)
                p_order = min(
                    max(
                        expected_price + price_adjustment,
                        min_price),
                    max_price)
                # print(f"Temperature delta: {temp_delta:.2f}°F")
                # print(f"Temperature range: {temp_range:.2f}°F")
                # print(f"Price adjustment: ${price_adjustment:.2f}")
                # print(f"Final bid price: ${p_order:.2f}")
                
        elif mode == 'cooling' or (mode == 'auto' and last_change_temp > 0):
            # print("\nCalculating cooling mode bid price...")
            if current_temp < min_temp:
                p_order = min_price
                # print(f"Temperature below minimum ({min_temp:.2f}°F), using min price: ${p_order:.2f}")
            elif current_temp > max_temp:
                p_order = max_price
                # print(f"Temperature above maximum ({max_temp:.2f}°F), using max price: ${p_order:.2f}")
            elif (current_temp != desired_temp):
                temp_delta = current_temp - desired_temp
                temp_range = abs(max_temp - desired_temp)
                price_adjustment = K_hvac * expected_stdev * (temp_delta/temp_range)
                p_order = min(
                    max(
                        expected_price + price_adjustment,
                        min_price),
                    max_price)
                # print(f"Temperature delta: {temp_delta:.2f}°F")
                # print(f"Temperature range: {temp_range:.2f}°F")
                # print(f"Price adjustment: ${price_adjustment:.2f}")
                # print(f"Final bid price: ${p_order:.2f}")
        
        # print("="*50 + "\n")
        return p_order

    def get_power_quantity(self) -> float:
        """
        Calculate the power quantity for the HVAC system through heuristic strategy.

        Parameters
        ----------
        None

        Returns
        -------
        q_order : float
            The quantity of power that the participant will buy from the auction for the current time period
        """
        q_order = 0
        # Device parameters
        current_temp = self.device_parameters['current_temp']
        last_change_temp = self.device_parameters['last_change_temp']
        max_temp = self.device_parameters['max_temp']
        min_temp = self.device_parameters['min_temp']
        mode = self.device_parameters['mode']
        state = self.device_parameters['state']
        fan_mode = self.device_parameters['fan_mode']
        fan_state = self.device_parameters['fan_state']
        power_rating = self.device_parameters['power_rating']

        # add auxiliary power either for heating or seperate
        q_fan = power_rating['fan'] if fan_state == 'on' else 0
        # add auxiliary power either for heating or seperate
        q_fan = power_rating['fan'] if fan_state == 'on' else 0
        if mode == 'heating' or (mode == 'auto' and last_change_temp < 0):
            q_order = power_rating['heating']
        elif mode == 'cooling' or (mode == 'auto' and last_change_temp > 0):
            q_order = power_rating['cooling']
        return q_order + q_fan

class BatteryOrder(Order):

    def __init__(self):
        super().__init__(DeviceType.BATTERY)

    def get_power_price(self)->float:
        """
        Calculate the power price for the battery heuristic strategy.
        Parameters
        ----------
        None

        Returns
        -------
        p_order : float
            The price at which the participant will buy or sell power from the auction for the current time period
        """
        # Market parameters
        expected_stdev = self.market_parameters['expected_stdev']
        expected_price = self.market_parameters['expected_price']
        min_price = self.market_parameters['min_price']
        max_price = self.market_parameters['max_price']
        market_interval = self.market_parameters['interval'] # market interval in seconds
        # Customer parameters
        K_battery = self.customer_parameters['K_battery']
        desired_state_of_charge = self.customer_parameters['desired_state_of_charge'] # desired state of charge in percentage
        # Device parameters
        buffer = self.device_parameters["buffer"]
        charging = self.device_parameters["charging"]
        discharging = self.device_parameters["discharging"]
        on_grid = self.device_parameters["on_grid"]
        state_of_charge = self.device_parameters["state_of_charge"] # current state of charge in percentage
        max_state_of_charge = self.device_parameters["max_state_of_charge"] # maximum state of charge in percentage
        min_state_of_charge = self.device_parameters["min_state_of_charge"] # minimum state of charge in percentage
        battery_rating = self.device_parameters["battery_rating"] # battery rating in kW
        battery_capacity = self.device_parameters["battery_capacity"] # battery capacity in kWh
        mode = self.device_parameters["mode"]

        dead_band = (max_state_of_charge - min_state_of_charge) * 0.05
        # dead_band = 0

        p_order = 0
        if expected_stdev == 0:
            expected_stdev = expected_price*0.001
        if state_of_charge <= min_state_of_charge:
            p_order = max_price
        elif state_of_charge >= max_state_of_charge:
            p_order = min_price
        elif min_state_of_charge < state_of_charge < desired_state_of_charge - dead_band:
            cdf = min(max(1 - (state_of_charge - min_state_of_charge)/(desired_state_of_charge - dead_band - min_state_of_charge), 0), 1)
            p_order = NormalDist(mu=expected_price, sigma=K_battery*expected_stdev).inv_cdf(cdf)
        elif desired_state_of_charge + dead_band < state_of_charge < max_state_of_charge:
            cdf = min(max(1 - (state_of_charge - (desired_state_of_charge + dead_band))/(max_state_of_charge - (desired_state_of_charge + dead_band)), 0), 1)
            p_order = NormalDist(mu=expected_price, sigma=K_battery*expected_stdev).inv_cdf(cdf)
        # Include the RTE to handle
        # Implement deadband 1% so that it is not always SOC
        p_order = min(max(p_order, min_price), max_price)
        return p_order

    def get_power_quantity(self)->float:

        Qorder = 0
        market_interval = self.market_parameters['interval'] # market interval in seconds
        dead_band = (self.device_parameters["max_state_of_charge"] - self.device_parameters["min_state_of_charge"]) * 0.05
        if self.device_parameters["state_of_charge"] < self.customer_parameters['desired_state_of_charge'] - dead_band: # Charging or Buying
            Qorder = min(
                self.device_parameters["battery_rating"],
                (self.customer_parameters['desired_state_of_charge'] - self.device_parameters["state_of_charge"]) * self.device_parameters["battery_capacity"] / (market_interval/3600) # convert to kW
            )
        elif self.device_parameters["state_of_charge"] > self.customer_parameters['desired_state_of_charge'] + dead_band: # Discharging or Selling
            Qorder = max(
                -self.device_parameters["battery_rating"],
                -(self.device_parameters["state_of_charge"] - self.customer_parameters['desired_state_of_charge']) * self.device_parameters["battery_capacity"] / (market_interval/3600) # convert to kW
            )
        return Qorder
