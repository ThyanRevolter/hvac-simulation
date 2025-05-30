"""
This module contains the heuristic order class for the devices.

"""

from abc import ABC, abstractmethod
from enum import Enum
import datetime as dt
from datetime import timezone
from statistics import NormalDist
import json
import numpy as np


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

    HVAC = "hvac"
    WATER_HEATER = "water_heater"
    BATTERY = "battery"


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
    auction_id : str
        The auction id
    device_id : str
        The device id
    market_parameters : dict
        The market parameters for the device order:
        - expected_price: The expected price of the power  # TODO: implement this - Mayank Code Review 3/11
        - expected_stdev: The expected standard deviation of the power # TODO: implement this - Mayank Code Review 3/11
        - min_price: The minimum price of the power
        - max_price: The maximum price of the power
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

    def __init__(self, device_type, auction_id):
        self.device_type = device_type
        self.auction_id = auction_id
        self.device_id = None
        self.market_parameters = None
        self.customer_parameters = None
        self.device_parameters = None
        self.bidding_strategy = None
        self.creation_timestamp = dt.datetime.now(timezone.utc).isoformat() + "Z"

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
    get_power_price_with_states(weights=None)
        Calculate the power price for the water heater heuristic strategy based on the states
    get_power_price_cdf(cdf_value)
        Calculate the power price for the water heater heuristic strategy based on the cumulative distribution function
    """

    def __init__(self, auction_id):
        super().__init__(DeviceType.WATER_HEATER, auction_id)

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
        # Market parameters
        expected_stdev = self.market_parameters["expected_stdev"]
        expected_price = self.market_parameters["expected_price"]
        min_price = self.market_parameters["min_price"]
        max_price = self.market_parameters["max_price"]
        
        # Customer parameters
        k_waterheater = self.customer_parameters["K_waterheater"]
        desired_temp = self.customer_parameters["desired_temp"]
        
        # Device parameters
        current_temp = self.device_parameters.get("current_temp", 0)
        min_temp = self.device_parameters.get("min_temp", 0)
        max_temp = self.device_parameters.get("max_temp", 0)
        
        # Initialize variables for skycentrics data
        present_capacity = 0
        total_capacity = 0
        previous_states = None
        
        # Parse skycentrics data if available
        if "skycentrics_data" in self.device_parameters:
            skycentrics_data = self.device_parameters["skycentrics_data"]
            if isinstance(skycentrics_data, str):
                skycentrics_data = json.loads(skycentrics_data)
            
            if "present_energy_storage_capacity" in skycentrics_data:
                present_capacity = skycentrics_data["present_energy_storage_capacity"][-1]
            if "total_energy_storage_capacity" in skycentrics_data:
                total_capacity = skycentrics_data["total_energy_storage_capacity"][-1]
            if "state" in skycentrics_data:
                previous_states = skycentrics_data["state"]

        # Apply appropriate bidding strategy
        if current_temp != 0:
            # Temperature-based heuristic
            self.bidding_strategy = "temperature_heuristic"
            if current_temp < min_temp:
                p_order = max_price
            elif current_temp > max_temp:
                p_order = min_price
            else:
                temp_ratio = (current_temp - desired_temp) / abs(min_temp - desired_temp)
                p_order = min(
                    max(
                        expected_price + 3 * k_waterheater * expected_stdev * temp_ratio,
                        min_price
                    ),
                    max_price
                )
        elif present_capacity != 0 and total_capacity != 0:
            # Capacity-based heuristic
            self.bidding_strategy = "capacity_heuristic"
            cdf = min(max(1 - present_capacity / total_capacity, 0), 1)
            p_order = self.get_power_price_cdf(cdf)
        elif previous_states is not None:
            # States-based heuristic
            self.bidding_strategy = "states_heuristic"
            weights = np.arange(len(previous_states)) + 1
            cdf = min(max(np.average(previous_states, weights=weights), 0), 1)
            p_order = self.get_power_price_cdf(cdf)
        else:
            # Random heuristic
            self.bidding_strategy = "random_heuristic"
            p_order = self.get_power_price_cdf(np.random.rand())
            
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
            weights = (np.arange(len(previous_states)) + 1) / len(previous_states)
        cdf_states = np.average(previous_states, weights=weights)
        return self.get_power_price_cdf(cdf_states)

    def get_power_price_cdf(self, cdf_value) -> float:
        """
        Calculate the power price based on a CDF value.

        Parameters
        ----------
        cdf_value : float
            The CDF value to use for price calculation

        Returns
        -------
        float
            The calculated power price
        """
        expected_stdev = self.market_parameters["expected_stdev"]
        expected_price = self.market_parameters["expected_price"]
        min_price = self.market_parameters["min_price"]
        max_price = self.market_parameters["max_price"]
        k_waterheater = self.customer_parameters["K_waterheater"]
        
        current_state = 0.5 - NormalDist(0, 1).cdf(cdf_value)
        p_order = min(
            max(
                expected_price + 3 * k_waterheater * expected_stdev * current_state,
                min_price
            ),
            max_price
        )
        return p_order

    def get_power_quantity(self) -> float:
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
        current_temp = self.device_parameters["current_temp"]
        max_temp = self.device_parameters["max_temp"]
        power_rating = self.device_parameters["power_rating"]
        q_order = 0
        if current_temp < max_temp:
            q_order = power_rating
        return q_order


class HVACOrder(Order):

    def __init__(self, auction_id):
        super().__init__(DeviceType.HVAC, auction_id)

    def get_heat_price(self, deadband: float = 0) -> float:
        """
        Calculate the power price for heating mode.

        Parameters
        ----------
        deadband : float, optional
            Temperature deadband, defaults to 0

        Returns
        -------
        float
            The calculated power price
        """
        current_temp = self.device_parameters["current_temp"]
        min_temp = self.device_parameters["min_temp"]
        max_temp = self.device_parameters["max_temp"]
        desired_temp = self.customer_parameters["desired_temp"]
        expected_price = self.market_parameters["expected_price"]
        expected_stdev = self.market_parameters["expected_stdev"]
        min_price = self.market_parameters["min_price"]
        max_price = self.market_parameters["max_price"]
        K_hvac = self.customer_parameters["K_hvac"]

        print(f"\nHeating Price Calculation:")
        print(f"  - Current Temp: {current_temp:.2f}°F")
        print(f"  - Min Temp: {min_temp:.2f}°F")
        print(f"  - Desired Temp: {desired_temp:.2f}°F")
        print(f"  - Deadband: {deadband:.2f}°F")

        if current_temp < min_temp:
            print("  - Current temp below min temp - Using max price")
            return max_price
        elif current_temp > max_temp:
            print("  - Current temp above max temp - Price set to 0")
            return 0
        else:
            # when the min temp equals the desired temp, we set the price to the min price
            if min_temp == (desired_temp - deadband):
                print("  - Min temp equals desired temp - Using min price")
                return min_price
            else:
                temp_ratio = ((desired_temp - deadband) - current_temp) ** 2 / abs(min_temp - (desired_temp - deadband))
                # because the temp ratio is squared, we need to make sure it is negative if the current temp is greater than the desired temp
                if current_temp > (desired_temp - deadband):
                    temp_ratio = -temp_ratio
                price_adjustment = K_hvac * expected_stdev * temp_ratio
                final_price = min(max(expected_price + price_adjustment, min_price), max_price)
                
                print(f"  - Temperature Ratio: {temp_ratio:.4f}")
                print(f"  - Price Adjustment: ${price_adjustment:.2f}")
                print(f"  - Base Price: ${expected_price:.2f}")
                print(f"  - Final Price: ${final_price:.2f}")
                return final_price

    def get_cool_price(self, deadband: float = 0) -> float:
        """
        Calculate the power price for cooling mode.

        Parameters
        ----------
        deadband : float, optional
            Temperature deadband, defaults to 0

        Returns
        -------
        float
            The calculated power price
        """
        current_temp = self.device_parameters["current_temp"]
        min_temp = self.device_parameters["min_temp"]
        max_temp = self.device_parameters["max_temp"]
        desired_temp = self.customer_parameters["desired_temp"]
        expected_price = self.market_parameters["expected_price"]
        expected_stdev = self.market_parameters["expected_stdev"]
        min_price = self.market_parameters["min_price"]
        max_price = self.market_parameters["max_price"]
        K_hvac = self.customer_parameters["K_hvac"]

        print(f"\nCooling Price Calculation:")
        print(f"  - Current Temp: {current_temp:.2f}°F")
        print(f"  - Max Temp: {max_temp:.2f}°F")
        print(f"  - Desired Temp: {desired_temp:.2f}°F")
        print(f"  - Deadband: {deadband:.2f}°F")

        if current_temp < min_temp:
            print("  - Current temp below min temp - Price set to 0")
            return 0
        elif current_temp > max_temp:
            print("  - Current temp above max temp - Using max price")
            return max_price
        else:
            # when the max temp equals the desired temp, we set the price to the min price
            if max_temp == (desired_temp + deadband):
                print("  - Max temp equals desired temp - Using min price")
                return min_price
            else:
                temp_ratio = ((current_temp - (desired_temp + deadband)) ** 2 / abs(max_temp - (desired_temp + deadband)))
                # because the temp ratio is squared, we need to make sure it is negative if the current temp is less than the desired temp
                if current_temp < (desired_temp + deadband):
                    temp_ratio = -temp_ratio
                price_adjustment = K_hvac * expected_stdev * temp_ratio
                final_price = min(max(expected_price + price_adjustment, min_price), max_price)
                
                print(f"  - Temperature Ratio: {temp_ratio:.4f}")
                print(f"  - Price Adjustment: ${price_adjustment:.2f}")
                print(f"  - Base Price: ${expected_price:.2f}")
                print(f"  - Final Price: ${final_price:.2f}")
                return final_price

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
        expected_stdev = self.market_parameters["expected_stdev"]
        expected_price = self.market_parameters["expected_price"]
        # Customer parameters
        desired_temp = self.customer_parameters["desired_temp"]
        # Device parameters
        current_temp = self.device_parameters["current_temp"]
        last_change_temp = self.device_parameters["last_change_temp"]
        mode = self.device_parameters["mode"]
        auto_mode_dead_band = 1 if mode == 'auto' else 0

        print(f"\nTemperature Parameters:")
        print(f"  - Current Temperature: {current_temp:.2f}°F")
        print(f"  - Desired Temperature: {desired_temp:.2f}°F")
        print(f"  - Last Temperature Change: {last_change_temp:.2f}°F")
        print(f"  - Mode: {mode}")
        print(f"  - Auto Mode Deadband: {auto_mode_dead_band}°F")
        print(f"\nMarket Parameters:")
        print(f"  - Expected Price: ${expected_price:.2f}")
        print(f"  - Expected Standard Deviation: ${expected_stdev:.2f}")

        if expected_stdev == 0:
            expected_stdev = expected_price
            print(f"  - Adjusted Standard Deviation: ${expected_stdev:.2f}")

        p_order = 0
        if mode == "heating":
            p_order = self.get_heat_price()
            print(f"\nHeating Mode - Final Price: ${p_order:.2f}")

        elif mode == "cooling":
            p_order = self.get_cool_price()
            print(f"\nCooling Mode - Final Price: ${p_order:.2f}")

        elif mode == "auto":
            print(f"\nAuto Mode Analysis:")
            print(f"  - Temperature Range: {desired_temp - auto_mode_dead_band:.2f}°F to {desired_temp + auto_mode_dead_band:.2f}°F")
            # if the current temp lies within the deadband, we set the price to zero
            if (desired_temp - auto_mode_dead_band) < current_temp < (desired_temp + auto_mode_dead_band):
                p_order = 0
                print("  - Temperature within deadband - Price set to $0.00")
            elif last_change_temp > 0:  # Heating mode
                p_order = self.get_heat_price(deadband=auto_mode_dead_band)
                print(f"  - Heating required - Final Price: ${p_order:.2f}")
            elif last_change_temp < 0:  # Cooling mode
                p_order = self.get_cool_price(deadband=auto_mode_dead_band)
                print(f"  - Cooling required - Final Price: ${p_order:.2f}")

        print(f"\nFinal Bid Price: ${p_order:.2f}")
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
        # Device parameters returns the power rating from the meter
        power_rating = self.device_parameters["power_rating"]
        mode = self.device_parameters["mode"]
        last_change_temp = self.device_parameters["last_change_temp"]
        fan_state = self.device_parameters["fan_state"]
        q_order = 0
        # add auxiliary power either for heating or seperate
        q_fan = power_rating["fan"] if fan_state == "on" else 0
        # add auxiliary power either for heating or seperate
        if mode == "heat" or (mode == "auto" and last_change_temp > 0):
            q_order = power_rating["heat"]
        elif mode == "cool" or (mode == "auto" and last_change_temp < 0):
            q_order = power_rating["cool"]
        return q_order + q_fan


class BatteryOrder(Order):

    def __init__(self, auction_id):
        super().__init__(DeviceType.BATTERY, auction_id)

    def get_buy_price(self) -> float:
        """
        Calculate the buy price for the battery heuristic strategy.
        """
        expected_price = self.market_parameters["expected_price"]
        expected_stdev = self.market_parameters["expected_stdev"]
        max_price = self.market_parameters["max_price"]
        min_price = self.market_parameters["min_price"]
        soc = self.device_parameters["state_of_charge"]
        flexibility = self.customer_parameters["K_battery"]

        if soc == 1:
            return 0

        buy_price = expected_price * (
            -flexibility * expected_stdev * (soc) ** 2 / 10 + 1.1
        )
        print(
            f"Buy price: {buy_price}, expected price: {expected_price}, expected stdev: {expected_stdev}, flexibility: {flexibility}, soc: {soc}"
        )
        buy_price = min(max(buy_price, min_price), max_price)

        return buy_price

    def get_sell_price(self) -> float:
        """
        Calculate the sell price for the battery heuristic strategy.
        """
        expected_price = self.market_parameters["expected_price"]
        expected_stdev = self.market_parameters["expected_stdev"]
        min_price = self.market_parameters["min_price"]
        max_price = self.market_parameters["max_price"]
        flexibility = self.customer_parameters["K_battery"]  # [0-1]
        soc = self.device_parameters["state_of_charge"]  # [0-1]

        if soc == 0:
            return 0

        sell_price = expected_price * (
            flexibility * expected_stdev * (1 - soc) ** 2 + 1.1
        )
        print(
            f"Sell price: {sell_price}, expected price: {expected_price}, expected stdev: {expected_stdev}, flexibility: {flexibility}, soc: {soc}"
        )
        sell_price = min(max(sell_price, min_price), max_price)

        return sell_price

    def get_power_price(self) -> float: # pragma: no cover
        """
        DEPRECATED: This method is deprecated. Use get_buy_price and get_sell_price instead.
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
        expected_stdev = self.market_parameters["expected_stdev"]
        expected_price = self.market_parameters["expected_price"]
        min_price = self.market_parameters["min_price"]
        max_price = self.market_parameters["max_price"]
        # Customer parameters
        K_battery = self.customer_parameters["K_battery"]
        desired_state_of_charge = self.customer_parameters[
            "desired_state_of_charge"
        ]  # desired state of charge in percentage
        # Device parameters
        state_of_charge = self.device_parameters[
            "state_of_charge"
        ]  # current state of charge in percentage
        max_state_of_charge = self.device_parameters[
            "max_state_of_charge"
        ]  # maximum state of charge in percentage
        min_state_of_charge = self.device_parameters[
            "min_state_of_charge"
        ]  # minimum state of charge in percentage

        dead_band = (max_state_of_charge - min_state_of_charge) * 0.05

        p_order = 0
        if expected_stdev == 0:
            expected_stdev = expected_price * 0.001
        if state_of_charge <= min_state_of_charge:
            p_order = max_price
        elif state_of_charge >= max_state_of_charge:
            p_order = min_price
        elif (
            min_state_of_charge < state_of_charge < desired_state_of_charge - dead_band
        ):
            cdf = min(
                max(
                    1
                    - (state_of_charge - min_state_of_charge)
                    / (desired_state_of_charge - dead_band - min_state_of_charge),
                    0,
                ),
                1,
            )
            p_order = NormalDist(
                mu=expected_price, sigma=K_battery * expected_stdev
            ).inv_cdf(cdf)
        elif (
            desired_state_of_charge + dead_band < state_of_charge < max_state_of_charge
        ):
            cdf = min(
                max(
                    1
                    - (state_of_charge - (desired_state_of_charge + dead_band))
                    / (max_state_of_charge - (desired_state_of_charge + dead_band)),
                    0,
                ),
                1,
            )
            p_order = NormalDist(
                mu=expected_price, sigma=K_battery * expected_stdev
            ).inv_cdf(cdf)
        # Include the RTE to handle
        # Implement deadband 1% so that it is not always SOC
        p_order = min(max(p_order, min_price), max_price)
        return p_order

    def get_buy_quantity(self) -> float:
        """
        Calculate the power quantity for the battery heuristic strategy.
        """
        soc = self.device_parameters["state_of_charge"]
        if soc == 1:
            return 0
        return self.device_parameters["battery_rating"]

    def get_sell_quantity(self) -> float:
        """
        Calculate the power quantity for the battery heuristic strategy.
        """
        soc = self.device_parameters["state_of_charge"]
        if soc == 0:
            return 0
        return -self.device_parameters["battery_rating"]

    def get_power_quantity(self) -> float: # pragma: no cover
        """
        DEPRECATED: This method is deprecated. Use get_buy_quantity and get_sell_quantity instead.
        Calculate the power quantity for the battery heuristic strategy.
        """
        return self.device_parameters["battery_rating"]
