"""TESS Auction

The TESS auction computes the price and quantity at which supply equals demand
at a constrained device in a distribution system.  

Construction
------------

The auction constructor can be used with no arguments to create a new auction.
An auction can be constructor from existing data using the following
arguments:

    buyers - provides a list of buy orders

    sellers - provices a list of sell orders

    history - provides a clearing history

    clearing - provides the current clearing

    interval - sets the 

    unresponsive_demand - sets the load that is not responsive to price
    (default is 0.0)

    units - sets the units of quantity (default is "unit")

    price_ceiling - sets the maximum price allowed (default is 1000.0)

    price_floor - sets the minimum price allowed (default is 0.0)

Orders
------

Orders contain information about a devices willingness to produce (seller) or
consume (buyer) at a particular price.  Orders contain the following
information:

    order_id - a unique id number assigned to the order by the auction when it
    is submitted

    quantity - the quantity in units that is submitted, which is positive for
    buy orders and negative for sell orders.

    price - the price at which the order will be cleared

    flexible - specifies whether the device is flexible when operating as
    marginal.

Clearing
--------

When the market is cleared, all buyers with a price greater and all sellers with
a price lower than the clearing price will be dispatched at the order quantity.
If the order price is the same, the order is marginal and the dispatched
quantity will be less than or equal order quantity. The clearing data includes
the following:

    price - the price at which the auction cleared

    quantity - the total quantity dispatched by the auction when it cleared

    marginal_type - the order type of the marginal order, i.e., 'buyer'
    or 'seller'
    
    marginal_order - the marginal order data

    marginal_quantity - the marginal quantity dispatched

    marginal_rank - the rank of the marginal order in the merit order of the
    dispatch

History
-------

The history data is a time-series of auction clearing price and quantity.

"""
import os, sys
import datetime
import logging
from math import floor, log10
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# import geometry classes from the same directory
from hvac_simulation.geometry import Point, Line, Curve

def calculate_units_precision(
    cost=0.01, # $
    quantity=0.001, # MW
    interval=300, # seconds
    ):
    """Calculate the precision of units used in TESS auctions """
    interval = interval / 3600 # hours
    cost_decimal = -floor(log10(cost))
    quantity_decimal = -floor(log10(quantity))
    time_decimal = -floor(log10(interval))
    price_decimal = cost_decimal + quantity_decimal + time_decimal + 3
    price_precision = round(cost/quantity/interval, price_decimal) # $/MW.h
    return price_precision

############################################################################################################
#
# Auction classes
#
############################################################################################################

class Tess2AuctionException(Exception):
    """This Auction exception class for handling errors in the auction class"""

class Tess2Auction:
    """
    The Tess2Auction class implements a two-sided auction

    Attributes:    
            buyers (DataFrame) - the buyer data (order_id, price, quantity, flexible)    
            sellers (DataFrame) - the seller data (order_id, price, quantity, flexible)    
            history (DataFrame) - the clearing history data (datetime, price, quantity)    
            clearing (dict) - the latest clearing data (datetime, price, quantity, etc.)    
            interval (int) - the market time interval    
            unresponsive_demand (float) - the unresponsive demand quantity    
            units (str) - the units of quantity    
            price_ceiling (float) - the maximum allowed price    
            price_floor (float) - the minimum allowed price    
            minimum_quantity (float) - the minimum allowed quantity    
            guid_order (int) - the order of the guid    
            clear_timeout (float) - the time allowed for clearing the auction
    Methods:
            clear - clear the auction
            submit - submit an order to the auction
            withdraw - withdraw an order from the auction
            amend - amend an order in the auction
            get_cleared_price - get the most recent clearing price
            get_cleared_quantity - get the most recent clearing quantity
            get_marginal_type - get the most recent marginal order type
            get_marginal_order - get the most recent marginal order
            get_marginal_quantity - get the most recent marginal quantity
            get_marginal_rank - get the most recent marginal order rank
            get_clearing_timer - get the time required to complete the most recent clearing
            get_order - get an order by order_id
            get_dispatch - get the dispatch quantity by order_id
            get_buyer - get a buyer order by order_id
            get_seller - get a seller order by order_id
            guid - generate a guid
            random - generate random orders
    """

    def __init__(self,
        buyers = None,
        sellers = None,
        history = None,
        clearing = None,
        interval = 300,
        unresponsive_demand = 0.0,
        units = 'unit',
        price_ceiling = 1000.0,
        price_floor = 0.0,
        minimum_quantity = 0.001,
        guid_order = 12,
        clear_timeout = 1.0,
        ):
        """Construct an auction object

        Arguments:

            buyers (DataFrame or dict) - the buyer data (order_id, price, quantity, flexible)

            sellers (DataFrame or dict) - the seller data (order_id, price, quantity, flexible)

            history (DataFrame) - the clearing history data (datetime, price, quantity)

            clearing (dict) - the latest clearing data (datetime, price, quantity, etc.)

            interval (int) - the market time interval

            unresponsive_demand (float) - the unresponsive demand quantity

            units (str) - the units of quantity

            price_ceiling (float) - the maximum allowed price

            price_floor (float) - the minimum allowed price
        """

        if isinstance(buyers, (pd.DataFrame, type(None))):
            self.buyers = buyers
        elif isinstance(buyers, dict):
            self.buyers = pd.DataFrame(buyers)
            self.buyers.set_index('order_id', inplace=True)
        else:
            raise Tess2AuctionException(f"buyers type '{type(buyers)}' is invalid")
        if buyers is not None:
            for column in ["order_id","price","quantity"]:
                if column not in buyers.columns:
                    raise Tess2AuctionException(f"buyers missing required columns '{column}'")
            if "flexible" not in buyers.columns:
                logging.warning("buyers missing flexible data -- assume flexible=True")
                buyers["flexible"] = True

        if isinstance(sellers, (pd.DataFrame, type(None))):
            self.sellers = sellers
        elif isinstance(sellers, dict):
            self.sellers = pd.DataFrame(sellers)
            self.sellers.set_index('order_id',inplace=True)
        else:
            raise Tess2AuctionException("sellers type is invalid")
        if sellers is not None:
            for column in ["order_id","price","quantity"]:
                if column not in sellers.columns:
                    raise Tess2AuctionException(f"sellers missing required columns '{column}'")
            if "flexible" not in sellers.columns:
                logging.warning("sellers missing flexible data -- assume flexible=True")
                sellers["flexible"] = True

        if isinstance(history, (pd.DataFrame, type(None))):
            self.history = history
        else:
            raise Tess2AuctionException(f"history type '{type(history)}' is invalid")

        if isinstance(clearing, (dict, type(None))):
            self.clearing = clearing
        else:
            raise Tess2AuctionException(f"clearing type '{type(clearing)}' is invalid")

        assert(isinstance(interval, int) and interval > 0)
        self.interval = interval

        assert(isinstance(unresponsive_demand, float) and unresponsive_demand >= 0)
        self.unresponsive_demand = unresponsive_demand

        assert(isinstance(units, str))
        self.units = units

        assert(isinstance(price_ceiling, float))
        self.price_ceiling = price_ceiling

        assert(isinstance(price_floor, float))
        self.price_floor = price_floor

        assert(price_ceiling > price_floor)

        assert(isinstance(minimum_quantity, float) and minimum_quantity > 0.0)
        self.minimum_quantity = minimum_quantity

        assert(isinstance(guid_order, int) and guid_order > 1)
        self.guid_order = guid_order

        self.fig = None

    def __del__(self):
        if self.fig: # check if the figure is none
            if self.fig == plt.gcf():
                plt.close()

    def clear(self, update=True):
        """Clear auction

        Arguments:

            update (bool) - update the auction data with results if successful

        Returns:

            dict - price, quantity
        """

        # timer
        tic = datetime.datetime.now().timestamp()

        # setup buyers
        if self.buyers is not None and len(self.buyers) > 0:

            buyers = self.buyers.copy()

            # check inputs
            bad_buyers = buyers[buyers.quantity<=0]
            if len(bad_buyers) > 0:
                bad_buyers = ','.join(sorted(bad_buyers.index))
                raise Tess2AuctionException(f"{len(bad_buyers)} invalid quantity for buyer ids: {bad_buyers}")

            # prepare buyers
            if "dispatch" not in buyers.columns or update:
                buyers.sort_values(by="price",ascending=False,inplace=True)
                buyers["dispatch"] = buyers.quantity.cumsum() + self.unresponsive_demand
            logging.debug(f"{len(buyers)} buyers, total quantity={buyers.dispatch.iloc[0]:.1f} {self.units}")
        else:
            buyers = None

        # setup sellers
        if self.sellers is not None and len(self.sellers) > 0:

            sellers = self.sellers.copy()

            # check inputs
            bad_sellers = sellers[sellers.quantity<=0]
            if len(bad_sellers) > 0:
                bad_sellers = ','.join(sorted(bad_sellers.index))
                raise Tess2AuctionException(f"{len(bad_sellers)} invalid quantity for seller ids: {bad_sellers}")

            # prepare sellers
            if "dispatch" not in sellers.columns or update:
                sellers.sort_values(by="price",ascending=True,inplace=True)
                sellers["dispatch"] = sellers.quantity.cumsum()
            logging.debug(f"{len(sellers)} sellers, total quantity={sellers.dispatch.iloc[-1]:.1f} {self.units}")
        else:
            sellers = None

        if buyers is None:
            if sellers is None:
                result = {
                    "price": 0.0, 
                    "quantity": 0.0,
                    "marginal_type": None, 
                    "marginal_quantity": None, 
                    "marginal_order": None, 
                    "marginal_rank": None
                }
            else:
                result = {
                    "price": sellers.price.iloc[0], 
                    "quantity": 0.0,
                    "marginal_type": 'seller', 
                    "marginal_quantity": 0.0, 
                    "marginal_order": sellers.index[0], 
                    "marginal_rank": 0
                }
        elif sellers is None:
            result = {
                "price": buyers.price.iloc[0], 
                "quantity": 0.0,
                "marginal_type": 'buyer', 
                "marginal_quantity": 0.0, 
                "marginal_order": buyers.index[0], 
                "marginal_rank": 0
            }
        else: # clearing is possible
            buys = [Point(0,self.price_ceiling)]
            buy_index = []
            buy_rank = []
            rank = 0
            if self.unresponsive_demand > 0:
                buys.append(Point(self.unresponsive_demand,self.price_ceiling))
                buy_index.append(None)
                buy_rank.append(None)
            for n,buy in buyers.iterrows():
                if buy.price != buys[-1].y:
                    buys.append(Point(buys[-1].x,buy.price))
                    buy_index.append(None)
                    buy_rank.append(None)
                    rank += 1
                else:
                    rank = 0
                if buy.dispatch != buys[-1].x:
                    buys.append(Point(buy.dispatch,buy.price))
                    buy_index.append(n)
                    buy_rank.append(rank)
                buy_rank.append(rank)
            buys.append(Point(buy.dispatch,self.price_floor))
            buy_index.append(None)
            buy_rank.append(None)
            buys = Curve(buys)

            sells = [Point(0,self.price_floor)]
            sell_index = []
            sell_rank = []
            rank = 0
            for m,sell in sellers.iterrows():
                if sell.price != sells[-1].y:
                    sells.append(Point(sells[-1].x,sell.price))
                    sell_index.append(None)
                    sell_rank.append(None)
                    rank += 1
                else:
                    rank = 0
                if sell.dispatch != sells[-1].x:
                    sells.append(Point(sell.dispatch,sell.price))
                    sell_index.append(m)
                    sell_rank.append(rank)
            sells.append(Point(sell.dispatch,self.price_ceiling))
            sell_index.append(None)
            sell_rank.append(None)
            sells = Curve(sells)
            clearing = sells.intersect(buys,locate=True)
            if clearing is None:
                result,seller,buyer = None,None,None
            else:
                if isinstance(clearing[0][0],Point):
                    result,seller,buyer = clearing[0][0], clearing[0][1], clearing[0][2]
                elif isinstance(clearing[0][0],Line):
                    result,seller,buyer = clearing[0][0].reduce_point(), clearing[0][1], clearing[0][2]

            # TODO: calculate marginal quantities, types and ranks
            marginal_type = None if seller is None and buyer is None else 'seller' if buyer is None else 'buyer'
            marginal_unit = None if seller is None and buyer is None else sell_index[seller] if buyer is None else buy_index[buyer]
            marginal_rank = None if seller is None and buyer is None else sell_rank[seller] if buyer is None else buy_rank[buyer]
            marginal_quantity = None


            result = {
                "quantity": 0.0 if result is None else result.x,
                "price": 0.0 if result is None else result.y,
                "marginal_type": marginal_type,
                "marginal_unit": marginal_unit,
                "marginal_quantity": marginal_quantity,
                "marginal_rank": marginal_rank,
                "marginal_order": None
            }


        if update:
            self.buyers = buyers
            self.sellers = sellers

        toc = datetime.datetime.now().timestamp()
        result['clearing_timer'] = round(toc-tic,6)

        if update:
            self.clearing = result

        return result

    # ask john if the bad order should be accepted with arguments
    def submit(self,price,quantity, flexible=True):
        """Submit an order to the auction
        Arguments:
            price (float)
            quantity (float)
            flexible (bool)
        Returns:
            str - the order id
        Exception:
            Tess2AuctionException - order was not accepted
        """
        if self.clearing is not None:
            raise Tess2AuctionException("market has already cleared")
        if abs(quantity) < self.minimum_quantity:
            raise Tess2AuctionException("quantity is less than minimum_quantity")
        if price < self.price_floor or price > self.price_ceiling:
            raise Tess2AuctionException("price is out of bounds") # TODO: Ignore the orders that are out of bounds or any orders that breaks

        guid = self.guid(1)[0]
        order = pd.DataFrame({"price":[price], "quantity":[abs(quantity)]},index=[guid])
        if quantity < 0.0:
            if isinstance(self.sellers, pd.DataFrame):
                self.sellers = pd.concat([self.sellers,order])
            else:
                self.sellers = order
        elif price is None:
            self.unresponsive_demand += quantity
            return None
        else:
            if isinstance(self.buyers, pd.DataFrame):
                self.buyers = pd.concat([self.buyers,order])
            else:
                self.buyers = order
        return guid
    
    def withdraw(self,order_id):
        """Withdraw an order
        Arguments:
            order_id (str) - the order id
        Exception:
            Tess2AuctionException - order id does not exist
        """
        if self.clearing != None:
            raise Tess2AuctionException("market has already cleared")

        raise Exception("not implemented yet")

    def amend(self,order_id,name,value):
        """Change an order
        Arguments:
            order_id
            name - order property to change, i.e., `price`, `quantity` or `flexible`
            value - value to change
        Returns:
            varies - previous value
        Exceptions:
            Tess2AuctionException - order id does not exist or data type does not match
        """
        if self.clearing != None:
            raise Tess2AuctionException("market has already cleared")

        raise Exception("not implemented yet")

    def get_cleared_price(self):
        """Get the most recent clearing price
        Returns:
            float - the price in $/unit.h
            float('nan') - the auction is degenerate (no price)
            None - the auction has not cleared yet
        """
        return self.clearing['price'] if isinstance(self.clearing, dict) else None

    def get_cleared_quantity(self):
        """Get the most recent clearing quantity
        Returns:
            float - the quantity in units
            float('nan') - the auction is degenerate (no quantity)
            None - the auction has not cleared yet
        """
        return self.clearing['quantity'] if isinstance(self.clearing, dict) else None

    def get_marginal_type(self):
        """Get the most recent marginal order type
        Returns:
            str - the order type, i.e., 'buyer' or 'seller'
            None - The auction has not cleared yet
        """
        return self.clearing['marginal_type'] if isinstance(self.clearing, dict) else None

    def get_marginal_order(self):
        """Get the most recent marginal order
        Returns:
            dict - marginal order data
            None - the marginal has not cleared yet
        """
        return self.clearing['marginal_order'] if isinstance(self.clearing, dict) else None

    def get_marginal_quantity(self):
        """Get the most recent marginal quantity cleared
        Returns:
            float - the quantity in units
            None - the marginal has not cleared yet
        """
        return self.clearing['marginal_quantity'] if isinstance(self.clearing, dict) else None

    def get_marginal_rank(self):
        """Get the most recent marginal order rank cleared
        Returns:
            int - the order rank as an index into the order list
            None - the marginal has not cleared yet
        """ 
        return self.clearing['marginal_rank'] if isinstance(self.clearing, dict) else None

    def get_clearing_timer(self):
        """Get the time required to complete the most recent clearing
        Returns:
            float - the elapsed time taken to clear the auction in seconds
        """
        return self.clearing['clearing_timer'] if isinstance(self.clearing, dict) else None

    def get_order(self,order_id):
        """TODO
        Arguments:
        Returns:
        Exceptions:
        """
        try:
            return 'buyer',self.get_buyer(order_id)
        except:
            pass
        try:
            return 'seller',self.get_seller(order_id)
        except:
            return None,None

    def get_dispatch(self,order_id):
        """TODO
        Arguments:
        Returns:
        Exceptions:
        """
        side,order = self.get_order(order_id)
        return {
            "side": side, 
            "quantity" : order.quantity
        } if side in ['buyer','seller'] else {}

    def get_buyer(self,order_id):
        """TODO
        Arguments:
        Returns:
        Exceptions:
        """
        if isinstance(order_id, str):
            return self.buyers.loc[order_id]
        elif isinstance(order_id, int):
            return self.buyers.iloc[order_id]
        else:
            raise Tess2AuctionException("invalid order_id")

    def get_seller(self,order_id):
        """TODO
        Arguments:
        Returns:
        Exceptions:
        """
        if isinstance(order_id, str):
            return self.sellers.loc[order_id]
        elif isinstance(order_id, int):
            return self.sellers.iloc[order_id]
        else:
            raise Tess2AuctionException("invalid order_id")

    def guid(self,N=1,Z=None):
        """Generate a guid

        Arguments:            
                N (int) - the number of guids to generate    
                Z (int) - the number of characters in the guid
        Returns:            
                list of str - guids
        """
        result = ""
        if Z is None:
            Z = self.guid_order
        if Z <= 6:
            return [hex(x)[2:] for x in np.random.randint(2**((2**Z)-2),2**((2**Z)-1),N, dtype=np.int64)]

        prefix = self.guid(N,Z-6)
        body = self.guid(N,6)
        result = []
        for n in range(N):
            result.append(prefix[n]+body[n])
        return result

    def random(self,N,
            price_distribution = np.random.normal,
            price_arguments = (50,10),
            price_limits = (0,1000), # min, max $/MW.h
            quantity_distribution = np.random.normal,
            quantity_arguments = (0,10),
            quantity_limits = (-0.02,0.02), # min, max MW
            solar_capacity = 1.0, # MW
            unresponsive_load = 2.0, # MW
            feeder_capacity = 1.0, # MW
            feeder_fractional_rank = 0.15, # per unit total supply available
            mininum_quantity = 0.001, # MW
            auction_interval = 300, # seconds
            history_length = 24*7*12, # 1 week of five minute clearings (0 for no history)
            history_now = datetime.datetime.now().timestamp() % 300,
            update = True,
            ):
        """Generate a random set of orders

        Arguments:

            price_distribution (callable) - the generating function for prices
            (default np.random.normal)

            price_arguments (float,float) - the price distribution arguments
            (default 50,10) 

            price_limits (float,float) - the price clipping limits
            (default 0,1000)

            quantity_distribution (callable) - the generating function for
            quantities (default np.random.normal)

            quantity_arguments (float,float) - the quantity distribution
            arguments (default 0,10)

            quantity_limits (float,float) - the quantity distribution clipping
            limits (default -20,20)

            solar_capacity (float) - the capacity of the solar generation units
            (default 1.0 MW)

            unresponsive_load (float) - the unreponsive load (default 2.0 MW)

            feeder_capacity (float) - the feeder capacity (default 1.0)

            feeder_fractional_rank (float) - the rank of feeder as a fraction of
            the total supply (default 0.15)

            mininum_quantity (float) - the minimum quantity ask/offer allowed
            (default 0.001 MW),

            history_length (float) - the length of the clearing history
            (default 1 week)
            
            history_now = datetime.datetime.now().timestamp() % 300

            update (bool) - apply the data to this auction (default True)

        Returns:

            pandas.DataFrame - buyer offers
            
            pandas.DataFrame - seller asks

            pandas.DataFrame - clearing history
        """
        
        self.unresponsive_demand = unresponsive_load
        P = price_distribution(*price_arguments,N).clip(*price_limits)
        Q = quantity_distribution(*quantity_arguments,N).clip(*quantity_limits)
        B = pd.DataFrame(
            data = dict(price=P.round(4),quantity=Q.round()),
            index = self.guid(N))

        buy = B[Q>0].sort_values(by="price",ascending=False)
        buy.loc[buy["quantity"]<mininum_quantity, "quantity"] = mininum_quantity
        
        sell = B[Q<0].sort_values(by="price",ascending=True)
        sell.loc[sell.index[0], "price"] = 0.0
        sell.loc[sell.index[0], "quantity"] = -solar_capacity # solar
        sell.loc[sell.index[int(N*feeder_fractional_rank)], "quantity"] = -feeder_capacity # feeder
        sell.loc[sell["quantity"] > -mininum_quantity, "quantity"] = -mininum_quantity
        sell["quantity"] = -sell.quantity

        if history_length > 0:
            P = np.hstack([
                    np.random.normal(P.mean(),P.std()/3,288).clip(P.min(),P.max()),
                    np.random.normal(P.mean(),P.std(),history_length-288).clip(P.min(),P.max()),
                    ])
            Q = np.hstack([
                    np.random.normal(Q.cumsum().mean(),Q.cumsum().std()/2,288),
                    np.random.normal(Q.cumsum().mean(),Q.cumsum().std(),history_length-288),
                    ])
            history = pd.DataFrame(
                data = dict(price=P.round(4),quantity=Q.round(3)+unresponsive_load),
                columns=['price','quantity'],
                )
        else:
            history = None

        if update:
            self.buyers = buy
            self.sellers = sell
            self.history = history

        return buy, sell, history

    def plot(self,
            saveas = None,
            show = False,
            quantity_zoom = None,
            price_zoom = None):
        """Plot an auction

        Arguments:

        Returns:

            matplotlib.pyplot.figure - 
        """
        self.fig = plt.figure(figsize=(10,7))

        if isinstance(self.history, pd.DataFrame):
            plt.plot(self.history.quantity.to_list(),self.history.price.to_list(),".m",markersize=0.5,label="Past week")
            plt.plot(self.history.quantity[:int(86400/self.interval)].to_list(),self.history.price[:int(86400/self.interval)].to_list(),".k",markersize=1,label="Past day")

        # plot supply curve
        P0 = 0
        Q0 = 0
        for order_id, order in [] if self.sellers is None else self.sellers.iterrows():
            Q1 = Q0 + order['quantity']
            P1 = order['price']
            plt.plot([Q0,Q0,Q1],[P0,P1,P1],"-r",linewidth=2.5)
            Q0 = Q1
            P0 = P1
        plt.plot([Q0,Q0],[P0,self.price_ceiling],"-r",linewidth=2.5,label="Supply")

        # plot demand curve
        P0 = self.price_ceiling
        Q0 = self.unresponsive_demand
        plt.plot([0,Q0],[P0,P0],"-b")
        for order_id,order in [] if self.buyers is None else self.buyers.iterrows():
            Q1 = Q0 + order['quantity']
            P1 = order['price']
            plt.plot([Q0,Q0,Q1],[P0,P1,P1],"-b")
            Q0 = Q1
            P0 = P1
        plt.plot([Q0,Q0],[P0,self.price_floor],"-b",label="Demand")

        if isinstance(self.clearing, dict):
            plt.plot(self.clearing["quantity"],self.clearing["price"],'*k',markersize=8,label="Clearing")

        plt.grid()
        plt.xlabel(f"Quantity [{self.units}]")
        plt.ylabel(f"Price [$/{self.units}.h]")
        plt.legend()

        if isinstance(quantity_zoom, float) and isinstance(self.clearing, dict):
            plt.xlim([self.clearing["quantity"]-quantity_zoom/2,self.clearing["quantity"]+quantity_zoom/2])
        elif isinstance(quantity_zoom, list):
            plt.xlim(*quantity_zoom)

        if isinstance(price_zoom, float) and isinstance(self.clearing, dict):
            plt.ylim([self.clearing["price"]-price_zoom/2,self.clearing["price"]+price_zoom/2])
        elif isinstance(price_zoom, list):
            plt.ylim(*price_zoom)
        
        if saveas:
            plt.savefig(saveas)
        if show:
            plt.show()

        return plt.gcf()

############################################################################################################
# 
# Validation tests
#
############################################################################################################

if __name__ == "__main__":

    import unittest
    import inspect
    def fname(fmt=None):
        if fmt:
            return fmt.format(fname=inspect.stack()[1][3])        
        return inspect.stack()[1][3]

    os.makedirs("test",exist_ok=True)

    class TestAuction(unittest.TestCase):

        def test_empty(self):
            auction = Tess2Auction(units='MW')
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(auction.get_marginal_order(),None)
            self.assertEqual(auction.get_order('nosuch'),(None,None))
            self.assertEqual(auction.get_dispatch('nosuch'),{})
            self.assertGreaterEqual(result["clearing_timer"],0.0)
            self.assertEqual(result["price"],0)
            self.assertEqual(result["quantity"],0)
            # #self.assertEqual(result["marginal_type"],None)
            # #self.assertEqual(result["marginal_rank"],None)

        def test_nodemand(self):
            auction = Tess2Auction(units='MW')
            auction.submit(quantity=-0.01,price=10.0)
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(result["price"],10)
            self.assertEqual(result["quantity"],0)
            # #self.assertEqual(result["marginal_type"],"seller")
            # #self.assertEqual(result["marginal_rank"],0)

        def test_nosupply(self):
            auction = Tess2Auction(units='MW')
            auction.submit(quantity=0.01,price=10.0)
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(result["price"],10)
            self.assertEqual(result["quantity"],0)
            # #self.assertEqual(result["marginal_type"],"buyer")
            # #self.assertEqual(result["marginal_rank"],0)

        # # # ########################################################################################
        
        def test_s1d2_1(self):
            auction = Tess2Auction(units='MW')
            auction.submit(quantity=-0.005,price=20.0)
            auction.submit(quantity=0.01,price=30.0)
            auction.submit(quantity=0.005,price=10.0)
            auction.plot(saveas=fname("test/{fname}.png"))
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(result["price"],30.0)
            self.assertEqual(result["quantity"],0.005)
            # #self.assertEqual(result["marginal_type"],"buyer")
            # #self.assertEqual(result["marginal_rank"], 1)

        def test_s1d2_2(self):
            auction = Tess2Auction(units='MW')
            auction.submit(quantity=-0.01,price=20.0)
            auction.submit(quantity=0.01,price=30.0)
            auction.submit(quantity=0.005,price=10.0)
            auction.plot(saveas=fname("test/{fname}.png"))
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(result["price"],25.0)
            self.assertEqual(result["quantity"],0.01)
            # #self.assertEqual(result["marginal_type"],None)
            # #self.assertEqual(result["marginal_rank"],None)

        def test_s1d2_3(self):
            auction = Tess2Auction(units='MW')
            auction.submit(quantity=-0.015,price=20.0)
            auction.submit(quantity=0.01,price=30.0)
            auction.submit(quantity=0.005,price=10.0)
            auction.plot(saveas=fname("test/{fname}.png"))
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(result["price"],20.0)
            self.assertEqual(result["quantity"],0.01)
            # #self.assertEqual(result["marginal_type"],"seller")
            # #self.assertEqual(result["marginal_rank"],0)

        def test_s1d2_4(self):
            auction = Tess2Auction(units='MW')
            auction.submit(quantity=-0.005,price=10.0)
            auction.submit(quantity=0.01,price=30.0)
            auction.submit(quantity=0.005,price=10.0)
            auction.plot(saveas=fname("test/{fname}.png"))
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(result["price"],30.0)
            self.assertEqual(result["quantity"],0.005)
            # #self.assertEqual(result["marginal_type"],"buyer")
            # #self.assertEqual(result["marginal_rank"],0)

        def test_s1d2_5(self):
            auction = Tess2Auction(units='MW')
            auction.submit(quantity=-0.01,price=10.0)
            auction.submit(quantity=0.01,price=30.0)
            auction.submit(quantity=0.005,price=10.0)
            auction.plot(saveas=fname("test/{fname}.png"))
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(result["price"],20.0)
            self.assertEqual(result["quantity"],0.01)
            # #self.assertEqual(result["marginal_type"],None)
            # #self.assertEqual(result["marginal_rank"],None)

        def test_s1d2_6(self):
            auction = Tess2Auction(units='MW')
            auction.submit(quantity=-0.0125,price=10.0)
            auction.submit(quantity=0.01,price=30.0)
            auction.submit(quantity=0.005,price=10.0)
            auction.plot(saveas=fname("test/{fname}.png"))
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(result["price"],10.0)
            self.assertEqual(result["quantity"],0.0125)
            # #self.assertEqual(result["marginal_type"],'buyer')
            # #self.assertEqual(result["marginal_rank"],1)

        def test_s1d2_7(self):
            auction = Tess2Auction(units='MW')
            auction.submit(quantity=-0.015,price=10.0)
            auction.submit(quantity=0.01,price=30.0)
            auction.submit(quantity=0.005,price=10.0)
            auction.plot(saveas=fname("test/{fname}.png"))
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(result["price"],10.0)
            self.assertEqual(result["quantity"],0.015)
            # #self.assertEqual(result["marginal_type"],'buyer')
            # #self.assertEqual(result["marginal_rank"],1)

        def test_s1d2_8(self):
            auction = Tess2Auction(units='MW')
            auction.submit(quantity=-0.02,price=10.0)
            auction.submit(quantity=0.01,price=30.0)
            auction.submit(quantity=0.005,price=10.0)
            auction.plot(saveas=fname("test/{fname}.png"))
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(result["price"],10.0)
            self.assertEqual(result["quantity"],0.015)
            # #self.assertEqual(result["marginal_type"],'seller')
            # #self.assertEqual(result["marginal_rank"],0)

        def test_s1d2_9(self):
            auction = Tess2Auction(units='MW')
            auction.submit(quantity=-0.005,price=5.0)
            auction.submit(quantity=0.01,price=30.0)
            auction.submit(quantity=0.005,price=10.0)
            auction.plot(saveas=fname("test/{fname}.png"))
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(result["price"],30.0)
            self.assertEqual(result["quantity"],0.005)
            # #self.assertEqual(result["marginal_type"],"buyer")
            # #self.assertEqual(result["marginal_rank"],0)

        def test_s1d2_10(self):
            auction = Tess2Auction(units='MW')
            auction.submit(quantity=-0.01,price=5.0)
            auction.submit(quantity=0.01,price=30.0)
            auction.submit(quantity=0.005,price=10.0)
            auction.plot(saveas=fname("test/{fname}.png"))
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(result["price"],20.0)
            self.assertEqual(result["quantity"],0.01)
            # #self.assertEqual(result["marginal_type"],None)
            # #self.assertEqual(result["marginal_rank"],None)

        def test_s1d2_11(self):
            auction = Tess2Auction(units='MW')
            auction.submit(quantity=-0.0125,price=5.0)
            auction.submit(quantity=0.01,price=30.0)
            auction.submit(quantity=0.005,price=10.0)
            auction.plot(saveas=fname("test/{fname}.png"))
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(result["price"],10.0)
            self.assertEqual(result["quantity"],0.0125)
            # #self.assertEqual(result["marginal_type"],'buyer')
            # #self.assertEqual(result["marginal_rank"],1)

        def test_s1d2_12(self):
            auction = Tess2Auction(units='MW')
            auction.submit(quantity=-0.015,price=5.0)
            auction.submit(quantity=0.01,price=30.0)
            auction.submit(quantity=0.005,price=10.0)
            auction.plot(saveas=fname("test/{fname}.png"))
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(result["price"],7.5)
            self.assertEqual(result["quantity"],0.015)
            # #self.assertEqual(result["marginal_type"],None)
            # #self.assertEqual(result["marginal_rank"],None)

        def test_s1d2_13(self):
            auction = Tess2Auction(units='MW')
            auction.submit(quantity=-0.02,price=5.0)
            auction.submit(quantity=0.01,price=30.0)
            auction.submit(quantity=0.005,price=10.0)
            auction.plot(saveas=fname("test/{fname}.png"))
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(result["price"],5.0)
            self.assertEqual(result["quantity"],0.015)
            # #self.assertEqual(result["marginal_type"],'seller')
            # #self.assertEqual(result["marginal_rank"],0)

        def test_s1d2_14(self):
            auction = Tess2Auction(units='MW')
            auction.submit(quantity=-0.005,price=30.0)
            auction.submit(quantity=0.01,price=30.0)
            auction.submit(quantity=0.005,price=10.0)
            auction.plot(saveas=fname("test/{fname}.png"))
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(result["price"],30.0)
            self.assertEqual(result["quantity"],0.005)
            # #self.assertEqual(result["marginal_type"],"buyer")
            # #self.assertEqual(result["marginal_rank"],0)

        def test_s1d2_15(self):
            auction = Tess2Auction(units='MW')
            auction.submit(quantity=-0.01,price=30.0)
            auction.submit(quantity=0.01,price=30.0)
            auction.submit(quantity=0.005,price=10.0)
            auction.plot(saveas=fname("test/{fname}.png"))
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(result["price"],30.0)
            self.assertEqual(result["quantity"],0.01)
            # #self.assertEqual(result["marginal_type"],None)
            # #self.assertEqual(result["marginal_rank"],None)

        def test_s1d2_16(self):
            auction = Tess2Auction(units='MW')
            auction.submit(quantity=-0.015,price=30.0)
            auction.submit(quantity=0.01,price=30.0)
            auction.submit(quantity=0.005,price=10.0)
            auction.plot(saveas=fname("test/{fname}.png"))
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(result["price"],30.0)
            self.assertEqual(result["quantity"],0.01)
            #self.assertEqual(result["marginal_type"],"seller")
            #self.assertEqual(result["marginal_rank"],0)

        def test_s1d2_17(self):
            auction = Tess2Auction(units='MW')
            auction.submit(quantity=-0.005,price=40.0)
            auction.submit(quantity=0.01,price=30.0)
            auction.submit(quantity=0.005,price=10.0)
            auction.plot(saveas=fname("test/{fname}.png"))
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(result["price"],35.0)
            self.assertEqual(result["quantity"],0.0)
            #self.assertEqual(result["marginal_type"],None)
            #self.assertEqual(result["marginal_rank"],None)

        ########################################################################################

        def test_s2d1_1(self):
            auction = Tess2Auction(units='MW')
            auction.submit(quantity=-0.005,price=10.0)
            auction.submit(quantity=-0.01,price=30.0)
            auction.submit(quantity=0.01,price=20.0)
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(result["price"],20.0)
            self.assertEqual(result["quantity"],0.005)
            #self.assertEqual(result["marginal_type"],"buyer")
            #self.assertEqual(result["marginal_rank"],0)

        def test_s2d1_2(self):
            auction = Tess2Auction(units='MW')
            auction.submit(quantity=-0.01,price=10.0)
            auction.submit(quantity=-0.01,price=30.0)
            auction.submit(quantity=0.01,price=20.0)
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(result["price"],15.0)
            self.assertEqual(result["quantity"],0.01)
            #self.assertEqual(result["marginal_type"],None)
            #self.assertEqual(result["marginal_rank"],None)

        def test_s2d1_3(self):
            auction = Tess2Auction(units='MW')
            auction.submit(quantity=-0.015,price=10.0)
            auction.submit(quantity=-0.01,price=30.0)
            auction.submit(quantity=0.01,price=20.0)
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(result["price"],10.0)
            self.assertEqual(result["quantity"],0.01)
            #self.assertEqual(result["marginal_type"],"seller")
            #self.assertEqual(result["marginal_rank"],0)

        def test_s2d1_4(self):
            auction = Tess2Auction(units='MW')
            auction.submit(quantity=-0.005,price=10.0)
            auction.submit(quantity=-0.01,price=15.0)
            auction.submit(quantity=0.01,price=20.0)
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(result["price"],15.0)
            self.assertEqual(result["quantity"],0.01)
            #self.assertEqual(result["marginal_type"],"seller")
            #self.assertEqual(result["marginal_rank"],1)

        def test_s2d1_5(self):
            auction = Tess2Auction(units='MW')
            auction.submit(quantity=-0.01,price=10.0)
            auction.submit(quantity=-0.01,price=15.0)
            auction.submit(quantity=0.01,price=20.0)
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(result["price"],12.5)
            self.assertEqual(result["quantity"],0.01)
            #self.assertEqual(result["marginal_type"],None)
            #self.assertEqual(result["marginal_rank"],None)

        def test_s2d1_6(self):
            auction = Tess2Auction(units='MW')
            auction.submit(quantity=-0.015,price=10.0)
            auction.submit(quantity=-0.01,price=15.0)
            auction.submit(quantity=0.01,price=20.0)
            result = auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])
            self.assertEqual(result["price"],10.0)
            self.assertEqual(result["quantity"],0.01)
            #self.assertEqual(result["marginal_type"],'seller')
            #self.assertEqual(result["marginal_rank"],0)

        ########################################################################################

        def test_clearing(self):
            np.random.seed(3)
            auction = Tess2Auction(units='MW')
            auction.random(10,unresponsive_load=2)
            result = auction.clear()
            print("buyer",auction.buyers)
            print("seller",auction.sellers)
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[45,55],quantity_zoom=[2,2.0025])
            self.assertEqual(round(result["price"], 4), 50.2634)
            self.assertEqual(result["quantity"],2.002)
            #self.assertEqual(result["marginal_type"],'buyer')
            #self.assertEqual(result["marginal_rank"],4)
            # self.assertEqual(result["marginal_unit"],2)
            self.assertEqual(auction.get_cleared_price(),result["price"])
            self.assertEqual(auction.get_cleared_quantity(),result["quantity"])
            self.assertEqual(auction.get_marginal_type(),result["marginal_type"])
            self.assertEqual(auction.get_marginal_order(),result["marginal_order"])
            self.assertEqual(auction.get_marginal_quantity(),result["marginal_quantity"])
            self.assertEqual(auction.get_marginal_rank(),result["marginal_rank"])
            order_id = auction.get_marginal_order()
            side,order = auction.get_order(order_id)
            # self.assertGreaterEqual(order.quantity,auction.get_marginal_quantity())

        def test_history(self):
            auction = Tess2Auction(units='MWh')
            auction.random(1000,unresponsive_load=1.5)
            auction.clear()
            auction.plot(saveas=fname("test/{fname}.png"),price_zoom=[0,50],quantity_zoom=[0,0.1])

        def test_guid_len(self):
            auction = Tess2Auction(units='MW')
            guid = auction.guid(2,12)
            self.assertEqual(len(guid[0]),32)

    unittest.main()
