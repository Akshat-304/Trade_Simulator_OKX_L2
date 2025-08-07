import logging
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class OrderBookManager:
    """
    Manages the L2 order book data (asks and bids) received from the WebSocket.
    Provides methods to update and query the book.
    """

    def __init__(self):
        self.asks: List[Tuple[float, float]] = []  # List of (price, quantity) tuples
        self.bids: List[Tuple[float, float]] = []  # List of (price, quantity) tuples
        self.timestamp: str = ""
        self.symbol: str = ""
        self.exchange: str = ""
        logger.info("OrderBookManager initialized.")

    def update_book(self, data: Dict[str, Any]):
        """
        Updates the order book with new data from the WebSocket.
        The data is expected to be a full snapshot of the order book.
        """
        try:
            self.timestamp = data.get("timestamp", "")
            self.symbol = data.get("symbol", "")
            self.exchange = data.get("exchange", "")

            # Process asks: convert strings to floats, sort by price ascending
            raw_asks = data.get("asks", [])
            self.asks = sorted(
                [(float(price), float(quantity)) for price, quantity in raw_asks],
                key=lambda x: x[0],  # Sort by price (first element)
            )

            # Process bids: convert strings to floats, sort by price descending
            raw_bids = data.get("bids", [])
            self.bids = sorted(
                [(float(price), float(quantity)) for price, quantity in raw_bids],
                key=lambda x: x[0],  # Sort by price (first element)
                reverse=True,
            )
            # logger.debug(f"Order book updated for {self.symbol} @ {self.timestamp}. "
            #              f"Asks: {len(self.asks)}, Bids: {len(self.bids)}")
        except KeyError as e:
            logger.error(f"KeyError while updating order book: {e} in data {data}")
        except ValueError as e:
            logger.error(
                f"ValueError (likely float conversion) while updating order book: {e} in data {data}"
            )
        except Exception as e:
            logger.error(f"Unexpected error updating order book: {e} in data {data}")

    def get_best_ask(self) -> Tuple[float, float] | None:
        """Returns the best (lowest) ask price and its quantity."""
        return self.asks[0] if self.asks else None

    def get_best_bid(self) -> Tuple[float, float] | None:
        """Returns the best (highest) bid price and its quantity."""
        return self.bids[0] if self.bids else None

    def get_spread(self) -> float | None:
        """Calculates the spread between best ask and best bid."""
        best_ask = self.get_best_ask()
        best_bid = self.get_best_bid()
        if best_ask and best_bid:
            return best_ask[0] - best_bid[0]
        return None

    def __str__(self):
        best_ask_str = (
            f"Best Ask: {self.get_best_ask()}" if self.asks else "Best Ask: N/A"
        )
        best_bid_str = (
            f"Best Bid: {self.get_best_bid()}" if self.bids else "Best Bid: N/A"
        )
        spread_str = (
            f"Spread: {self.get_spread()}"
            if self.get_spread() is not None
            else "Spread: N/A"
        )
        return (
            f"OrderBook ({self.exchange} - {self.symbol} @ {self.timestamp}):\n"
            f"  {best_ask_str}\n"
            f"  {best_bid_str}\n"
            f"  {spread_str}\n"
            f"  Total Asks: {len(self.asks)}, Total Bids: {len(self.bids)}"
        )
