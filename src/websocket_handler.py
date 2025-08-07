import asyncio
import websockets
import json
import logging
import time

logger = logging.getLogger(__name__)

WEBSOCKET_URL = "wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP"


async def connect_and_listen(book_manager, ui_update_callback=None):
    """
    Connects to the WebSocket server, listens for messages,
    updates the OrderBookManager, and calls the UI update callback.
    """
    websocket_client = None  # Define here to ensure it's in scope for finally
    logger.info(f"Attempting to connect to WebSocket: {WEBSOCKET_URL}")
    connection_established = False
    try:
        async with websockets.connect(WEBSOCKET_URL, ping_interval=None) as ws:
            websocket_client = ws  # Assign to outer scope variable
            connection_established = True
            logger.info("Successfully connected to WebSocket.")
            if ui_update_callback:
                ui_update_callback(
                    book_manager, "connected", None
                )  # Initial connected status

            logger.info("Listening for L2 order book data...")

            async for message in websocket_client:
                # --- START: L1 (WS Message to Book Update) Latency Measurement ---
                ws_msg_arrival_time = (
                    time.perf_counter()
                )  # Mark the moment the message is available

                try:
                    data = json.loads(message)
                    book_manager.update_book(data)
                    # --- END: L1 Latency Measurement ---
                    if ui_update_callback:
                        # Pass status and ws_msg_arrival_time as separate arguments
                        ui_update_callback(
                            book_manager, "data_update", ws_msg_arrival_time
                        )
                except json.JSONDecodeError:
                    logger.error(f"Could not decode JSON: {message}")
                    if (
                        ui_update_callback
                    ):  # Ensure callback is still made on error if needed, without latency
                        ui_update_callback(book_manager, "data_error", None)
                except Exception as e:
                    logger.error(
                        f"Error processing message in connect_and_listen: {e} - Data: {message}",
                        exc_info=True,
                    )  # Added exc_info
                    if ui_update_callback:
                        ui_update_callback(book_manager, "data_error", None)

    except websockets.exceptions.ConnectionClosed as e:  # More specific catch
        logger.error(f"WebSocket connection closed: {e.reason} (Code: {e.code})")
        if ui_update_callback:
            ui_update_callback(book_manager, "disconnected_error", None)
    except websockets.exceptions.InvalidURI:
        logger.error(f"Invalid WebSocket URI: {WEBSOCKET_URL}")
        if ui_update_callback:
            ui_update_callback(book_manager, "disconnected_error", None)
    except ConnectionRefusedError:
        logger.error(
            f"Connection refused. Ensure the server is running and accessible (VPN?)."
        )
        if ui_update_callback:
            ui_update_callback(book_manager, "disconnected_error", None)
    except (
        Exception
    ) as e:  # Catch other potential errors during connection or listening
        logger.error(
            f"An unexpected error occurred during WebSocket operation: {e}",
            exc_info=True,
        )  # Added exc_info
        if ui_update_callback:
            ui_update_callback(book_manager, "disconnected_error", None)
    finally:
        logger.info("WebSocket connection process finished or attempt failed.")
        if ui_update_callback:
            # Determine if it was a clean disconnect or error
            is_clean_disconnect = False
            if (
                connection_established
                and websocket_client
                and websocket_client.close_code in [1000, 1001]
            ):
                is_clean_disconnect = True
            if is_clean_disconnect:
                ui_update_callback(book_manager, "disconnected_clean", None)
            elif connection_established:  # if it was connected but closed uncleanly
                ui_update_callback(book_manager, "disconnected_error", None)
            # If it never connected, "disconnected_error" would have been called by exception handlers.
            # If it was connected but the callback for error was not called above, call it now.
            elif not connection_established and ui_update_callback:
                pass  # ui_update_callback(book_manager, "disconnected_error", None) - likely already called


if __name__ == "__main__":
    logger.info("websocket_handler.py should be run as part of main_app.py")
