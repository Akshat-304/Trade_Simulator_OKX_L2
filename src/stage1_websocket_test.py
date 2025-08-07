import asyncio
import websockets
import json
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# WebSocket URL provided by GoQuant
WEBSOCKET_URL = "wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP"


async def connect_and_listen():
    """
    Connects to the WebSocket server and listens for messages.
    """
    logging.info(f"Attempting to connect to WebSocket: {WEBSOCKET_URL}")
    try:
        async with websockets.connect(WEBSOCKET_URL) as websocket:
            logging.info("Successfully connected to WebSocket.")
            logging.info("Listening for L2 order book data...")

            async for message in websocket:
                try:
                    data = json.loads(message)
                    # For now, just print the received data structure
                    # In later stages, we'll process this data
                    logging.info(
                        f"Received data: {data['exchange']} {data['symbol']} @ {data['timestamp']}"
                    )
                    # Optionally print more details or the whole data structure
                    # print(json.dumps(data, indent=2))
                except json.JSONDecodeError:
                    logging.error(f"Could not decode JSON: {message}")
                except KeyError as e:
                    logging.error(
                        f"Missing key in received data: {e} - Data: {message}"
                    )
                except Exception as e:
                    logging.error(f"Error processing message: {e} - Data: {message}")

    except websockets.exceptions.ConnectionClosedError as e:
        logging.error(f"WebSocket connection closed unexpectedly: {e}")
    except websockets.exceptions.InvalidURI:
        logging.error(f"Invalid WebSocket URI: {WEBSOCKET_URL}")
    except ConnectionRefusedError:
        logging.error(
            f"Connection refused. Ensure the server is running and accessible (VPN?)."
        )
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        logging.info("WebSocket connection closed or attempt failed.")


if __name__ == "__main__":
    try:
        asyncio.run(connect_and_listen())
    except KeyboardInterrupt:
        logging.info("Program interrupted by user. Exiting.")

# correct output :
# (.venv) aarya_gupta@ShreyG Google-GoQuant % python -u "/Users/aarya_gupta/Github_Projects/Google-GoQuant/src/stage1_websocket_test.py"
# 2025-05-20 02:09:42,655 - INFO - Attempting to connect to WebSocket: wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP
# 2025-05-20 02:09:43,227 - INFO - Successfully connected to WebSocket.
# 2025-05-20 02:09:43,227 - INFO - Listening for L2 order book data...
# 2025-05-20 02:09:43,275 - INFO - Received data: OKX BTC-USDT-SWAP @ 2025-05-19T20:39:43Z
# 2025-05-20 02:09:43,378 - INFO - Received data: OKX BTC-USDT-SWAP @ 2025-05-19T20:39:43Z
# 2025-05-20 02:09:43,476 - INFO - Received data: OKX BTC-USDT-SWAP @ 2025-05-19T20:39:43Z
# 2025-05-20 02:09:43,575 - INFO - Received data: OKX BTC-USDT-SWAP @ 2025-05-19T20:39:43Z
# 2025-05-20 02:09:43,676 - INFO - Received data: OKX BTC-USDT-SWAP @ 2025-05-19T20:39:43Z
# 2025-05-20 02:09:43,776 - INFO - Received data: OKX BTC-USDT-SWAP @ 2025-05-19T20:39:43Z
# 2025-05-20 02:09:43,875 - INFO - Received data: OKX BTC-USDT-SWAP @ 2025-05-19T20:39:43Z
# 2025-05-20 02:09:43,976 - INFO - Received data: OKX BTC-USDT-SWAP @ 2025-05-19T20:39:43Z
# 2025-05-20 02:09:44,076 - INFO - Received data: OKX BTC-USDT-SWAP @ 2025-05-19T20:39:43Z
# 2025-05-20 02:09:44,188 - INFO - Received data: OKX BTC-USDT-SWAP @ 2025-05-19T20:39:44Z
# 2025-05-20 02:09:44,276 - INFO - Received data: OKX BTC-USDT-SWAP @ 2025-05-19T20:39:44Z
# ^C2025-05-20 02:09:44,463 - INFO - WebSocket connection closed or attempt failed.
# 2025-05-20 02:09:44,464 - INFO - Program interrupted by user. Exiting.
