"""Example of using Pydantic settings with IBapi."""

import threading
import time

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper

from settings import AppSettings


class IBapi(EWrapper, EClient):
    def __init__(self) -> None:
        EClient.__init__(self, self)
        self.settings = AppSettings()

    def connect(self):
        return super().connect(
            host=self.settings.ib.host,
            port=self.settings.ib.port,
            clientId=self.settings.ib.client_id,
        )

    def tickPrice(self, reqId: int, tickType: int, price: float, attrib: dict):
        if tickType == 2 and reqId == 1:
            print(
                f"The current ask price for {self.settings.contract.symbol} is: {price}"
            )


def main():
    """Connect to IB and request market data using settings."""

    app = IBapi()

    # Connect using settings
    print(f"Connecting to {app.settings.ib.host}:{app.settings.ib.port}...")
    print(f"Debug mode: {app.settings.debug}")

    connection_successful = app.connect()

    if not connection_successful:
        print("Failed to connect to IB Gateway/TWS")
        return

    # Start the socket in a daemon thread
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    time.sleep(1)  # Allow time for connection

    # Create contract using settings
    contract = Contract()
    contract.symbol = app.settings.contract.symbol
    contract.secType = app.settings.contract.sec_type
    contract.exchange = app.settings.contract.exchange
    contract.currency = app.settings.contract.currency

    # Request Market Data
    print(f"Requesting market data for {contract.symbol}...")
    app.reqMktData(1, contract, "", False, False, [])

    # Wait for data
    time.sleep(app.settings.timeout)

    app.disconnect()
    print("Disconnected")


if __name__ == "__main__":
    main()
