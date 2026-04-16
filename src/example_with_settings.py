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
            host=self.settings.ib_host,
            port=self.settings.ib_port,
            clientId=self.settings.ib_client_id,
        )

    def tickPrice(self, reqId: int, tickType: int, price: float, attrib: dict):
        if tickType == 2 and reqId == 1:
            print(
                f"The current ask price for {self.settings.contract_symbol} is: {price}"
            )


def main():
    """Connect to IB and request market data using settings."""

    app = IBapi()

    # Connect using settings
    print(f"Connecting to {app.settings.ib_host}:{app.settings.ib_port}...")
    print(f"Debug mode: {app.settings.debug}")

    app.connect()
    if app.isConnected():
        print("connected :-)")
    else:
        print("Connection failed")
        return

    app.reqMarketDataType(3)

    # Start the socket in a daemon thread
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    time.sleep(1)  # Allow time for connection

    # Create contract using settings
    contract = Contract()
    contract.symbol = "AAPL"
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"

    # Request Market Data
    print(f"Requesting market data for {contract}...")
    app.reqMktData(
        reqId=1,
        contract=contract,
        genericTickList="",
        snapshot=False,
        regulatorySnapshot=False,
        mktDataOptions=[],
    )

    # Wait for data
    time.sleep(app.settings.timeout)

    app.disconnect()
    print("Disconnected")


if __name__ == "__main__":
    main()
