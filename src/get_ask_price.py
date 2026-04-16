# %%
import threading
import time

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper


class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def tickPrice(self, reqId, tickType, price, attrib):
        if tickType == 2 and reqId == 1:
            print("The current ask price is: ", price)


app = IBapi()
print(app.connect( 7497, 58))

# Start the socket in a daemon thread so that it will automatically close when the main program ends
api_thread = threading.Thread(target=app.run, daemon=True)
api_thread.start()

time.sleep(1)  # Sleep interval to allow time for connection to server
# %%
# Create contract object
apple_contract = Contract()
apple_contract.symbol = "AAPL"
apple_contract.secType = "STK"
apple_contract.exchange = "SMART"
apple_contract.currency = "USD"

# Request Market Data
app.reqMktData(1, apple_contract, "", False, False, [])
# %%
time.sleep(10)  # Sleep interval to allow time for incoming price data
app.disconnect()
