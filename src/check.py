# %%
import time

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from rich import print


# %%
class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)


# %%
app = IBapi()
# %%
app.connect(host="127.0.0.1", port=4002, clientId=123)
print(app.isConnected())
# %%
# app.run()
# %%
# Uncomment this section if unable to connect
# and to prevent errors on a reconnect
for k in range(4):
    print(f"Waiting for connection... {k + 1}/10")
    time.sleep(1)

app.disconnect()
print(app.isConnected())
