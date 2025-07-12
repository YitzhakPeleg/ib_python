# %%
import time
from ibapi.client import EClient
from ibapi.wrapper import EWrapper


# %%
class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)


# %%
app = IBapi()
app.connect('host.docker.internal', 7497, 123)
# %%
app.run()
# %%
# Uncomment this section if unable to connect
# and to prevent errors on a reconnect


time.sleep(5)
app.disconnect()
