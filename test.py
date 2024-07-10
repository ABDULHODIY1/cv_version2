from snap7 import util, client
from snap7.util import *
import snap7

def mWriteBool(plcObj, byte, bit, value):
    data = plcObj.read_area(snap7.types.Areas.MK, 0, byte, 1)
    set_bool(data, 0, bit, value)
    plcObj.write_area(snap7.types.Areas.MK, 0, byte, data)

plc = client.Client()
plc.set_connection_type(3)
plc.connect("192.168.1.11", 0, 1)
state = plc.get_connected()
mWriteBool(plc,5,0, True)
print(state)

