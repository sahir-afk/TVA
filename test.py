"""RH-02 <-> ESC1 CAN echo test.

Sends 0x123 [DE AD BE EF] at 10 Hz and prints anything echoed back.
Success = a steady stream of  RX id=0x7FF data=de ad be ef  at 10 Hz.

Smoke test first (proves the USB plumbing):
    python -c "import can; b=can.Bus(interface='gs_usb',channel=0,index=0,bitrate=1_000_000); print('bus open')"
"""
import can
import time

bus = can.Bus(interface="gs_usb", channel=0, index=0, bitrate=1_000_000)
print("bus open")

tx = can.Message(
    arbitration_id=0x123,
    data=[0xDE, 0xAD, 0xBE, 0xEF],
    is_extended_id=False,
)

last = 0.0
try:
    while True:
        now = time.time()
        if now - last >= 0.1:  # 10 Hz
            try:
                bus.send(tx)
            except can.CanError as e:
                print("TX failed:", e)  # no ACK -> nothing on the bus is listening
            last = now

        rx = bus.recv(timeout=0.01)
        if rx is not None:
            print(f"RX id=0x{rx.arbitration_id:03X} data={rx.data.hex(' ')}")
except KeyboardInterrupt:
    pass
finally:
    bus.shutdown()
