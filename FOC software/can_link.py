"""CAN link layer.

Wraps python-can: one RX thread, dispatch by CAN ID, typed send helpers
that go through protocol.py so nothing else in the codebase touches struct.

Bus selection is the ONLY thing that changes between sandbox and hardware:
  sandbox (cross-process, no hardware):  --iface udp_multicast  (default)
  sandbox (single process):              --iface virtual
  real adapter later:                    --iface gs_usb / slcan / socketcan

Run this file directly to get a bus spy (checkpoint 2):
  python can_link.py
"""
from __future__ import annotations

import threading
import time
from typing import Callable

import can

import protocol as P
import sys
if sys.platform == "darwin":
    import usb.core
    usb.core.Device.is_kernel_driver_active = lambda self, i: False

    
# default sandbox bus: udp multicast loops back on localhost, works across
# separate terminals/processes with zero hardware and zero config
DEFAULT_IFACE = "udp_multicast"
DEFAULT_CHANNEL = "239.74.163.2:43113"
BITRATE = 1_000_000  # meaningless on virtual buses, required on real ones

HEARTBEAT_TIMEOUT_S = 0.050  # 5 missed beats at 100 Hz


def add_bus_args(parser) -> None:
    parser.add_argument("--iface", default=DEFAULT_IFACE)
    parser.add_argument("--channel", default=DEFAULT_CHANNEL)


class CanLink:
    def __init__(self, iface: str = DEFAULT_IFACE, channel: str = DEFAULT_CHANNEL,
                 name: str = "node") -> None:
        self.name = name
        self._bus = can.Bus(interface=iface, channel=channel, bitrate=BITRATE,
                            receive_own_messages=False)
        self._by_id: dict[int, list[Callable[[can.Message], None]]] = {}
        self._any: list[Callable[[can.Message], None]] = []
        self._alive = True
        self._rx = threading.Thread(target=self._rx_loop, daemon=True,
                                    name=f"can-rx-{name}")
        self._rx.start()

    # ---- subscriptions ----------------------------------------------------
    def on(self, can_id: int, handler: Callable[[can.Message], None]) -> None:
        self._by_id.setdefault(int(can_id), []).append(handler)

    def on_any(self, handler: Callable[[can.Message], None]) -> None:
        self._any.append(handler)

    def _rx_loop(self) -> None:
        while self._alive:
            try:
                msg = self._bus.recv(timeout=0.2)
            except can.CanError:
                continue
            if msg is None:
                continue
            for h in self._any + self._by_id.get(msg.arbitration_id, []):
                try:
                    h(msg)
                except Exception as e:  # a bad frame must not kill the thread
                    print(f"[{self.name}] handler error on 0x{msg.arbitration_id:03X}: {e}")

    # ---- raw + typed send -------------------------------------------------
    def send(self, can_id: int, data: bytes) -> None:
        self._bus.send(can.Message(arbitration_id=int(can_id), data=data,
                                   is_extended_id=False))

    def send_heartbeat(self, counter: int, armed: bool) -> None:
        self.send(P.CanId.Heartbeat, P.pack_heartbeat(counter, armed))

    def send_command(self, mode: P.Mode, target: float, enable: bool) -> None:
        self.send(P.CanId.COMMAND, P.pack_command(mode, target, enable))

    def send_pid(self, loop: P.Loop, kp: float, ki: float, kd: float) -> None:
        self.send(P.CanId.PID_GAINS, P.pack_pid(loop, kp, ki, kd))

    def send_telem_fast(self, t: P.FastTelemetry) -> None:
        self.send(P.CanId.TELEM_FAST, P.pack_telem_fast(t))

    def send_telem_slow(self, t: P.SlowTelemetry) -> None:
        self.send(P.CanId.TELEM_SLOW, P.pack_telem_slow(t))

    def shutdown(self) -> None:
        self._alive = False
        self._rx.join(timeout=1.0)
        self._bus.shutdown()


class HeartbeatMonitor:
    """Node-side fail-safe timer. beat() on every heartbeat frame; ok() is the
    gate the control loop checks. Mirrors what the firmware will do."""

    def __init__(self, timeout_s: float = HEARTBEAT_TIMEOUT_S) -> None:
        self._timeout = timeout_s
        self._last: float | None = None

    def beat(self) -> None:
        self._last = time.monotonic()

    def ok(self) -> bool:
        return self._last is not None and (time.monotonic() - self._last) < self._timeout


# ---- checkpoint 2: bus spy ------------------------------------------------

def _decode(msg: can.Message) -> str:
    i, d = msg.arbitration_id, bytes(msg.data)
    try:
        if i == P.CanId.Heartbeat:
            c, armed = P.unpack_heartbeat(d)
            return f"HEARTBEAT  n={c} armed={armed}"
        if i == P.CanId.COMMAND:
            mode, target, en = P.unpack_command(d)
            return f"COMMAND    {mode.name} target={target:.1f} enable={en}"
        if i == P.CanId.PID_GAINS:
            loop, kp, ki, kd = P.unpack_pid(d)
            return f"PID        {loop.name} kp={kp} ki={ki} kd={kd}"
        if i == P.CanId.TELEM_FAST:
            t = P.unpack_telem_fast(d)
            return (f"TELEM_FAST iq={t.iq_ma:.0f}mA id={t.id_ma:.0f}mA "
                    f"ang={t.angle_deg:.2f} vel={t.velocity_dps:.0f}dps")
        if i == P.CanId.TELEM_SLOW:
            t = P.unpack_telem_slow(d)
            return (f"TELEM_SLOW vbus={t.vbus_v:.2f}V temp={t.temp_c:.1f}C "
                    f"status={t.status!r} loop={t.loop_us}us")
    except Exception as e:
        return f"?? decode error: {e}  raw={d.hex()}"
    return f"?? unknown id  raw={d.hex()}"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="CAN bus spy (checkpoint 2)")
    add_bus_args(ap)
    args = ap.parse_args()

    link = CanLink(args.iface, args.channel, name="spy")
    print(f"spying on {args.iface}:{args.channel} — ctrl-c to stop")
    link.on_any(lambda m: print(f"0x{m.arbitration_id:03X}  {_decode(m)}"))
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        link.shutdown()