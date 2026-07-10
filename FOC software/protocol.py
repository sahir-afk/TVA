from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum, IntFlag


# Canid enum which defines different types of data packet labels and their priorities
class CanId(IntEnum):
# lower hex means higher priority in arbitration, telemetry is up one place to leave more room for potential additional vital info
    Heartbeat = 0x010
    COMMAND = 0x020
    PID_GAINS = 0x030
    TELEM_FAST = 0X100
    TELEM_SLOW = 0X200

class Mode(IntEnum):
    DISABLE = 0
    TORQUE = 1      #target iq in mA 
    VELOCITY = 2    #target velocity in degrees per second
    POSITION = 3    #target angle, it will be mutliplied by 10 to save the bit space needed for floating point numbers

class Loop(IntEnum):
    CURRENT = 0
    VELOCITY = 1
    POSITION = 2


# flags are binary codes representing different states/statuses
class CmdFlags(IntFlag):
    ENABLE = 1 << 0


class StatusFlags(IntFlag):
    ENABLED = 1 << 0
    FAULT = 1 << 1
    HEARTBEAT_OK = 1 << 2
    OVERCURRENT = 1 << 3

# scales for each values staying within 8 byte limit while maximizing precision
POSITION_SCALE = 10.0     # degrees   -> int16  (0.1 deg resolution, +/-3276 deg)
ANGLE_SCALE = 100.0       # degrees   -> uint16 (0.01 deg, 0..360 -> 0..36000)
CURRENT_SCALE = 1.0       # mA        -> int16  (+/-32 A)
VELOCITY_SCALE = 1.0      # deg/s     -> int16
GAIN_SCALE = 1000.0       # gain      -> uint16 (0..65.535)
VBUS_SCALE = 1000.0       # volts     -> uint16 mV (0..65.5 V)
TEMP_SCALE = 100.0        # deg C     -> int16 centi-deg

# for the sliders, if the person adjusts to a value above or below the limits it caps at the max/min vals
def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v

# scales the data packet being SENT depending on the Mode, e.g. position needs to be scaled by 10, but Torque and Velocity are only 1
def target_to_wire(mode: Mode, target: float) -> int:
    scale = POSITION_SCALE if mode == Mode.POSITION else 1.0
    return int(round(_clamp(target * scale, -32768, 32767)))

# scales the data packet being RECEIVED depending on the mode, it divides now because it is reversing the scaling done by the motor microcontroller
def target_from_wire(mode: Mode, raw: int) -> float:
    scale = POSITION_SCALE if mode == Mode.POSITION else 1.0
    return raw / scale


# packs the CAN heartbeat message, makes sure it is the right size
def pack_heartbeat(counter: int, armed: bool) -> bytes:
    return struct.pack("<IB", counter & 0xFFFFFFFF, 1 if armed else 0)

# unpacks the CAN hearbeat message, returns counter and whether it is armed or not
def unpack_heartbeat(data: bytes) -> tuple[int, bool]:
    counter, flags = struct.unpack("<IB", bytes(data[:5]))
    return counter, bool(flags & 1)


# pack/unpack commands, one byte for flags, one byte for mode, two bytes for the actual data

def pack_command(mode: Mode, target: float, enable: bool) -> bytes:
    flags = CmdFlags.ENABLE if enable else CmdFlags(0)
    return struct.pack("<BBh", int(mode), int(flags), target_to_wire(mode, target))


# order matters, unpack reads mode before calling target_from_wire to scale the incoming data
def unpack_command(data: bytes) -> tuple[Mode, float, bool]:
    mode_raw, flags, target_raw = struct.unpack("<BBh", bytes(data[:4]))
    mode = Mode(mode_raw)
    return mode, target_from_wire(mode, target_raw), bool(flags & CmdFlags.ENABLE)


# pack/unpack pid gains, CAN messages are capped at 8 bytes so floats arent an option, gains must be used

def pack_pid(loop: Loop, kp: float, ki: float, kd: float) -> bytes:
    def g(x: float) -> int:
        return int(round(_clamp(x * GAIN_SCALE, 0, 65535)))
    return struct.pack("<BHHH", int(loop), g(kp), g(ki), g(kd))


def unpack_pid(data: bytes) -> tuple[Loop, float, float, float]:
    loop_raw, kp, ki, kd = struct.unpack("<BHHH", bytes(data[:7]))
    return Loop(loop_raw), kp / GAIN_SCALE, ki / GAIN_SCALE, kd / GAIN_SCALE



# data class for telem, direct and quadrature current, angle, and angular velo
@dataclass
class FastTelemetry:
    iq_ma: float
    id_ma: float
    angle_deg: float
    velocity_dps: float


def pack_telem_fast(t: FastTelemetry) -> bytes:
    return struct.pack(
        "<hhHh",
        int(round(_clamp(t.iq_ma * CURRENT_SCALE, -32768, 32767))),
        int(round(_clamp(t.id_ma * CURRENT_SCALE, -32768, 32767))),
        int(round(_clamp((t.angle_deg % 360.0) * ANGLE_SCALE, 0, 65535))),
        int(round(_clamp(t.velocity_dps * VELOCITY_SCALE, -32768, 32767))),
    )


def unpack_telem_fast(data: bytes) -> FastTelemetry:
    iq, id_, angle, vel = struct.unpack("<hhHh", bytes(data[:8]))
    return FastTelemetry(iq / CURRENT_SCALE, id_ / CURRENT_SCALE,
                         angle / ANGLE_SCALE, vel / VELOCITY_SCALE)



# this telemetry is necessary but at a lower frequency than the fast telem
@dataclass
class SlowTelemetry:
    vbus_v: float
    temp_c: float
    status: StatusFlags
    # how long the loop has been running in us
    loop_us: int


def pack_telem_slow(t: SlowTelemetry) -> bytes:
    return struct.pack(
        "<HhHH",
        int(round(_clamp(t.vbus_v * VBUS_SCALE, 0, 65535))),
        int(round(_clamp(t.temp_c * TEMP_SCALE, -32768, 32767))),
        int(t.status) & 0xFFFF,
        int(_clamp(t.loop_us, 0, 65535)),
    )


def unpack_telem_slow(data: bytes) -> SlowTelemetry:
    vbus, temp, status, loop_us = struct.unpack("<HhHH", bytes(data[:8]))
    return SlowTelemetry(vbus / VBUS_SCALE, temp / TEMP_SCALE,
                         StatusFlags(status), loop_us)
