"""Simulated actuator node — the zero-hardware sandbox.

Behaves like the firmware will: listens for heartbeat/command/PID frames,
runs a toy motor model + cascade control at 100 Hz, publishes fast telem
every tick and slow telem at 2 Hz, and FAULTS + COASTS if the heartbeat
goes quiet for 50 ms. Pull the (virtual) cable and watch it safe itself.

Run standalone:  python actuator_sim.py
"""
from __future__ import annotations

import argparse
import threading
import time

import protocol as P
from can_link import CanLink, HeartbeatMonitor, add_bus_args

LOOP_HZ = 100
SLOW_TELEM_DIV = 50          # 100 Hz / 50 = 2 Hz
IQ_LIMIT_MA = 1500.0         # sane cap for the 2208 on a 12 V bus

# toy plant: accel proportional to iq, viscous damping. NOT physics — just
# stable, plausible-looking dynamics so the GUI has something honest to plot.
ACCEL_DPS2_PER_MA = 40.0
DAMPING_PER_S = 3.0


class ActuatorSim(threading.Thread):
    def __init__(self, link: CanLink) -> None:
        super().__init__(daemon=True, name="actuator-sim")
        self.link = link
        self.hb = HeartbeatMonitor()

        self.mode = P.Mode.DISABLE
        self.target = 0.0
        self.enabled = False
        self.fault = False

        self.angle = 0.0          # deg, wraps 0..360 for telem
        self._unwrapped = 0.0     # deg, continuous — what position control acts on
        self.velocity = 0.0       # dps
        self.iq = 0.0             # mA
        self._vel_i = 0.0         # velocity-loop integrator

        # gains keyed by loop; defaults give a soft but stable demo
        self.gains = {
            P.Loop.VELOCITY: {"kp": 2.0, "ki": 1.0, "kd": 0.0},
            P.Loop.POSITION: {"kp": 5.0, "ki": 0.0, "kd": 0.0},
            P.Loop.CURRENT:  {"kp": 0.0, "ki": 0.0, "kd": 0.0},  # unused in sim
        }

        link.on(P.CanId.Heartbeat, self._on_heartbeat)
        link.on(P.CanId.COMMAND, self._on_command)
        link.on(P.CanId.PID_GAINS, self._on_pid)

    # ---- frame handlers (RX thread) ---------------------------------------
    def _on_heartbeat(self, msg) -> None:
        self.hb.beat()

    def _on_command(self, msg) -> None:
        mode, target, enable = P.unpack_command(bytes(msg.data))
        self.mode, self.target = mode, target
        if enable and self.hb.ok():
            self.enabled = True
            self.fault = False       # a fresh enable after heartbeat returns clears the latch
            self._vel_i = 0.0
        elif not enable:
            self.enabled = False

    def _on_pid(self, msg) -> None:
        loop, kp, ki, kd = P.unpack_pid(bytes(msg.data))
        self.gains[loop] = {"kp": kp, "ki": ki, "kd": kd}

    # ---- control + plant (sim thread) --------------------------------------
    def _control(self, dt: float) -> None:
        if self.mode == P.Mode.TORQUE:
            self.iq = self.target
        elif self.mode == P.Mode.VELOCITY:
            self.iq = self._vel_pi(self.target, dt)
        elif self.mode == P.Mode.POSITION:
            err = self.target - self._unwrapped
            vel_cmd = self.gains[P.Loop.POSITION]["kp"] * err
            self.iq = self._vel_pi(vel_cmd, dt)
        else:
            self.iq = 0.0
        self.iq = max(-IQ_LIMIT_MA, min(IQ_LIMIT_MA, self.iq))

    def _vel_pi(self, vel_target: float, dt: float) -> float:
        g = self.gains[P.Loop.VELOCITY]
        err = vel_target - self.velocity
        self._vel_i = max(-IQ_LIMIT_MA, min(IQ_LIMIT_MA, self._vel_i + g["ki"] * err * dt * 100))
        return g["kp"] * err * 10 + self._vel_i

    def run(self) -> None:
        dt = 1.0 / LOOP_HZ
        tick = 0
        next_t = time.monotonic()
        while True:
            t0 = time.perf_counter()

            # fail-safe gate — the whole point of the heartbeat
            if self.enabled and not self.hb.ok():
                self.enabled = False
                self.fault = True
                print("[sim] HEARTBEAT LOST -> fault, coasting")

            if self.enabled:
                self._control(dt)
            else:
                self.iq = 0.0  # coast: no drive, plant just damps out

            # plant
            accel = ACCEL_DPS2_PER_MA * self.iq / 1000.0
            self.velocity += (accel - DAMPING_PER_S * self.velocity) * dt
            self._unwrapped += self.velocity * dt
            self.angle = self._unwrapped % 360.0

            loop_us = int((time.perf_counter() - t0) * 1e6)

            self.link.send_telem_fast(P.FastTelemetry(
                iq_ma=self.iq, id_ma=0.0,
                angle_deg=self.angle, velocity_dps=self.velocity))

            if tick % SLOW_TELEM_DIV == 0:
                status = P.StatusFlags(0)
                if self.enabled:
                    status |= P.StatusFlags.ENABLED
                if self.fault:
                    status |= P.StatusFlags.FAULT
                if self.hb.ok():
                    status |= P.StatusFlags.HEARTBEAT_OK
                self.link.send_telem_slow(P.SlowTelemetry(
                    vbus_v=12.0, temp_c=25.0 + abs(self.iq) / 200.0,
                    status=status, loop_us=loop_us))

            tick += 1
            next_t += dt
            time.sleep(max(0.0, next_t - time.monotonic()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="simulated actuator node")
    add_bus_args(ap)
    args = ap.parse_args()

    link = CanLink(args.iface, args.channel, name="sim")
    sim = ActuatorSim(link)
    sim.start()
    print(f"actuator sim on {args.iface}:{args.channel} — ctrl-c to stop")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        link.shutdown()