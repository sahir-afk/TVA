"""Host GUI — the flight-computer stand-in.

Sends the 100 Hz heartbeat, commands mode/target/enable, pushes PID gains,
and plots fast telemetry live. The "Pause heartbeat" box is the fail-safe
demo: check it and the node must fault + coast within ~50 ms.

Sandbox, two terminals:              or everything in one process:
  python actuator_sim.py               python app.py --sim
  python app.py
(optional third terminal:  python can_link.py  to spy the bus)
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque

import pyqtgraph as pg
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QMainWindow, QPushButton, QVBoxLayout, QWidget)

import protocol as P
from can_link import CanLink, add_bus_args

HISTORY_S = 10.0
PLOT_HZ = 30
HEARTBEAT_MS = 10       # 100 Hz
COMMAND_MS = 100        # refresh command 10x/s so the node tracks the GUI


class HostWindow(QMainWindow):
    def __init__(self, link: CanLink) -> None:
        super().__init__()
        self.link = link
        self.setWindowTitle("HITL Actuator Host")
        self._t0 = time.monotonic()
        self._hb_counter = 0

        # telem buffers — appended from the CAN RX thread, drained by the GUI
        # timer. deque appends are atomic in CPython, so no lock needed.
        n = int(HISTORY_S * 100)
        self.t_buf = deque(maxlen=n)
        self.angle_buf = deque(maxlen=n)
        self.vel_buf = deque(maxlen=n)
        self.iq_buf = deque(maxlen=n)
        self._slow = None  # latest SlowTelemetry, read by GUI timer

        link.on(P.CanId.TELEM_FAST, self._on_fast)
        link.on(P.CanId.TELEM_SLOW, self._on_slow)

        self._build_ui()

        self._hb_timer = QTimer(self, interval=HEARTBEAT_MS, timeout=self._tick_heartbeat)
        self._hb_timer.start()
        self._cmd_timer = QTimer(self, interval=COMMAND_MS, timeout=self._send_command)
        self._cmd_timer.start()
        self._plot_timer = QTimer(self, interval=int(1000 / PLOT_HZ), timeout=self._redraw)
        self._plot_timer.start()

    # ---- CAN RX (not the GUI thread — buffer only, never touch widgets) ----
    def _on_fast(self, msg) -> None:
        t = P.unpack_telem_fast(bytes(msg.data))
        self.t_buf.append(time.monotonic() - self._t0)
        self.angle_buf.append(t.angle_deg)
        self.vel_buf.append(t.velocity_dps)
        self.iq_buf.append(t.iq_ma)

    def _on_slow(self, msg) -> None:
        self._slow = P.unpack_telem_slow(bytes(msg.data))

    # ---- TX ----------------------------------------------------------------
    def _tick_heartbeat(self) -> None:
        if self.hb_pause.isChecked():
            return  # fail-safe demo: starve the node
        self._hb_counter += 1
        self.link.send_heartbeat(self._hb_counter, armed=self.enable_box.isChecked())

    def _send_command(self) -> None:
        mode = P.Mode(self.mode_combo.currentData())
        self.link.send_command(mode, self.target_spin.value(),
                               self.enable_box.isChecked())

    def _send_pid(self) -> None:
        loop = P.Loop(self.loop_combo.currentData())
        self.link.send_pid(loop, self.kp.value(), self.ki.value(), self.kd.value())

    # ---- UI ----------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QWidget(); self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        # left column: controls
        ctrl = QVBoxLayout(); layout.addLayout(ctrl, 0)

        cmd_box = QGroupBox("Command"); f = QFormLayout(cmd_box)
        self.mode_combo = QComboBox()
        for m in (P.Mode.DISABLE, P.Mode.TORQUE, P.Mode.VELOCITY, P.Mode.POSITION):
            self.mode_combo.addItem(m.name, int(m))
        self.target_spin = QDoubleSpinBox(minimum=-3000, maximum=3000, decimals=1)
        self.enable_box = QCheckBox("Enable drive")
        self.hb_pause = QCheckBox("Pause heartbeat (fail-safe demo)")
        f.addRow("Mode", self.mode_combo)
        f.addRow("Target", self.target_spin)
        f.addRow(self.enable_box)
        f.addRow(self.hb_pause)
        ctrl.addWidget(cmd_box)

        pid_box = QGroupBox("PID gains"); f = QFormLayout(pid_box)
        self.loop_combo = QComboBox()
        for l in (P.Loop.CURRENT, P.Loop.VELOCITY, P.Loop.POSITION):
            self.loop_combo.addItem(l.name, int(l))
        self.kp = QDoubleSpinBox(maximum=65.5, decimals=3, singleStep=0.1)
        self.ki = QDoubleSpinBox(maximum=65.5, decimals=3, singleStep=0.1)
        self.kd = QDoubleSpinBox(maximum=65.5, decimals=3, singleStep=0.01)
        send_pid = QPushButton("Send gains"); send_pid.clicked.connect(self._send_pid)
        f.addRow("Loop", self.loop_combo)
        f.addRow("Kp", self.kp); f.addRow("Ki", self.ki); f.addRow("Kd", self.kd)
        f.addRow(send_pid)
        ctrl.addWidget(pid_box)

        self.status_lbl = QLabel("no slow telem yet")
        self.status_lbl.setWordWrap(True)
        ctrl.addWidget(self.status_lbl)
        ctrl.addStretch(1)

        # right column: plots
        plots = QVBoxLayout(); layout.addLayout(plots, 1)
        pg.setConfigOptions(antialias=True)
        self.p_angle = pg.PlotWidget(title="Angle (deg)")
        self.p_vel = pg.PlotWidget(title="Velocity (dps)")
        self.p_iq = pg.PlotWidget(title="Iq (mA)")
        self.c_angle = self.p_angle.plot(pen="c")
        self.c_vel = self.p_vel.plot(pen="y")
        self.c_iq = self.p_iq.plot(pen="m")
        for p in (self.p_angle, self.p_vel, self.p_iq):
            p.showGrid(x=True, y=True, alpha=0.3)
            plots.addWidget(p)

    def _redraw(self) -> None:
        t = list(self.t_buf)
        self.c_angle.setData(t, list(self.angle_buf))
        self.c_vel.setData(t, list(self.vel_buf))
        self.c_iq.setData(t, list(self.iq_buf))
        s = self._slow
        if s is not None:
            flags = ", ".join(f.name for f in P.StatusFlags if f in s.status) or "none"
            self.status_lbl.setText(
                f"Vbus {s.vbus_v:.2f} V   temp {s.temp_c:.1f} °C   "
                f"loop {s.loop_us} µs\nstatus: {flags}")


def main() -> None:
    ap = argparse.ArgumentParser(description="HITL host GUI")
    add_bus_args(ap)
    ap.add_argument("--sim", action="store_true",
                    help="run the actuator sim in-process (single-terminal sandbox)")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    link = CanLink(args.iface, args.channel, name="host")

    if args.sim:
        from actuator_sim import ActuatorSim
        sim_link = CanLink(args.iface, args.channel, name="sim")
        ActuatorSim(sim_link).start()

    win = HostWindow(link)
    win.resize(1100, 700)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()