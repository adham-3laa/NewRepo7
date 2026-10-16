"""Simulated smart home state and actions.

The house has exactly:
- one front door (open/closed + locked/unlocked)
- one window (open/closed)
- three rooms (kitchen, bathroom, bedroom),
  each with one light and one electrical device.

Devices (approved):
- kitchen = oven
- bathroom = water heater
- bedroom = air conditioner
"""


class SmartHome:
    """Holds the house state. No Streamlit or speech code here."""

    ROOMS = ("kitchen", "bathroom", "bedroom")

    # Device name shown in the UI per room.
    DEVICE_NAMES = {
        "kitchen": "oven",
        "bathroom": "water heater",
        "bedroom": "air conditioner",
    }

    def __init__(self):
        # Door and window state.
        self.door_open = False
        self.door_locked = False
        self.window_open = False
        # All lights and devices start OFF.
        self.lights = {room: False for room in self.ROOMS}
        self.devices = {room: False for room in self.ROOMS}

    # ---- lights ----
    def turn_light_on(self, room):
        """Turn a room light on. Returns (message, changed)."""
        if room not in self.ROOMS:
            return ("Unknown room: %s." % room, False)
        if self.lights[room]:
            return ("%s light is already ON. Nothing changed." % room.title(), False)
        self.lights[room] = True
        return ("%s light turned ON." % room.title(), True)

    def turn_light_off(self, room):
        """Turn a room light off. Returns (message, changed)."""
        if room not in self.ROOMS:
            return ("Unknown room: %s." % room, False)
        if not self.lights[room]:
            return ("%s light is already OFF. Nothing changed." % room.title(), False)
        self.lights[room] = False
        return ("%s light turned OFF." % room.title(), True)

    # ---- devices ----
    def turn_device_on(self, room):
        """Turn a room device on. Returns (message, changed)."""
        if room not in self.ROOMS:
            return ("Unknown room: %s." % room, False)
        name = self.DEVICE_NAMES[room]
        if self.devices[room]:
            return ("%s is already ON. Nothing changed." % name.title(), False)
        self.devices[room] = True
        return ("%s turned ON." % name.title(), True)

    def turn_device_off(self, room):
        """Turn a room device off. Returns (message, changed)."""
        if room not in self.ROOMS:
            return ("Unknown room: %s." % room, False)
        name = self.DEVICE_NAMES[room]
        if not self.devices[room]:
            return ("%s is already OFF. Nothing changed." % name.title(), False)
        self.devices[room] = False
        return ("%s turned OFF." % name.title(), True)

    # ---- door ----
    def open_door(self):
        """Open the front door. A locked door cannot be opened."""
        if self.door_locked:
            return ("Door is locked. Unlock it first.", False)
        if self.door_open:
            return ("Door is already OPEN. Nothing changed.", False)
        self.door_open = True
        return ("Door opened.", True)

    def close_door(self):
        """Close the front door."""
        if not self.door_open:
            return ("Door is already CLOSED. Nothing changed.", False)
        self.door_open = False
        return ("Door closed.", True)

    def lock_door(self):
        """Lock the front door. An open door cannot be locked."""
        if self.door_open:
            return ("Door is open. Close it before locking.", False)
        if self.door_locked:
            return ("Door is already LOCKED. Nothing changed.", False)
        self.door_locked = True
        return ("Door locked.", True)

    def unlock_door(self):
        """Unlock the front door."""
        if not self.door_locked:
            return ("Door is already UNLOCKED. Nothing changed.", False)
        self.door_locked = False
        return ("Door unlocked.", True)

    # ---- window ----
    def open_window(self):
        """Open the window."""
        if self.window_open:
            return ("Window is already OPEN. Nothing changed.", False)
        self.window_open = True
        return ("Window opened.", True)

    def close_window(self):
        """Close the window."""
        if not self.window_open:
            return ("Window is already CLOSED. Nothing changed.", False)
        self.window_open = False
        return ("Window closed.", True)

    # ---- global actions ----
    def all_lights_off(self):
        """Switch all three lights off. Door, window and devices untouched."""
        changed_any = False
        for room in self.ROOMS:
            if self.lights[room]:
                self.lights[room] = False
                changed_any = True
        if changed_any:
            return ("All lights turned OFF.", True)
        return ("All lights were already OFF. Nothing changed.", False)

    def everything_off(self):
        """Switch all lights and all devices off. Door and window untouched."""
        changed_any = False
        for room in self.ROOMS:
            if self.lights[room]:
                self.lights[room] = False
                changed_any = True
            if self.devices[room]:
                self.devices[room] = False
                changed_any = True
        if changed_any:
            return ("Everything turned OFF (all lights and devices).", True)
        return ("Everything was already OFF. Nothing changed.", False)

    # ---- status ----
    def status_text(self):
        """Return a multi-line English summary of the whole house."""
        door = "OPEN" if self.door_open else "CLOSED"
        lock = "LOCKED" if self.door_locked else "UNLOCKED"
        window = "OPEN" if self.window_open else "CLOSED"
        lines = ["House status:", "Front door: %s, %s." % (door, lock),
                 "Window: %s." % window]
        for room in self.ROOMS:
            light = "ON" if self.lights[room] else "OFF"
            dev_state = "ON" if self.devices[room] else "OFF"
            lines.append("%s: light %s, %s %s." % (
                room.title(), light, self.DEVICE_NAMES[room], dev_state))
        return "\n".join(lines)
