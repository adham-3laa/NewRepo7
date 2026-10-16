"""Top-down 2D floor plan as an inline SVG string.

Pure functions: the ONLY inputs are the nine SmartHome booleans
(door open/locked, window open, three lights, three devices).
No transcripts, no user text, no session state enter this module,
so user input can never become markup. All labels are fixed English
constants (escaped at the boundary anyway).

Geometry (viewBox 0 0 720 460): outer walls, three equal rooms
kitchen | bathroom | bedroom, front door gap in the kitchen's
north wall segment, window gap in the bedroom's east wall.
"""

import html as _html
import os
import tempfile

# Fixed geometry: outer walls, room divider x positions, openings.
WALL = (20, 20, 680, 420)
DIVIDERS = (250, 480)
DOOR_GAP = (100, 170)      # x-range of the front-door gap (north wall).
WINDOW_GAP = (200, 290)    # y-range of the window gap (east wall).
FLOORPLAN_FILE = os.path.join(tempfile.gettempdir(),
                              "smart_home_floorplan.svg")

_LABELS = {
    "kitchen": "Kitchen", "bathroom": "Bathroom", "bedroom": "Bedroom",
    "oven": "Oven", "heater": "Water heater", "ac": "Air conditioner",
    "bulb": "Light", "door": "Front door", "window": "Window",
}

_STYLE = """
.hp-glow { animation: hp-pulse 2s ease-in-out infinite; }
@keyframes hp-pulse {
  0%, 100% { opacity: 0.45; }
  50% { opacity: 1.0; }
}
.hp-wave { animation: hp-rise 2s linear infinite; }
@keyframes hp-rise {
  from { transform: translateY(0px); opacity: 0.9; }
  to { transform: translateY(-22px); opacity: 0.1; }
}
.hp-air { animation: hp-dash 1.6s linear infinite; }
@keyframes hp-dash {
  to { stroke-dashoffset: -28; }
}
@media (prefers-reduced-motion: reduce) {
  .hp-glow, .hp-wave, .hp-air { animation: none; }
}
"""


def _esc(text):
    """Escape text for SVG. All labels are constants; belt and braces."""
    return _html.escape(str(text), quote=True)


def _room_base(x0, x1, room, label):
    """Floor rect, divider walls and the room name caption."""
    parts = [
        '<rect x="%d" y="28" width="%d" height="404" fill="#eef1f4"/>' % (
            x0, x1 - x0),
        '<text x="%d" y="52" font-size="15" fill="#5b6b7a">%s</text>' % (
            x0 + 10, _esc(label)),
    ]
    return "".join(parts)


def _room_overlay(x0, x1, room, light_on):
    """Warm illumination rect plus a marker id used by tests."""
    if not light_on:
        return '<g id="glow-%s-off"/>' % room
    return ('<g id="glow-%s-on">'
            '<rect x="%d" y="28" width="%d" height="404" '
            'fill="#ffd97a" opacity="0.35" class="hp-glow"/>'
            '</g>' % (room, x0, x1 - x0))


def _bulb(cx, room, light_on):
    """Bulb glyph: yellow + glow ring when ON, gray when OFF."""
    if light_on:
        return ('<g id="bulb-%s-on">'
                '<circle cx="%d" cy="90" r="22" fill="#ffd97a" '
                'opacity="0.5" class="hp-glow"/>'
                '<circle cx="%d" cy="90" r="11" fill="#f5b301" '
                'stroke="#8a5b00" stroke-width="2"/>'
                '<text x="%d" y="122" font-size="12" fill="#5b6b7a">%s: '
                'ON</text></g>' % (room, cx, cx, cx, _esc(_LABELS["bulb"])))
    return ('<g id="bulb-%s-off">'
            '<circle cx="%d" cy="90" r="11" fill="#c3ccd4" '
            'stroke="#5b6b7a" stroke-width="2"/>'
            '<text x="%d" y="122" font-size="12" fill="#5b6b7a">%s: '
            'OFF</text></g>' % (room, cx, cx, _esc(_LABELS["bulb"])))


def _oven(x, y, device_on):
    """Kitchen stove: heating bars + rising waves only when ON."""
    parts = ['<g id="oven-%s">' % ("on" if device_on else "off"),
             '<rect x="%d" y="%d" width="120" height="80" rx="6" '
             'fill="#dfe5ea" stroke="#5b6b7a" stroke-width="2"/>' % (x, y),
             '<text x="%d" y="%d" font-size="13" fill="#333333">%s</text>' % (
                 x + 8, y + 98, _esc(_LABELS["oven"]))]
    for i, dx in enumerate((20, 55, 90)):
        color = "#e07b39" if device_on else "#9aa5b1"
        parts.append('<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="%s" '
                     'stroke-width="6"/>' % (x + dx, y + 55, x + dx, y + 68,
                                            color))
    if device_on:
        parts.append('<circle cx="%d" cy="%d" r="7" fill="#2f9e44"/>' % (
            x + 108, y + 12))
        for dx in (35, 75):
            parts.append('<path class="hp-wave" d="M %d %d q 8 -16 0 -32 '
                         'q -8 -16 0 -32" fill="none" stroke="#e07b39" '
                         'stroke-width="4"/>' % (x + dx, y - 8))
    else:
        parts.append('<circle cx="%d" cy="%d" r="7" fill="#9aa5b1"/>' % (
            x + 108, y + 12))
    parts.append("</g>")
    return "".join(parts)


def _heater(x, y, device_on):
    """Bathroom water heater: pilot dot + one restrained wave when ON."""
    parts = ['<g id="heater-%s">' % ("on" if device_on else "off"),
             '<rect x="%d" y="%d" width="90" height="120" rx="10" '
             'fill="#dfe5ea" stroke="#5b6b7a" stroke-width="2"/>' % (x, y),
             '<text x="%d" y="%d" font-size="13" fill="#333333">%s</text>' % (
                 x, y + 138, _esc(_LABELS["heater"]))]
    dot = "#2f9e44" if device_on else "#9aa5b1"
    parts.append('<circle cx="%d" cy="%d" r="7" fill="%s"/>' % (
        x + 45, y + 25, dot))
    if device_on:
        parts.append('<path class="hp-wave" d="M %d %d q 8 -16 0 -32 '
                     'q -8 -16 0 -32" fill="none" stroke="#e07b39" '
                     'stroke-width="4"/>' % (x + 45, y - 8))
    parts.append("</g>")
    return "".join(parts)


def _ac(x, y, device_on):
    """Bedroom air conditioner: airflow dashes only when ON."""
    parts = ['<g id="ac-%s">' % ("on" if device_on else "off"),
             '<rect x="%d" y="%d" width="130" height="44" rx="6" '
             'fill="#dfe5ea" stroke="#5b6b7a" stroke-width="2"/>' % (x, y),
             '<text x="%d" y="%d" font-size="13" fill="#333333">%s</text>' % (
                 x, y + 62, _esc(_LABELS["ac"]))]
    dot = "#2f7fd1" if device_on else "#9aa5b1"
    parts.append('<circle cx="%d" cy="%d" r="7" fill="%s"/>' % (
        x + 117, y + 22, dot))
    if device_on:
        for dy in (70, 84, 98):
            parts.append('<line class="hp-air" x1="%d" y1="%d" x2="%d" '
                         'y2="%d" stroke="#2f7fd1" stroke-width="4" '
                         'stroke-dasharray="10 4"/>' % (x + 15, y + dy,
                                                       x + 100, y + dy))
    parts.append("</g>")
    return "".join(parts)


def _door(door_open, door_locked):
    """North-wall gap, leaf position, padlock state. Reads actual state."""
    x0, x1 = DOOR_GAP
    parts = ['<g id="door-%s-%s">' % ("open" if door_open else "closed",
                                      "locked" if door_locked else "unlocked"),
             '<line x1="20" y1="20" x2="%d" y2="20" stroke="#5b6b7a" '
             'stroke-width="8"/>' % x0,
             '<line x1="%d" y1="20" x2="700" y2="20" stroke="#5b6b7a" '
             'stroke-width="8"/>' % x1]
    if door_open:
        parts.append('<line x1="%d" y1="20" x2="%d" y2="86" stroke="#8a5b00" '
                     'stroke-width="7"/>' % (x0, x0 + 24))
    else:
        parts.append('<line x1="%d" y1="20" x2="%d" y2="20" stroke="#8a5b00" '
                     'stroke-width="7"/>' % (x0, x1))
    shackle = "#e8a100" if door_locked else "#9aa5b1"
    lock_text = "LOCKED" if door_locked else "UNLOCKED"
    parts.append('<rect x="%d" y="30" width="26" height="20" rx="3" '
                 'fill="%s"/>' % (x1 + 8, shackle))
    parts.append('<text x="%d" y="66" font-size="12" fill="#333333">%s</text>' %
                 (x1 + 8, lock_text))
    parts.append('<text x="60" y="14" font-size="12" fill="#5b6b7a">%s '
                 '(in: kitchen)</text>' % _esc(_LABELS["door"]))
    parts.append("</g>")
    return "".join(parts)


def _window(window_open):
    """East-wall gap with sliding panes: apart when open."""
    y0, y1 = WINDOW_GAP
    mid = (y0 + y1) // 2
    parts = ['<g id="window-%s">' % ("open" if window_open else "closed"),
             '<line x1="700" y1="20" x2="700" y2="%d" stroke="#5b6b7a" '
             'stroke-width="8"/>' % y0,
             '<line x1="700" y1="%d" x2="700" y2="440" stroke="#5b6b7a" '
             'stroke-width="8"/>' % y1]
    if window_open:
        top, bottom = (y0, y0 + 28), (y1 - 28, y1)
        caption = "OPEN"
    else:
        top, bottom = (y0, mid), (mid, y1)
        caption = "CLOSED"
    for ya, yb in (top, bottom):
        parts.append('<rect x="694" y="%d" width="12" height="%d" '
                     'fill="#7ab8ff" stroke="#33576f" '
                     'stroke-width="2"/>' % (ya, yb - ya))
    parts.append('<text x="600" y="%d" font-size="12" fill="#333333">%s: '
                 '%s</text>' % (y1 + 22, _esc(_LABELS["window"]), caption))
    parts.append("</g>")
    return "".join(parts)


def _walls():
    """Outer walls (with door/window gaps left to those builders)."""
    x0, y0, w, h = WALL
    return (
        '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#5b6b7a" '
        'stroke-width="8"/>' % (x0, y0 + h, x0 + w, y0 + h) +
        '<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#5b6b7a" '
        'stroke-width="8"/>' % (x0, y0, x0, y0 + h) +
        '<line x1="%d" y1="20" x2="%d" y2="440" stroke="#5b6b7a" '
        'stroke-width="5"/>' % (DIVIDERS[0], DIVIDERS[0]) +
        '<line x1="%d" y1="20" x2="%d" y2="440" stroke="#5b6b7a" '
        'stroke-width="5"/>' % (DIVIDERS[1], DIVIDERS[1]))


def render_floor_plan(home):
    """Build the full SVG string from the SmartHome booleans."""
    rooms = {"kitchen": (28, 250), "bathroom": (250, 480),
             "bedroom": (480, 692)}
    bulbs = {"kitchen": 139, "bathroom": 365, "bedroom": 586}
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 460" '
        'width="720" height="460" role="img" aria-label="Smart home '
        'floor plan" data-door="%s" data-lock="%s" data-window="%s">' % (
            "open" if home.door_open else "closed",
            "locked" if home.door_locked else "unlocked",
            "open" if home.window_open else "closed"),
        "<style>%s</style>" % _STYLE,
        '<rect x="0" y="0" width="720" height="460" fill="#ffffff"/>',
    ]
    for room, (x0, x1) in rooms.items():
        parts.append(_room_base(x0, x1, room, _LABELS[room]))
        parts.append(_room_overlay(x0, x1, room, home.lights[room]))
        parts.append(_bulb(bulbs[room], room, home.lights[room]))
    parts.append(_oven(60, 300, home.devices["kitchen"]))
    parts.append(_heater(315, 260, home.devices["bathroom"]))
    parts.append(_ac(520, 160, home.devices["bedroom"]))
    parts.append(_walls())
    parts.append(_door(home.door_open, home.door_locked))
    parts.append(_window(home.window_open))
    parts.append("</svg>")
    return "".join(parts)


def write_floorplan(home, path=FLOORPLAN_FILE):
    """Write the current SVG to a single reused temp path. No buildup."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(render_floor_plan(home))
    return path
