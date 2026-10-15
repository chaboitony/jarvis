# jarvis-install: tools
"""
tools/weather.py  |  PROJECT RIMA, RIMA-01 J.A.R.V.I.S.
Weather for home or any named place.
  Forecasts: Open-Meteo (free for non-commercial use, no account or key)
  Alerts:    U.S. National Weather Service (free, no key; U.S. places only)

Read-only: it fetches public data and changes nothing on the computer.
Only numbers and short alert names from these services reach the model, never
free-form text, so this plugin isn't marked untrusted.

Settings in config.toml:
  [tools.weather]
  place = "Manhattan, Kansas"   # what to call home
  latitude = 39.18361
  longitude = -96.57167
  alerts = true                 # check NWS alerts (U.S. only)
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime

from tools import tool

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
ALERTS_URL = "https://api.weather.gov/alerts/active"
USER_AGENT = "PROJECT-RIMA-JARVIS/1.0 (personal assistant)"  # NWS requires one
TIMEOUT = 10             # seconds to wait for a weather service
FORECAST_CACHE = 600     # reuse a forecast for 10 minutes
ALERT_CACHE = 120        # alerts matter more, so recheck after 2 minutes

HOME = {"name": "home", "lat": None, "lon": None, "alerts": True}
PROMPT = ""  # filled in by setup() with the home town's name
_cache = {}  # url -> (time fetched, data)

EXAMPLES = [
    "what's the weather", "will it rain tomorrow", "do I need a jacket", "how cold is it outside",
    "what's the forecast for the weekend", "is there a storm warning", "what's it like in Wichita",
]

# Open-Meteo reports conditions as WMO weather codes
WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "light freezing drizzle", 57: "heavy freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "violent rain showers",
    85: "light snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 97: "heavy thunderstorms",
    99: "thunderstorms with heavy hail",
}

# So "Wichita, KS" finds the Kansas one
STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
COMPASS = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]


def setup(settings):
    """Read [tools.weather] from config.toml. Refuses to load without coordinates."""
    global PROMPT
    try:
        lat, lon = float(settings["latitude"]), float(settings["longitude"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("add latitude and longitude numbers under [tools.weather] in config.toml")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("latitude must be -90 to 90 and longitude -180 to 180")
    HOME.update(name=str(settings.get("place", "home")), lat=lat, lon=lon,
                alerts=bool(settings.get("alerts", True)))
    PROMPT = f"""Weather: use get_weather for questions about weather, temperature, rain, snow, wind or what to wear.
Tony's home is {HOME['name']}. Leave place empty for home; only fill it in for somewhere else.
If the result lists active weather alerts, mention them first.
Give the key numbers briefly instead of reading out every detail."""


# --------------------------------------------------------------------------
# Talking to the weather services
# --------------------------------------------------------------------------
def _get_json(url, params, cache_seconds=FORECAST_CACHE, accept="application/json"):
    full_url = f"{url}?{urllib.parse.urlencode(params, safe=',')}"
    hit = _cache.get(full_url)
    if hit and time.monotonic() - hit[0] < cache_seconds:
        return hit[1]

    request = urllib.request.Request(full_url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            data = json.load(response)
    except urllib.error.HTTPError as e:
        reason = ""
        try:
            reason = json.load(e).get("reason", "")  # Open-Meteo explains bad requests
        except Exception:
            pass
        raise ValueError(f"the weather service returned an error ({e.code}) {reason}".strip())
    except (urllib.error.URLError, TimeoutError, OSError):
        raise ValueError("I can't reach the weather service; the internet connection may be down.")

    _cache[full_url] = (time.monotonic(), data)
    return data


def _find_place(place):
    """Turn a place name into coordinates. Empty, 'home' or the home town's name = home."""
    text = " ".join((place or "").split())
    home_names = {"", "home", "here", HOME["name"].lower(), HOME["name"].split(",")[0].strip().lower()}
    if text.lower() in home_names:
        return HOME
    if len(text) > 80:
        raise ValueError("That place name is too long.")

    name, _, region = (part.strip() for part in text.partition(","))
    region = STATES.get(region.upper(), region).lower()
    data = _get_json(GEOCODE_URL, {"name": name, "count": 10, "language": "en", "format": "json"},
                     cache_seconds=86400)
    results = data.get("results") or []
    if region:  # "Springfield, Missouri": keep only matching states or countries
        results = [r for r in results if region in (str(r.get("admin1", "")).lower(),
                                                    str(r.get("country", "")).lower(),
                                                    str(r.get("country_code", "")).lower())]
    if not results:
        raise ValueError(f"I couldn't find a place called '{text}'. Try adding the state, like 'Salina, Kansas'.")

    best = results[0]  # the service lists the most likely match first
    is_us = best.get("country_code") == "US"
    area = best.get("admin1") if is_us else best.get("country")
    return {
        "name": f"{best['name']}, {area}" if area else best["name"],
        "lat": float(best["latitude"]), "lon": float(best["longitude"]),
        "alerts": is_us and HOME["alerts"],
    }


def _forecast(where):
    return _get_json(FORECAST_URL, {
        "latitude": f"{where['lat']:.4f}", "longitude": f"{where['lon']:.4f}",
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,"
                   "wind_speed_10m,wind_direction_10m,wind_gusts_10m",
        "hourly": "precipitation_probability",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,"
                 "precipitation_sum,wind_speed_10m_max,wind_gusts_10m_max,sunrise,sunset",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "precipitation_unit": "inch",
        "timezone": "auto", "forecast_days": 7,
    })


def _alerts(where):
    """Active NWS alerts as one sentence, or None if alerts are off for this place."""
    if not where["alerts"]:
        return None
    try:
        data = _get_json(ALERTS_URL, {"point": f"{where['lat']:.4f},{where['lon']:.4f}"},
                         cache_seconds=ALERT_CACHE, accept="application/geo+json")
    except ValueError:
        return "Weather alerts couldn't be checked right now."
    found = []
    for feature in data.get("features") or []:
        p = feature.get("properties") or {}
        if p.get("status") != "Actual" or p.get("messageType") == "Cancel":
            continue  # skip test messages and cancellations
        event = " ".join(str(p.get("event") or "Weather alert").split())[:60]
        ends = p.get("ends") or p.get("expires")
        text = f"{event} until {_clock(ends)} {_weekday(ends)}" if ends else event
        if text not in found:
            found.append(text)
    if not found:
        return "No active weather alerts."
    return "ACTIVE WEATHER ALERTS: " + "; ".join(found) + "."


# --------------------------------------------------------------------------
# Turning numbers into sentences
# --------------------------------------------------------------------------
def _clock(stamp):
    """'2026-10-15T15:00' or '...-05:00' -> '3 PM' / '3:30 PM'."""
    t = datetime.fromisoformat(stamp)
    hour = t.hour % 12 or 12
    minutes = f":{t.minute:02d}" if t.minute else ""
    return f"{hour}{minutes} {'AM' if t.hour < 12 else 'PM'}"


def _weekday(stamp):
    return datetime.fromisoformat(stamp).strftime("%A")


def _describe(code):
    return WEATHER_CODES.get(int(code), "unknown conditions") if code is not None else "unknown conditions"


def _now_sentence(c):
    wind = round(c["wind_speed_10m"])
    text = (f"Now: {round(c['temperature_2m'])}°F, feels like {round(c['apparent_temperature'])}°F, "
            f"{_describe(c['weather_code'])}, humidity {round(c['relative_humidity_2m'])}%")
    if wind < 2:
        return text + ", calm wind."
    direction = COMPASS[round(c["wind_direction_10m"] / 45) % 8]
    text += f", wind from the {direction} at {wind} mph"
    gusts = round(c.get("wind_gusts_10m") or 0)
    return text + (f", gusts to {gusts} mph." if gusts >= wind + 8 else ".")


def _day_sentence(d, i, dates):
    day = dates[i]
    label = day.strftime("%A, %b ") + str(day.day)
    if i == 0:
        label = f"Today ({label})"
    elif i == 1:
        label = f"Tomorrow ({label})"
    text = (f"{label}: {_describe(d['weather_code'][i])}, high {round(d['temperature_2m_max'][i])}°F, "
            f"low {round(d['temperature_2m_min'][i])}°F, "
            f"{round(d['precipitation_probability_max'][i] or 0)}% chance of precipitation")
    amount = d["precipitation_sum"][i] or 0
    if amount >= 0.01:
        text += f" (about {amount:.2f} in)"
    wind, gusts = round(d["wind_speed_10m_max"][i] or 0), round(d["wind_gusts_10m_max"][i] or 0)
    if wind >= 15:
        text += f", wind up to {wind} mph with gusts to {gusts}"
    return text + "."


def _rain_timing(data, day, after=None):
    """When precipitation is most likely on one day, if it's worth mentioning."""
    hourly = data.get("hourly") or {}
    best = None
    for stamp, chance in zip(hourly.get("time", []), hourly.get("precipitation_probability", [])):
        if chance is None or not stamp.startswith(day.isoformat()):
            continue
        if after and stamp < after:  # today: only the hours still ahead
            continue
        if best is None or chance > best[1]:
            best = (stamp, chance)
    if best and best[1] >= 30:
        return f"Precipitation is most likely around {_clock(best[0])} ({round(best[1])}% chance)."
    return None


def _pick_days(day, dates):
    """Which forecast days a request means, as positions in the forecast (0 = today)."""
    text = " ".join((day or "today").lower().replace("this ", "").split())
    if text in {"", "today", "now", "current", "currently", "tonight", "right now"}:
        return [0]
    if text == "tomorrow":
        return [1]
    if text in {"week", "next 7 days", "7 days", "forecast", "next week"}:
        return list(range(len(dates)))
    if text == "weekend":
        start = next(i for i, d in enumerate(dates) if d.weekday() >= 5)
        days = [start]
        if dates[start].weekday() == 5 and start + 1 < len(dates):
            days.append(start + 1)
        return days
    if text in WEEKDAYS:
        return [next(i for i, d in enumerate(dates) if d.weekday() == WEEKDAYS.index(text))]
    try:
        wanted = date.fromisoformat(text)
    except ValueError:
        raise ValueError(f"I don't understand the day '{day}'. Use today, tomorrow, a weekday name, "
                         "weekend, week, or YYYY-MM-DD.")
    if wanted not in dates:
        raise ValueError(f"I only have forecasts from {dates[0]} to {dates[-1]}.")
    return [dates.index(wanted)]


# --------------------------------------------------------------------------
# The tool
# --------------------------------------------------------------------------
@tool
def get_weather(day: str = "today", place: str = "") -> str:
    """Get the weather: current conditions, the forecast for a day or the week, and active weather alerts.

    Args:
        day: today, tomorrow, a weekday name, weekend, week, or YYYY-MM-DD (up to 7 days ahead)
        place: a town such as "Wichita, Kansas". Leave empty for Tony's home.
    """
    where = _find_place(place)
    data = _forecast(where)
    daily = data["daily"]
    dates = [date.fromisoformat(t) for t in daily["time"]]
    picks = _pick_days(day, dates)

    parts = [f"Weather for {where['name']}."]
    alerts = _alerts(where)
    if alerts:
        parts.append(alerts)
    if picks == [0]:
        parts.append(_now_sentence(data["current"]))
    parts += [_day_sentence(daily, i, dates) for i in picks]
    if len(picks) == 1:
        i = picks[0]
        timing = _rain_timing(data, dates[i], after=data["current"]["time"][:13] if i == 0 else None)
        if timing:
            parts.append(timing)
        parts.append(f"Sunrise {_clock(daily['sunrise'][i])}, sunset {_clock(daily['sunset'][i])}.")
    return " ".join(parts)


def briefing():
    """Today's weather in one or two sentences for the welcome briefing (home only)."""
    data = _forecast(HOME)
    c, d = data["current"], data["daily"]
    text = (f"Weather: {round(c['temperature_2m'])}°F and {_describe(c['weather_code'])} now; "
            f"today's high {round(d['temperature_2m_max'][0])}°F, low {round(d['temperature_2m_min'][0])}°F, "
            f"{round(d['precipitation_probability_max'][0] or 0)}% chance of precipitation.")
    timing = _rain_timing(data, date.fromisoformat(d["time"][0]), after=c["time"][:13])
    if timing:
        text += " " + timing
    alerts = _alerts(HOME)
    if alerts and alerts.startswith("ACTIVE"):
        text = alerts + " " + text
    return text
