import requests


class Geocoding:

    def __init__(self):
        self.nominatim_url = "https://nominatim.openstreetmap.org/search"
        self.ip_api_url = "https://ipapi.co"

    def get_location(self, place=None, ip=None, browser_coords=None):

        if browser_coords:
            return self._coords_from_browser(browser_coords)

        if place:
            return self._coords_from_place(place)

        if ip:
            return self._coords_from_ip(ip)

        return None

    def _coords_from_ip(self, ip):

        url = f"{self.ip_api_url}/{ip}/json/" if ip else f"{self.ip_api_url}/json/"

        r = requests.get(url, timeout=5)
        data = r.json()

        coords = {
            "lat": data.get("latitude"),
            "lon": data.get("longitude"),
            "source": "ip"
        }

        address = self._reverse_geocode(coords["lat"], coords["lon"])
        if address:
            coords["address"] = address

        return coords

    def _reverse_geocode(self, lat, lon):
        if not lat or not lon:
            return None

        params = {
            "lat": lat,
            "lon": lon,
            "format": "json"
        }

        headers = {
            "User-Agent": "cognitor-bot"
        }

        r = requests.get("https://nominatim.openstreetmap.org/reverse", params=params, headers=headers, timeout=5)
        data = r.json()

        if not data:
            return None

        return {
            "city": data.get("address", {}).get("city"),
            "town": data.get("address", {}).get("town"),
            "village": data.get("address", {}).get("village"),
            "state": data.get("address", {}).get("state"),
            "country": data.get("address", {}).get("country"),
            "display_name": data.get("display_name")
        }

    @staticmethod
    def _coords_from_browser(coords):

        result = {
            "lat": coords["lat"],
            "lon": coords["lon"],
            "source": "browser"
        }

        geocoding = Geocoding()
        address = geocoding._reverse_geocode(coords["lat"], coords["lon"])
        if address:
            result["address"] = address

        return result

    def _coords_from_place(self, place):

        params = {
            "q": place,
            "format": "json",
            "limit": 1
        }

        headers = {
            "User-Agent": "cognitor-bot"
        }

        r = requests.get(self.nominatim_url, params=params, headers=headers, timeout=5)
        data = r.json()

        if not data:
            return None

        result = data[0]

        return {
            "lat": float(result["lat"]),
            "lon": float(result["lon"]),
            "display_name": result["display_name"],
            "source": "nominatim"
        }