import requests


# url = "api.openweathermap.org/data/2.5/weather?q={london}&appid={f87283ebd7f8ee743f5abf251d32505b}"
location = "helsinki"
apiKey = "f87283ebd7f8ee743f5abf251d32505b"
url = "https://api.openweathermap.org/data/2.5/weather?q="+location+"&appid="+apiKey

response = requests.request("GET", url)
#response = requests.request("GET", url, headers=headers, params=querystring)

print(response.text)