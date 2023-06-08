from google_images_download import google_images_download   #importing the library

chromedriver = "chromedriver.exe"
response = google_images_download.googleimagesdownload()
response.download({
    "keywords": "weasel",
    "size": "medium",
    "limit": 500,
    "chromedriver": chromedriver})