import time
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager

options = Options()
options.headless = False  # ponemos visible para debug

driver = webdriver.Firefox(
    service=Service(GeckoDriverManager().install()),
    options=options
)

driver.get("https://marketplace.watchcharts.com/listings?page=1&source=watchcharts+marketplace&status=all")

time.sleep(10)  # esperar 10 segundos a que React cargue contenido

html = driver.page_source
with open("html_watchcharts.html", "w", encoding="utf-8") as f:
    f.write(html)

driver.quit()
print("HTML guardado en html_watchcharts.html")
