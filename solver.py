import os
import time
import traceback
import csv
from datetime import datetime
from time import sleep
import torch

import cv2
import numpy as np
from PIL import Image

from ultralytics import YOLO

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException


from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions


CAPTCHA_URL = "https://www.google.com/recaptcha/api2/demo"
MODEL_PATH = "./best.pt"   
USE_TOP_N_STRATEGY = False
N = 3

CLASSES = ["bicycle", "bridge", "bus", "car", "chimney", "crosswalk", "hydrant", "motorcycle", "other", "palm", "stairs", "traffic"]
YOLO_CLASSES = ['bicycle', 'bridge', 'bus', 'car', 'chimney', 'crosswalk', 'hydrant', 'motorcycle', 'mountain', 'other', 'palm', 'traffic']


HEADLESS = False  
ENABLE_LOGS = True

# Data dir for temporary tiles
data_dir = os.path.join(os.getcwd(), "data")
os.makedirs(data_dir, exist_ok=True)

_yolo_model = None

def get_yolo_model(model_path=MODEL_PATH):
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model

    print("Loading YOLO model from:", model_path)
    _yolo_model = YOLO(model_path)
    return _yolo_model


def predict_tile_yolo(tile_path, target_size=(128,128)):
    model = get_yolo_model()

    # Force YOLO to handle preprocessing internally
    results = model.predict(source=tile_path, imgsz=target_size[0], verbose=False)

    if not results or len(results) < 1:
        raise RuntimeError("YOLO returned no results")

    res = results[0]

    if not hasattr(res, "probs") or res.probs is None:
        raise RuntimeError("result.probs not available. Ensure this is a classification-style model")

    probs = res.probs.data.cpu().numpy() 
    top_idx = int(np.argmax(probs))
    names = list(res.names.values()) if isinstance(res.names, dict) else res.names
    top_name = names[top_idx]

    return probs, top_name, top_idx


COUNT = 0

def js_click(driver, el):
    try:
        driver.execute_script("arguments[0].click();", el)
    except Exception:
        el.click()

def rename_tile(old_path, class_name, count):
    dirn = os.path.dirname(old_path)
    safe_name = class_name.replace(" ", "_")
    newname = f"{safe_name}_tile_{count}.png"
    newpath = os.path.join(dirn, newname)
    try:
        os.replace(old_path, newpath)
    except Exception:
        try:
            os.rename(old_path, newpath)
        except Exception:
            pass
    return newpath


def open_browser():
    """
    Opens Chrome webdriver via webdriver-manager and returns driver.
    """
    options = ChromeOptions()
    if HEADLESS:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1366,768")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-blink-features=AutomationControlled")
    service = ChromeService(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_window_size(1366, 768)
    return driver

def switch_to_challenge_iframe(driver, timeout=6):
    """
    Switches into the recaptcha challenge iframe (the one with images).
    Returns True if successful.
    """
    driver.switch_to.default_content()
    
    candidates = [
        "//iframe[@title='reCAPTCHA']",
        "//iframe[contains(@title,'challenge')]",
        "//iframe[contains(@title,'recaptcha challenge')]",
        "//iframe[contains(@src,'/recaptcha/') or contains(@src,'recaptcha') ]"
    ]
    for xpath in candidates:
        try:
            WebDriverWait(driver, timeout).until(EC.frame_to_be_available_and_switch_to_it((By.XPATH, xpath)))
            # quick check for rc-imageselect
            if len(driver.find_elements(By.ID, "rc-imageselect")) > 0:
                return True
            driver.switch_to.default_content()
        except Exception:
            continue

    # fallback: enumerate iframes and search for the one that contains rc-imageselect
    driver.switch_to.default_content()
    frames = driver.find_elements(By.TAG_NAME, "iframe")
    for f in frames:
        try:
            driver.switch_to.frame(f)
            if len(driver.find_elements(By.ID, "rc-imageselect")) > 0:
                return True
            driver.switch_to.default_content()
        except Exception:
            driver.switch_to.default_content()
            continue

    return False

def _wait_for_tiles(driver, timeout=6):
    """Wait for tiles inside rc-imageselect and return list of WebElements (table td or div wrappers)."""
    try:
        WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.ID, "rc-imageselect")))
    except Exception:
        return []

    # wait for either table td or div tile wrapper
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.XPATH, "//div[@id='rc-imageselect']//table//td | //div[@id='rc-imageselect']//div[contains(@class,'rc-image-tile-wrapper')]"))
        )
    except Exception:
        pass

    tiles = driver.find_elements(By.XPATH, "//div[@id='rc-imageselect']//table//td")
    if not tiles:
        tiles = driver.find_elements(By.XPATH, "//div[@id='rc-imageselect']//div[contains(@class,'rc-image-tile-wrapper')]")
    tiles = [t for t in tiles if t.is_displayed()]
    return tiles

def process_tile_element(tile_el, idx, class_index, driver):
    """
    Screenshot tile_el, run prediction, click if strategy says so.
    Returns True if clicked.
    """
    global COUNT
    print(f"processing tile idx={idx} class_index={class_index}")
    filename = f"tile_{COUNT}.png"
    tile_path = os.path.join(data_dir, filename)
    try:
        tile_el.screenshot(tile_path)
    except Exception as e:
        print("screenshot failed:", e)
        COUNT += 1
        return False

    # predict
    try:
        probs, top_name, top_idx = predict_tile_yolo(tile_path)
    except Exception as e:
        print("prediction error:", e)
        COUNT += 1
        return False

    if isinstance(probs, np.ndarray) and probs.ndim == 2 and probs.shape[0] == 1:
        probs = probs[0]
    elif not isinstance(probs, np.ndarray):
        probs = np.array(probs)


    if class_index is None or class_index < 0 or class_index >= probs.shape[-1]:
        print("invalid class_index:", class_index)
        COUNT += 1
        return False

    current_prob = float(probs[class_index])
    object_name = YOLO_CLASSES[top_idx] if top_idx < len(YOLO_CLASSES) else top_name
    rename_tile(tile_path, object_name, COUNT)
    print(f"{COUNT}: predicted {object_name} top={top_name} idx={top_idx} prob_for_target={current_prob:.4f}")
    COUNT += 1

    if USE_TOP_N_STRATEGY:
        top_n = sorted(range(len(probs)), key=lambda j: probs[j], reverse=True)[:N]
        if class_index in top_n:
            js_click(driver, tile_el)
            return True
    else:
        if current_prob > THRESHOLD:
            js_click(driver, tile_el)
            return True

    return False


log_filename = None
session_folder = None

def save_global_variables():
    global session_folder
    if session_folder is None:
        return
    with open(os.path.join(session_folder, 'global_variables.txt'), 'w') as file:
        file.write(f'CAPTCHA_URL = {CAPTCHA_URL}\n')
        file.write(f'THRESHOLD = {THRESHOLD}\n')
        file.write(f'CLASSES = {CLASSES}\n')
        file.write(f'YOLO_CLASSES = {YOLO_CLASSES}\n')

def log(captcha_type, captcha_object):
    global log_filename, session_folder
    if not ENABLE_LOGS:
        return
    if session_folder is None:
        highest = 0
        for d in os.listdir('.'):
            if d.startswith('Session'):
                try:
                    n = int(d[7:])
                    highest = max(highest, n)
                except:
                    pass
        session_folder = f"Session{highest+1:02}"
        os.makedirs(session_folder, exist_ok=True)
        save_global_variables()
    if log_filename is None:
        highest_log = 0
        for fn in os.listdir(session_folder):
            if fn.startswith('logs_'):
                try:
                    p = int(fn[5:7])
                    highest_log = max(highest_log, p)
                except:
                    pass
        log_filename = os.path.join(session_folder, f"logs_{highest_log+1:02}.csv")
    with open(log_filename, 'a', newline='') as f:
        w = csv.writer(f)
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        w.writerow([ts, captcha_type, captcha_object])


def get_class_index_from_text(captcha_object):
    text = captcha_object.text.lower()
    for i in YOLO_CLASSES:
        if i in text:
            return YOLO_CLASSES.index(i)
    for i in CLASSES:
        if i in text:
            return CLASSES.index(i)
    return None

def captcha_is_solved(driver):
    sleep(1)
    try:
        driver.switch_to.default_content()
        # attempt to find the main widget checkbox iframe and read aria-checked
        iframe = driver.find_element(By.XPATH, "//iframe[contains(@title,'reCAPTCHA')]")
        driver.switch_to.frame(iframe)
        checkbox = driver.find_element(By.XPATH, '//*[@id="recaptcha-anchor"]')
        checked = checkbox.get_attribute('aria-checked') == 'true'
        driver.switch_to.default_content()
        return checked
    except Exception:
        try:
            driver.switch_to.default_content()
        except:
            pass
        return False

def solve_classification_type(driver, dynamic_captcha=False):
    # switch into challenge iframe
    ok = switch_to_challenge_iframe(driver)
    if not ok:
        print("Could not locate challenge iframe")
        return

    try:
        rc = driver.find_element(By.ID, "rc-imageselect")
        captcha_object = rc.find_element(By.TAG_NAME, "strong")
    except Exception as e:
        print("Could not find rc-imageselect or its text:", e)
        return

    class_index = get_class_index_from_text(captcha_object)
    if class_index is None:
        print("Could not map captcha text to class:", captcha_object.text)
        return

    log("classification", captcha_object.text if ENABLE_LOGS else captcha_object.text)

    tiles = _wait_for_tiles(driver, timeout=6)
    if not tiles:
        print("No tiles found inside rc-imageselect")
        return

    tiles = tiles[:9]  # limit to 3x3

    clicked = []
    for idx, tile_el in enumerate(tiles):
        try:
            was_clicked = process_tile_element(tile_el, idx, class_index, driver)
            if was_clicked:
                clicked.append(idx)
        except Exception as e:
            print("Error processing tile", idx, e)

    # quick dynamic re-check if requested
    if dynamic_captcha and clicked:
        for _ in range(2):
            tiles = _wait_for_tiles(driver, timeout=4)[:9]
            new_clicked = []
            for idx in clicked:
                if idx < len(tiles):
                    try:
                        if process_tile_element(tiles[idx], idx, class_index, driver):
                            new_clicked.append(idx)
                    except Exception as e:
                        print("Dynamic re-check error:", e)
            if set(new_clicked) == set(clicked):
                break
            clicked = new_clicked

    # click verify button
    try:
        verify_btn = driver.find_element(By.ID, "recaptcha-verify-button")
        js_click(driver, verify_btn)
        sleep(1)
    except Exception as e:
        print("Could not click verify:", e)

def run():
    # preload model
    try:
        get_yolo_model()
    except Exception as e:
        print("Failed to load YOLO model:", e)
        return

    driver = None
    try:
        driver = open_browser()
        print("Opened browser, navigating to captcha demo")
        driver.get(CAPTCHA_URL)
        WebDriverWait(driver, 10).until(lambda d: d.execute_script('return document.readyState') == 'complete')
        sleep(1)

        # click the checkbox to open challenge
        # locate the outer recaptcha iframe and click inside
        outer_iframe = WebDriverWait(driver, 8).until(EC.presence_of_element_located((By.XPATH, "//iframe[contains(@title,'reCAPTCHA')]")))
        driver.switch_to.frame(outer_iframe)
        checkbox = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.CLASS_NAME, "recaptcha-checkbox-border")))
        js_click(driver, checkbox)
        driver.switch_to.default_content()
        print("Opened the captcha challenge")
        # allow time for challenge iframe to appear
        sleep(2)

        # main loop
        while True:
            # attempt to switch into challenge iframe and read the rc-imageselect text (if present)
            if not switch_to_challenge_iframe(driver):
                print("Challenge iframe not available yet. Waiting and retrying.")
                sleep(1)
                continue

            try:
                rc = driver.find_element(By.ID, "rc-imageselect")
                text = rc.text.lower()
            except Exception:
                print("rc-imageselect not found after switching frame; retrying.")
                sleep(1)
                continue

            if "squares" in text:
                print("Found segmentation (4x4) challenge; this script skips segmentation. Reloading.")
                try:
                    driver.find_element(By.ID, "recaptcha-reload-button").click()
                except Exception:
                    pass
                sleep(1)
                continue
            elif "none" in text:
                print("Found dynamic 3x3 captcha")
                solve_classification_type(driver, dynamic_captcha=True)
            else:
                print("Found 3x3 one-time selection captcha (classification)")
                solve_classification_type(driver, dynamic_captcha=False)

            # check solved
            if captcha_is_solved(driver):
                print("Captcha solved!")
                log("SOLVED", "captcha solved")
                break
            else:
                print("Captcha not solved yet; reloading challenge")
                try:
                    driver.find_element(By.ID, "recaptcha-reload-button").click()
                except Exception:
                    pass
                sleep(1)
                continue

    except Exception as e:
        print("Fatal error in run:", e)
        traceback.print_exc()
    finally:
        try:
            if driver:
                driver.quit()
        except Exception:
            pass

if __name__ == "__main__":
    run()