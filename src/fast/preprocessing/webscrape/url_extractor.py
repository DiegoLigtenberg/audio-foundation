import csv
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pickle

def fetch_youtube_videos(queries=None, num_videos=100):
    # Set up Chrome WebDriver with ChromeDriverManager
    options = webdriver.ChromeOptions()
    options.add_argument('--disable-gpu')  # Disable GPU to avoid GPU state errors
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    query_index = 0
    query = queries[query_index]
    output_file = f"database/urls/music_videos_{query[:query.rfind(' music')] if query.endswith(' music') else query}.csv".replace(' ', '_')# Initial output file
    start_time = time.time()
    all_urls = set()  # Track unique URLs for each query's file
    videos_collected = 0  # Total count of collected videos
    no_new_videos_count = 0  # Track number of consecutive scrolls with no new videos
    writer = None  # To hold the CSV writer
    f = None  # File handle for the output file

    try:
        # Open the YouTube search results for the initial query
        url = f"https://www.youtube.com/results?search_query={query}"
        driver.get(url)
        time.sleep(3)

        # Check for cookie consent
        try:
            accept_button = driver.find_element(By.XPATH, '//button[text()="Alles accepteren"]')
            accept_button.click()
            time.sleep(2)
        except Exception:
            pass  # No consent button found

        # Open a file for the current query to store the video data
        f = open(output_file, 'a', newline='', encoding='utf-8')
        writer = csv.DictWriter(f, fieldnames=['url', 'title'])
        if f.tell() == 0:  # Write header if the file is empty
            writer.writeheader()

        while videos_collected < num_videos:
            # Scroll down to load more videos
            driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
            time.sleep(2)

            # Wait for video elements to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "video-title"))
            )

            # Collect video elements
            video_elements = driver.find_elements(By.ID, "video-title")

            new_videos_found = False  # Flag to track if new videos are added

            for video in video_elements:
                video_url = video.get_attribute('href')
                title = video.get_attribute('title')

                # Check for unique URLs within this output file
                if video_url and "/watch?" in video_url and video_url not in all_urls:
                    writer.writerow({'url': video_url, 'title': title})
                    all_urls.add(video_url)
                    videos_collected += 1
                    print(f"Collected video: {title}, URL: {video_url}")
                    new_videos_found = True

                    # Flush to ensure data is written to disk immediately
                    f.flush()

                # Exit loop if limit is reached
                if videos_collected >= num_videos:
                    break

            # If no new videos were found after this scroll, increment the no_new_videos_count
            if not new_videos_found:
                no_new_videos_count += 1
            else:
                no_new_videos_count = 0  # Reset count if new videos are found

            # Exit loop if the limit is reached
            if videos_collected >= num_videos:
                break

            # If no new videos after 5 consecutive scrolls, switch query/output file
            if no_new_videos_count >= 5:
                print("No new videos found after 5 scrolls. Switching to next query/output file.")
                # Close current output file before switching
                f.close()

                query_index = (query_index + 1) % len(queries)
                query = queries[query_index]
                output_file = f"database/urls/music_videos_{query[:query.rfind(' music')] if query.endswith(' music') else query}.csv".replace(' ', '_')
                start_time = time.time()  # Reset time for the new query
                all_urls.clear()  # Clear the set of URLs to avoid duplicates across queries

                driver.get(f"https://www.youtube.com/results?search_query={query}")
                time.sleep(3)

                # Check for cookie consent on the new query page
                try:
                    accept_button = driver.find_element(By.XPATH, '//button[text()="Alles accepteren"]')
                    accept_button.click()
                    time.sleep(2)
                except Exception:
                    pass  # No consent button found

                # Open the new file for the new query
                f = open(output_file, 'a', newline='', encoding='utf-8')
                writer = csv.DictWriter(f, fieldnames=['url', 'title'])
                if f.tell() == 0:  # Write header if the file is empty
                    writer.writeheader()

            # Switch to a new query every 5 minutes
            if time.time() - start_time >= 300:
                print("5 minutes have passed. Switching to next query/output file.")
                # Close the current file before switching
                f.close()

                query_index = (query_index + 1) % len(queries)
                query = queries[query_index]
                output_file = f"urls/music_videos_{query[:query.rfind(' music')] if query.endswith(' music') else query}.csv".replace(' ', '_')
                start_time = time.time()  # Reset time for the new query
                all_urls.clear()  # Clear URLs to avoid duplicates across queries
                driver.get(f"https://www.youtube.com/results?search_query={query}")
                time.sleep(3)

                # Check for cookie consent
                try:
                    accept_button = driver.find_element(By.XPATH, '//button[text()="Alles accepteren"]')
                    accept_button.click()
                    time.sleep(2)
                except Exception:
                    pass  # No consent button found

                # Open the new file for the new query
                f = open(output_file, 'a', newline='', encoding='utf-8')
                writer = csv.DictWriter(f, fieldnames=['url', 'title'])
                if f.tell() == 0:  # Write header if the file is empty
                    writer.writeheader()

    except Exception as e:
        print(f"Error occurred: {e}")
    
    finally:
        # Ensure the file is closed when the program ends
        if f:
            f.close()
    
    driver.quit()
    print(f"Collected metadata for {videos_collected} unique videos.")
    return videos_collected



# Step 1: Open the file in read-binary mode
with open(r'src\fast\preprocessing\webscrape\lists_and_save_files\ordered_genres2.pkl', 'rb') as file:
    # Step 2: Load the serialized list
    loaded_genres = pickle.load(file)

# Step 3: Print the loaded list
print("Loaded list:")
print(len(loaded_genres))


with open(r'src\fast\preprocessing\webscrape\lists_and_save_files\ordered_genres2.pkl', 'rb') as file:
    # Step 2: Load the serialized list
    queries = pickle.load(file)
    queries = [query.replace(" /", "").replace("/","").lower() + " music" for query in queries]

from pathlib import Path 
start_from = len([f for f in Path('database/urls').iterdir() if f.is_file()])
# print(queries[start_from-1:])
print(start_from)
# asd
# Fetch video metadata
video_metadata = fetch_youtube_videos(queries=queries[start_from:], num_videos=10000000)
