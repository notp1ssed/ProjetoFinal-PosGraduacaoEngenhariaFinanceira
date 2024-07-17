from playwright.sync_api import sync_playwright
import json
import csv
from datetime import datetime

def convert_timestamp(timestamp):
    return datetime.utcfromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')

def scrape_highchart_data():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        #page.goto("https://tradingeconomics.com/united-states/manufacturing-pmi")
        #page.goto("https://tradingeconomics.com/euro-area/inflation-cpi")
        #page.goto("https://tradingeconomics.com/japan/inflation-cpi")
        page.goto("https://tradingeconomics.com/united-states/central-bank-balance-sheet")

        page.wait_for_selector('#UpdatePanelChart', state='visible') # '#UpdatePanelChart' "#bstContainer"
        page.wait_for_timeout(10000)  # Wait for 10 seconds
        
        chart_data = page.evaluate('''() => {
            if (typeof Highcharts === 'undefined') return 'Highcharts not found';
            const charts = Highcharts.charts.filter(c => c);
            if (charts.length === 0) return 'No charts found';
            return charts.map(chart => ({
                id: chart.renderTo.id,
                data: chart.series[0] ? chart.series[0].options.data : 'No data'
            }));
        }''')
        
        browser.close()
        return chart_data

def save_to_csv(data, filename='usa_balance-sheet.csv'):
    if not data or not isinstance(data, list) or not data[0]['data']:
        print("No valid data to save")
        return

    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['date', 'value']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for item in data[0]['data']:
            writer.writerow({
                'date': item['date'],
                'value': item['y']
            })

if __name__ == "__main__":
    data = scrape_highchart_data()
    if isinstance(data, list) and data:
        save_to_csv(data)
        print(f"Data saved to chart_data.csv")
    else:
        print("Failed to retrieve chart data")