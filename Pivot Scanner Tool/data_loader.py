import yfinance as yf


class DataLoader:

    def manual_download(self):
        ticker = input("Enter YFinance Ticker: ")
        start_date = input("Enter Data Start Date (YYYY-MM-DD): ")
        end_date = input("Enter Data End Date (YYYY-MM-DD): ")
        interval = input("Enter Data Interval (1m, 5m, 15m, etc.): ")
        filename = self.download(ticker, start_date, end_date, interval)

        return filename
    
    def download(self, ticker, start_date, end_date, interval):
        df = yf.download(ticker, interval=interval, start=start_date, end=end_date)
        filename = './Pivot Scanner Tool/Market Data/' + ticker + '_' + 'start_date' + '_' + end_date + '_' + interval + '.csv'
        df.to_csv(filename)

def main():
    dl = DataLoader()
    dl.manual_download()

if __name__ == "__main__":
    main()