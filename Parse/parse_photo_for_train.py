from icrawler.builtin import GoogleImageCrawler

google_crawler = GoogleImageCrawler(
    feeder_threads=1,
    parser_threads=1,
    downloader_threads=4,
    storage={'root_dir': 'images'}
)

filters = dict(
    size='large',
    color='green',
    license='commercial,modify',
    date=((2022, 1, 1), None),
    type='photo'
)

google_crawler.crawl(keyword='images', filters=filters, max_num=1)