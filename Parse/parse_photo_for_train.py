from icrawler.builtin import GoogleImageCrawler

google_crawler = GoogleImageCrawler(
    feeder_threads=1,      # количество потоков для загрузки ссылок на изображения
    parser_threads=1,      # количество потоков для парсинга страниц с изображениями
    downloader_threads=4,  # количество потоков для загрузки изображений
    storage={'root_dir': 'images'}  # путь для сохранения скачанных изображений
)

# задаем фильтры для поиска изображений
filters = dict(
    size='medium',                # размер изображения
    license='commercial,modify', # тип лицензии
    date=((2010, 1, 1), (2023, 1, 1)),  # дата публикации изображения
    type='photo'                 # тип изображения
)

# вызываем метод crawl объекта GoogleImageCrawler для поиска и скачивания изображений
google_crawler.crawl(
    keyword='hedgehog',        # ключевое слово для поиска
    filters=filters,      # фильтры для поиска
    max_num=2,            # максимальное количество изображений для скачивания
    file_idx_offset=0    # начальный индекс для названий файлов
)
