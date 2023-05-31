## Для работы кода из файла .py необходимо установить библиотеку torch командой:

```
pip install torch
```
---

# Использование:

#### аргументы:
`-text "<Текс для озвучивания>"` передается текст, который необходимо озвучить (обязательный аргумент)

`-voice <голос>` используется для выбора голоса озвучивания (Доступные голоса [`aidar`, `baya`, `kseniya`, `xenia`, `eugene`, `random`]. По умолчанию используется `kseniya`)

`-rate <rate>` Качество звука (Доступно `8000`, `24000`, `48000`. По умолчанию `48000`)

`-pitch <аргумент>` Установка тона. По умолчанию medium (Доступно [`x-low`, `low`, `medium`, `high`, `x-high`])

`-speed <аргумент>` Установка скорости озвчивания. По умолчанию medium (Доступно [`x-slow`, `slow`, `medium`, `fast`, `x-fast`])

`-name <Имя>` Имя выходного файла (По умолчанию будет назван по текущему внеремни и дате)

---

# Сборка в .exe

При сборке в один файл программа работает медленнее. Чтобы собрать спользуется комманда:

```
pyinstaller -F --hidden-import wave main.py
```

Сборка в одну директорию работает быстрее. Комманда для сборки в одну директорию:

```
pyinstaller -D --hidden-import wave main.py
```

Для сборки необходимо установить pyinstaller:

```
pip install pyinstaller
```
