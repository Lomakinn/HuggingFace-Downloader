# HuggingFace Downloader v1.01

Desktop downloader for Hugging Face repositories with resumable downloads, history, per-file progress, filters, sorting, themes, and EN/RU interface.

## Features

- Search Hugging Face models, datasets, and Spaces.
- Download selected files or all visible files.
- Resume interrupted or canceled downloads.
- Expand a repository job to see every file with its own progress and speed.
- Track total progress, current file, status, size, speed, and target folder.
- Filter history by status and sort by status, size, or progress.
- Open repository pages and download folders from the app.
- Choose and save the default download folder.
- Save settings, history, theme, language, filters, sort order, window layout, and column widths between sessions.
- Light, dark, and system themes.
- English by default, Russian optional.

Settings and download history are stored in:

```text
%APPDATA%\HuggingFace Downloader\settings.json
```

## Run

Double-click `run.bat`, or run:

```bat
run.bat
```

The launcher creates a local virtual environment and installs dependencies on first start.
