import json
import os
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests
from huggingface_hub import HfApi, get_token, hf_hub_url
from PySide6.QtCore import QByteArray, QObject, Qt, QUrl, Signal
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QFont, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


APP_NAME = "HuggingFace Downloader v1.01"
APP_DIR = Path(__file__).resolve().parent
LEGACY_SETTINGS_PATH = APP_DIR / "hf_downloader_settings.json"
USER_DATA_DIR = Path(
    os.environ.get("APPDATA")
    or os.environ.get("LOCALAPPDATA")
    or Path.home()
) / "HuggingFace Downloader"
SETTINGS_PATH = USER_DATA_DIR / "settings.json"
APP_VERSION = 2
JOBS_COLUMN_DEFAULT_WIDTHS = [44, 280, 120, 120, 260, 170, 120, 280, 430]
JOBS_COLUMN_MIN_WIDTHS = [44, 180, 96, 100, 160, 130, 90, 180, 400]

STATUS_QUEUED = "Queued"
STATUS_DOWNLOADING = "Downloading"
STATUS_INTERRUPTED = "Interrupted"
STATUS_FAILED = "Failed"
STATUS_VERIFIED = "Verified"
STATUS_CANCELED = "Canceled"

FILTERS = {
    "All files": None,
    "Weights": (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf"),
    "SafeTensors": (".safetensors",),
    "GGUF": (".gguf",),
    "BIN": (".bin",),
    "Tokenizer / config": (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ),
}


TRANSLATIONS = {
    "EN": {
        "repository_group": "Repository",
        "search": "Search",
        "settings": "Settings",
        "save": "Save",
        "type": "Type",
        "revision": "Revision",
        "folder": "Folder",
        "theme": "Theme",
        "language": "Language",
        "retries": "Retries",
        "timeout": "Timeout",
        "choose": "Choose",
        "open": "Open",
        "use_hf_token": "Use HF token",
        "no_repository": "No repository selected",
        "filter": "Filter",
        "select_all": "Select All",
        "clear_selection": "Clear Selection",
        "refresh_local": "Refresh Local",
        "download_checked": "Download Checked",
        "download_visible": "Download Visible",
        "downloads_history": "Downloads / History",
        "status_filter": "Status",
        "status_all": "All",
        "status_downloading": "Downloading",
        "status_downloaded": "Downloaded",
        "status_error": "Error",
        "status_verified": "Verified",
        "ready": "Ready",
        "files_file": "File",
        "files_size": "Size",
        "files_local": "Local",
        "jobs_repository": "Repository",
        "jobs_status": "Status",
        "jobs_progress": "Progress",
        "jobs_current_file": "Current file",
        "jobs_size": "Size",
        "jobs_speed": "Speed",
        "jobs_folder": "Folder",
        "jobs_actions": "Actions",
        "show": "Show",
        "hide": "Hide",
        "resume": "Resume",
        "cancel": "Cancel",
        "remove": "Remove",
        "open_repo_tooltip": "Open repository in browser",
        "enter_repo": "Enter a model name or full repo id.",
        "check_file": "Check at least one file.",
        "no_visible_files": "No visible files to download.",
        "load_repo_first": "Load a repository first.",
        "choose_download_folder": "Choose download folder",
    },
    "RU": {
        "repository_group": "Репозиторий",
        "search": "Поиск",
        "settings": "Настройки",
        "save": "Сохранить",
        "type": "Тип",
        "revision": "Ревизия",
        "folder": "Папка",
        "theme": "Тема",
        "language": "Язык",
        "retries": "Повторы",
        "timeout": "Таймаут",
        "choose": "Выбрать",
        "open": "Открыть",
        "use_hf_token": "Использовать HF token",
        "no_repository": "Репозиторий не выбран",
        "filter": "Фильтр",
        "select_all": "Выбрать все",
        "clear_selection": "Снять выбор",
        "refresh_local": "Обновить локально",
        "download_checked": "Скачать выбранное",
        "download_visible": "Скачать видимое",
        "downloads_history": "Загрузки / История",
        "status_filter": "Статус",
        "status_all": "Все",
        "status_downloading": "Скачивается",
        "status_downloaded": "Скачался",
        "status_error": "Ошибка",
        "status_verified": "Проверено",
        "ready": "Готово",
        "files_file": "Файл",
        "files_size": "Размер",
        "files_local": "Локально",
        "jobs_repository": "Репозиторий",
        "jobs_status": "Статус",
        "jobs_progress": "Прогресс",
        "jobs_current_file": "Текущий файл",
        "jobs_size": "Размер",
        "jobs_speed": "Скорость",
        "jobs_folder": "Папка",
        "jobs_actions": "Действия",
        "show": "Показать",
        "hide": "Скрыть",
        "resume": "Продолжить",
        "cancel": "Отмена",
        "remove": "Удалить",
        "open_repo_tooltip": "Открыть репозиторий в браузере",
        "enter_repo": "Введите имя модели или полный repo id.",
        "check_file": "Отметьте хотя бы один файл.",
        "no_visible_files": "Нет видимых файлов для скачивания.",
        "load_repo_first": "Сначала загрузите репозиторий.",
        "choose_download_folder": "Выберите папку загрузки",
    },
}


def create_app_icon() -> QIcon:
    pixmap = QPixmap(96, 96)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setBrush(QColor("#ffcc4d"))
    painter.setPen(QPen(QColor("#b87900"), 3))
    painter.drawRoundedRect(8, 8, 80, 80, 20, 20)
    painter.setBrush(QColor("#202124"))
    painter.setPen(Qt.NoPen)
    painter.drawEllipse(26, 32, 10, 10)
    painter.drawEllipse(60, 32, 10, 10)
    painter.setPen(QPen(QColor("#202124"), 5))
    painter.drawArc(31, 43, 34, 24, 200 * 16, 140 * 16)
    painter.setBrush(QColor("#7c4dff"))
    painter.setPen(QPen(QColor("#ffffff"), 3))
    painter.drawEllipse(55, 58, 24, 24)
    painter.setPen(QPen(QColor("#ffffff"), 4))
    painter.drawLine(67, 63, 67, 76)
    painter.drawLine(61, 70, 67, 76)
    painter.drawLine(73, 70, 67, 76)
    painter.end()
    return QIcon(pixmap)


@dataclass
class RepoFile:
    name: str
    size: int = 0
    local_state: str = "Missing"
    local_size: int = 0


def human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    value = float(num_bytes)
    for unit in ["KiB", "MiB", "GiB", "TiB"]:
        value /= 1024.0
        if value < 1024:
            return f"{value:.2f} {unit}"
    return f"{value:.2f} PiB"


def human_speed(bytes_per_second: float) -> str:
    if not bytes_per_second:
        return "-"
    return f"{human_size(int(bytes_per_second))}/s"


def load_settings() -> Dict:
    data = read_settings_file(SETTINGS_PATH)
    local_data = read_settings_file(LEGACY_SETTINGS_PATH)
    if local_data is not None and local_data.get("jobs") and not (data or {}).get("jobs"):
        data = local_data
        data["migrated_from"] = str(LEGACY_SETTINGS_PATH)
        save_settings(data)

    if data is None:
        data = {}
    data["version"] = APP_VERSION
    return data


def read_settings_file(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def write_settings_payload(path: Path, payload: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    try:
        tmp_path.replace(path)
    except PermissionError:
        try:
            path.write_text(payload, encoding="utf-8")
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)


def save_settings(settings: Dict):
    settings["version"] = APP_VERSION
    payload = json.dumps(settings, indent=2, ensure_ascii=False)
    try:
        write_settings_payload(SETTINGS_PATH, payload)
    except OSError:
        try:
            write_settings_payload(LEGACY_SETTINGS_PATH, payload)
        except OSError:
            pass


def repo_type_arg(repo_type: str) -> Optional[str]:
    return None if repo_type == "model" else repo_type


def make_file_url(repo_id: str, filename: str, repo_type: str, revision: str) -> str:
    return hf_hub_url(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type_arg(repo_type),
        revision=revision or "main",
    )


def make_repo_url(repo_id: str, repo_type: str) -> str:
    if repo_type == "dataset":
        return f"https://huggingface.co/datasets/{repo_id}"
    if repo_type == "space":
        return f"https://huggingface.co/spaces/{repo_id}"
    return f"https://huggingface.co/{repo_id}"


def auth_headers(use_token: bool) -> Dict[str, str]:
    if not use_token:
        return {}
    token = get_token()
    if not token:
        raise RuntimeError("Hugging Face token is not available. Run `hf auth login` first.")
    return {"Authorization": f"Bearer {token}"}


def local_state(path: Path, expected_size: int) -> tuple[str, int]:
    if not path.exists():
        return "Missing", 0
    size = path.stat().st_size
    if expected_size <= 0:
        return "Present", size
    if size == expected_size:
        return "Ready", size
    if 0 < size < expected_size:
        return "Partial", size
    return "Mismatch", size


class DownloadCanceled(Exception):
    pass


class Signals(QObject):
    error = Signal(str)
    status = Signal(str)
    search_results = Signal(list)
    files_ready = Signal(list)
    job_update = Signal(str, dict)
    job_finished = Signal(str)


def stream_file(
    url: str,
    dest_path: Path,
    expected_size: int,
    headers: Dict[str, str],
    timeout: int,
    cancel_event: threading.Event,
    progress_callback,
):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    current_size = dest_path.stat().st_size if dest_path.exists() else 0

    if expected_size > 0 and current_size == expected_size:
        progress_callback(expected_size, expected_size)
        return expected_size

    if expected_size > 0 and current_size > expected_size:
        dest_path.unlink()
        current_size = 0

    request_headers = dict(headers)
    resume = current_size > 0
    if resume:
        request_headers["Range"] = f"bytes={current_size}-"

    with requests.get(
        url,
        headers=request_headers,
        stream=True,
        allow_redirects=True,
        timeout=timeout,
    ) as response:
        response.raise_for_status()

        if resume and response.status_code == 206:
            total = expected_size or current_size + int(response.headers.get("content-length", 0))
            mode = "ab"
            downloaded = current_size
        else:
            total = expected_size or int(response.headers.get("content-length", 0))
            mode = "wb"
            downloaded = 0

        with dest_path.open(mode) as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if cancel_event.is_set():
                    raise DownloadCanceled()
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                progress_callback(downloaded, total)

    if expected_size > 0 and dest_path.stat().st_size != expected_size:
        raise RuntimeError(
            f"Size check failed for {dest_path.name}: "
            f"{dest_path.stat().st_size} of {expected_size} bytes"
        )
    progress_callback(dest_path.stat().st_size, expected_size or dest_path.stat().st_size)
    return dest_path.stat().st_size


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(create_app_icon())
        self.setMinimumSize(1180, 760)

        self.api = HfApi()
        self.signals = Signals()
        self.settings = load_settings()
        self.language = self.settings_language_default()
        self.jobs: List[Dict] = self.settings.get("jobs", [])
        self.expanded_jobs = set(self.settings.get("expanded_jobs", []))
        self.current_files: List[RepoFile] = []
        self.checked_file_names = set()
        self.selection_initialized = False
        self.current_repo_id = ""
        self.active_job_id: Optional[str] = None
        self.cancel_event = threading.Event()
        self._last_saved_progress: Dict[str, int] = {}
        self.jobs_sort_column = self.settings.get("jobs_sort_column")
        self.jobs_sort_ascending = bool(self.settings.get("jobs_sort_ascending", True))
        self._restoring_jobs_column_widths = False

        self._normalize_jobs_on_start()
        self._build_ui()
        self._connect_signals()
        self._restore_window_state()
        self._apply_theme(self.settings.get("theme", "System"))
        self.apply_language()
        self._refresh_jobs_table()

    def settings_language_default(self) -> str:
        language = self.settings.get("language", "EN")
        return language if language in TRANSLATIONS else "EN"

    def t(self, key: str) -> str:
        return TRANSLATIONS.get(self.language, TRANSLATIONS["EN"]).get(key, key)

    def _build_ui(self):
        self.setFont(QFont("", 10))

        root = QVBoxLayout()
        root.addWidget(self._build_top_panel())

        self.main_splitter = QSplitter(Qt.Vertical)
        self.main_splitter.addWidget(self._build_files_panel())
        self.main_splitter.addWidget(self._build_jobs_panel())
        self.main_splitter.setSizes(self.settings.get("splitter_sizes", [420, 300]))
        root.addWidget(self.main_splitter, 1)

        self.status_label = QLabel(self.t("ready"))
        root.addWidget(self.status_label)
        self.setLayout(root)

    def _build_top_panel(self):
        self.repository_group = QGroupBox()
        layout = QGridLayout()
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        layout.setColumnStretch(3, 1)
        layout.setColumnStretch(4, 1)
        layout.setColumnStretch(5, 0)
        layout.setColumnStretch(6, 0)
        layout.setColumnMinimumWidth(0, 1)

        self.query_input = QLineEdit()
        self.query_input.setText(self.settings.get("last_query", ""))
        self.query_input.setPlaceholderText("model name or org/repo")
        self.search_btn = QPushButton()
        self.settings_btn = QPushButton()
        self.repo_combo = QComboBox()
        self.repo_combo.setVisible(False)

        self.repo_type_combo = QComboBox()
        self.repo_type_combo.addItems(["model", "dataset", "space"])
        self.repo_type_combo.setCurrentText(self.settings.get("repo_type", "model"))
        self.revision_input = QLineEdit(self.settings.get("revision", "main"))
        self.use_token_check = QCheckBox("Use HF token")
        self.use_token_check.setChecked(bool(self.settings.get("use_token", False)))

        self.folder_input = QLineEdit(
            self.settings.get("download_dir")
            or str(Path.home() / "HF-Downloads")
        )
        self.choose_folder_btn = QPushButton()
        self.open_folder_btn = QPushButton()

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["System", "Light", "Dark"])
        self.theme_combo.setCurrentText(self.settings.get("theme", "System"))

        self.language_combo = QComboBox()
        self.language_combo.addItems(["EN", "RU"])
        self.language_combo.setCurrentText(self.language)

        self.retry_spin = QSpinBox()
        self.retry_spin.setRange(0, 10)
        self.retry_spin.setValue(int(self.settings.get("retries", 2)))
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(5, 300)
        self.timeout_spin.setValue(int(self.settings.get("timeout", 30)))

        self.search_label = QLabel()
        self.type_label = QLabel()
        self.revision_label = QLabel()
        self.folder_label = QLabel()
        self.theme_label = QLabel()
        self.language_label = QLabel()
        self.retries_label = QLabel()
        self.timeout_label = QLabel()

        layout.addWidget(self.search_label, 0, 0)
        layout.addWidget(self.query_input, 0, 1, 1, 4)
        layout.addWidget(self.search_btn, 0, 5)
        layout.addWidget(self.settings_btn, 0, 6)
        layout.addWidget(self.repo_combo, 1, 1, 1, 6)

        self.repository_group.setLayout(layout)
        return self.repository_group

    def _build_files_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()

        bar = QHBoxLayout()
        self.repo_label = QLabel()
        self.file_filter_combo = QComboBox()
        self.file_filter_combo.addItems(list(FILTERS.keys()))
        self.file_filter_combo.setCurrentText(self.settings.get("file_filter", "All files"))
        self.select_all_btn = QPushButton()
        self.clear_selection_btn = QPushButton()
        self.refresh_local_btn = QPushButton()
        self.download_selected_btn = QPushButton()
        self.download_visible_btn = QPushButton()
        self.download_selected_btn.setEnabled(False)
        self.download_visible_btn.setEnabled(False)

        bar.addWidget(self.repo_label, 1)
        self.filter_label = QLabel()
        bar.addWidget(self.filter_label)
        bar.addWidget(self.file_filter_combo)
        bar.addWidget(self.select_all_btn)
        bar.addWidget(self.clear_selection_btn)
        bar.addWidget(self.refresh_local_btn)
        bar.addWidget(self.download_selected_btn)
        bar.addWidget(self.download_visible_btn)
        layout.addLayout(bar)

        self.files_table = QTableWidget(0, 4)
        self.files_table.setHorizontalHeaderLabels(["", "", "", ""])
        header = self.files_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.files_table.setAlternatingRowColors(True)
        self.files_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.files_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.files_table)

        panel.setLayout(layout)
        return panel

    def _build_jobs_panel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        title = QHBoxLayout()
        self.downloads_history_label = QLabel()
        title.addWidget(self.downloads_history_label, 1)
        self.job_status_filter_label = QLabel()
        self.job_status_filter_combo = QComboBox()
        self.job_status_filter_combo.addItems(["all", "downloading", "downloaded", "error", "verified"])
        self.job_status_filter_combo.setCurrentText(self.settings.get("job_status_filter", "all"))
        title.addWidget(self.job_status_filter_label)
        title.addWidget(self.job_status_filter_combo)
        layout.addLayout(title)

        self.jobs_table = QTableWidget(0, 9)
        self.jobs_table.setHorizontalHeaderLabels(["", "", "", "", "", "", "", "", ""])
        self.jobs_table.verticalHeader().setVisible(False)
        header = self.jobs_table.horizontalHeader()
        header.setSectionsMovable(False)
        header.setMinimumSectionSize(44)
        for column in range(self.jobs_table.columnCount()):
            header.setSectionResizeMode(column, QHeaderView.Interactive)
        self._restore_jobs_column_widths()
        self.jobs_table.setAlternatingRowColors(True)
        self.jobs_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.jobs_table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.jobs_table)

        panel.setLayout(layout)
        return panel

    def _connect_signals(self):
        self.search_btn.clicked.connect(self.search_repo)
        self.query_input.returnPressed.connect(self.search_repo)
        self.query_input.editingFinished.connect(self._persist)
        self.settings_btn.clicked.connect(self.open_settings_dialog)
        self.repo_combo.currentIndexChanged.connect(self._repo_combo_changed)
        self.file_filter_combo.currentTextChanged.connect(self._file_filter_changed)
        self.job_status_filter_combo.currentTextChanged.connect(self._job_status_filter_changed)
        self.select_all_btn.clicked.connect(self.select_all_visible_files)
        self.clear_selection_btn.clicked.connect(self.clear_visible_selection)
        self.refresh_local_btn.clicked.connect(self.refresh_local_states)
        self.download_selected_btn.clicked.connect(self.download_checked_files)
        self.download_visible_btn.clicked.connect(self.download_visible_files)
        self.jobs_table.cellClicked.connect(self._jobs_table_clicked)
        self.jobs_table.horizontalHeader().sectionClicked.connect(self._jobs_header_clicked)
        self.jobs_table.horizontalHeader().sectionResized.connect(self._jobs_column_resized)

        self.signals.error.connect(self.show_error)
        self.signals.status.connect(self.status_label.setText)
        self.signals.search_results.connect(self._show_search_results)
        self.signals.files_ready.connect(self._set_repo_files)
        self.signals.job_update.connect(self._update_job)
        self.signals.job_finished.connect(self._job_finished)

    def _file_filter_changed(self, _text: str):
        self._persist()
        self._refresh_files_table()

    def _job_status_filter_changed(self, _text: str):
        self._persist()
        self._refresh_jobs_table()

    def current_job_status_filter(self) -> str:
        if not hasattr(self, "job_status_filter_combo"):
            return "all"
        value = self.job_status_filter_combo.currentData()
        return value or self.job_status_filter_combo.currentText() or "all"

    def _jobs_header_clicked(self, section: int):
        sort_map = {2: "status", 3: "progress", 5: "size"}
        sort_column = sort_map.get(section)
        if not sort_column:
            return
        if self.jobs_sort_column == sort_column:
            self.jobs_sort_ascending = not self.jobs_sort_ascending
        else:
            self.jobs_sort_column = sort_column
            self.jobs_sort_ascending = True
        self._persist()
        self._refresh_jobs_table()

    def _jobs_column_resized(self, _section: int, _old_size: int, _new_size: int):
        if self._restoring_jobs_column_widths:
            return
        self._enforce_jobs_column_min_widths()
        self._persist()

    def _restore_jobs_column_widths(self):
        widths = self.settings.get("jobs_column_widths", JOBS_COLUMN_DEFAULT_WIDTHS)
        self._restoring_jobs_column_widths = True
        try:
            for column, width in enumerate(widths[: self.jobs_table.columnCount()]):
                minimum = JOBS_COLUMN_MIN_WIDTHS[column]
                default = JOBS_COLUMN_DEFAULT_WIDTHS[column]
                self.jobs_table.setColumnWidth(column, max(minimum, int(width or default)))
        finally:
            self._restoring_jobs_column_widths = False

    def _enforce_jobs_column_min_widths(self):
        self._restoring_jobs_column_widths = True
        try:
            for column, minimum in enumerate(JOBS_COLUMN_MIN_WIDTHS[: self.jobs_table.columnCount()]):
                if self.jobs_table.columnWidth(column) < minimum:
                    self.jobs_table.setColumnWidth(column, minimum)
        finally:
            self._restoring_jobs_column_widths = False

    def open_settings_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle(self.t("settings"))
        layout = QGridLayout()

        repo_type_combo = QComboBox()
        repo_type_combo.addItems(["model", "dataset", "space"])
        repo_type_combo.setCurrentText(self.repo_type_combo.currentText())

        revision_input = QLineEdit(self.revision_input.text())
        use_token_check = QCheckBox(self.t("use_hf_token"))
        use_token_check.setChecked(self.use_token_check.isChecked())

        folder_input = QLineEdit(self.folder_input.text())
        choose_btn = QPushButton(self.t("choose"))
        open_btn = QPushButton(self.t("open"))

        theme_combo = QComboBox()
        theme_combo.addItems(["System", "Light", "Dark"])
        theme_combo.setCurrentText(self.theme_combo.currentText())

        language_combo = QComboBox()
        language_combo.addItems(["EN", "RU"])
        language_combo.setCurrentText(self.language_combo.currentText())

        retry_spin = QSpinBox()
        retry_spin.setRange(0, 10)
        retry_spin.setValue(self.retry_spin.value())

        timeout_spin = QSpinBox()
        timeout_spin.setRange(5, 300)
        timeout_spin.setValue(self.timeout_spin.value())

        save_btn = QPushButton(self.t("save"))

        def choose_folder_for_dialog():
            folder = QFileDialog.getExistingDirectory(
                dialog,
                self.t("choose_download_folder"),
                folder_input.text(),
            )
            if folder:
                folder_input.setText(folder)

        def open_folder_for_dialog():
            self.open_folder(Path(folder_input.text()))

        def save_dialog_settings():
            self.repo_type_combo.setCurrentText(repo_type_combo.currentText())
            self.revision_input.setText(revision_input.text().strip() or "main")
            self.use_token_check.setChecked(use_token_check.isChecked())
            self.folder_input.setText(str(Path(folder_input.text().strip()).expanduser()))
            self.theme_combo.setCurrentText(theme_combo.currentText())
            self.language_combo.setCurrentText(language_combo.currentText())
            self.retry_spin.setValue(retry_spin.value())
            self.timeout_spin.setValue(timeout_spin.value())
            self._apply_theme(self.theme_combo.currentText())
            self.change_language(self.language_combo.currentText())
            self.refresh_local_states()
            self._persist()
            dialog.accept()

        choose_btn.clicked.connect(choose_folder_for_dialog)
        open_btn.clicked.connect(open_folder_for_dialog)
        save_btn.clicked.connect(save_dialog_settings)

        layout.addWidget(QLabel(self.t("type")), 0, 0)
        layout.addWidget(repo_type_combo, 0, 1, 1, 3)
        layout.addWidget(QLabel(self.t("revision")), 1, 0)
        layout.addWidget(revision_input, 1, 1, 1, 3)
        layout.addWidget(use_token_check, 2, 1, 1, 3)
        layout.addWidget(QLabel(self.t("folder")), 3, 0)
        layout.addWidget(folder_input, 3, 1)
        layout.addWidget(choose_btn, 3, 2)
        layout.addWidget(open_btn, 3, 3)
        layout.addWidget(QLabel(self.t("theme")), 4, 0)
        layout.addWidget(theme_combo, 4, 1, 1, 3)
        layout.addWidget(QLabel(self.t("language")), 5, 0)
        layout.addWidget(language_combo, 5, 1, 1, 3)
        layout.addWidget(QLabel(self.t("retries")), 6, 0)
        layout.addWidget(retry_spin, 6, 1, 1, 3)
        layout.addWidget(QLabel(self.t("timeout")), 7, 0)
        layout.addWidget(timeout_spin, 7, 1, 1, 3)
        layout.addWidget(save_btn, 8, 3)

        dialog.setLayout(layout)
        dialog.exec()

    def change_language(self, language: str):
        self.language = language if language in TRANSLATIONS else "EN"
        self.apply_language()
        self._persist()
        self._refresh_jobs_table()

    def apply_language(self):
        self.setWindowTitle(APP_NAME)
        self.repository_group.setTitle(self.t("repository_group"))
        self.search_label.setText(self.t("search"))
        self.search_btn.setText(self.t("search"))
        self.settings_btn.setText(self.t("settings"))
        self.type_label.setText(self.t("type"))
        self.revision_label.setText(self.t("revision"))
        self.folder_label.setText(self.t("folder"))
        self.theme_label.setText(self.t("theme"))
        self.language_label.setText(self.t("language"))
        self.retries_label.setText(self.t("retries"))
        self.timeout_label.setText(self.t("timeout"))
        self.choose_folder_btn.setText(self.t("choose"))
        self.open_folder_btn.setText(self.t("open"))
        self.use_token_check.setText(self.t("use_hf_token"))
        if not self.current_repo_id:
            self.repo_label.setText(self.t("no_repository"))
        self.filter_label.setText(self.t("filter"))
        self.select_all_btn.setText(self.t("select_all"))
        self.clear_selection_btn.setText(self.t("clear_selection"))
        self.refresh_local_btn.setText(self.t("refresh_local"))
        self.download_selected_btn.setText(self.t("download_checked"))
        self.download_visible_btn.setText(self.t("download_visible"))
        self.downloads_history_label.setText(self.t("downloads_history"))
        self.job_status_filter_label.setText(self.t("status_filter"))
        self._refresh_job_status_filter_labels()
        if self.status_label.text() in {"Ready", "Готово"}:
            self.status_label.setText(self.t("ready"))
        self.files_table.setHorizontalHeaderLabels(
            ["", self.t("files_file"), self.t("files_size"), self.t("files_local")]
        )
        self.jobs_table.setHorizontalHeaderLabels(
            [
                "",
                self.t("jobs_repository"),
                self.t("jobs_status"),
                self.t("jobs_progress"),
                self.t("jobs_current_file"),
                self.t("jobs_size"),
                self.t("jobs_speed"),
                self.t("jobs_folder"),
                self.t("jobs_actions"),
            ]
        )

    def _refresh_job_status_filter_labels(self):
        if not hasattr(self, "job_status_filter_combo"):
            return
        current = self.job_status_filter_combo.currentText()
        options = [
            ("all", self.t("status_all")),
            ("downloading", self.t("status_downloading")),
            ("downloaded", self.t("status_downloaded")),
            ("error", self.t("status_error")),
            ("verified", self.t("status_verified")),
        ]
        self.job_status_filter_combo.blockSignals(True)
        self.job_status_filter_combo.clear()
        for value, label in options:
            self.job_status_filter_combo.addItem(label, value)
        index = self.job_status_filter_combo.findData(current)
        if index < 0:
            index = self.job_status_filter_combo.findData(self.settings.get("job_status_filter", "all"))
        self.job_status_filter_combo.setCurrentIndex(max(index, 0))
        self.job_status_filter_combo.blockSignals(False)

    def _normalize_jobs_on_start(self):
        changed = False
        for job in self.jobs:
            if job.get("status") in {STATUS_DOWNLOADING, STATUS_QUEUED}:
                job["status"] = STATUS_INTERRUPTED
                changed = True
        if changed:
            self._persist()

    def _persist(self):
        self.settings["jobs"] = self.jobs[:300]
        self.settings["expanded_jobs"] = list(self.expanded_jobs)
        if hasattr(self, "query_input"):
            self.settings["last_query"] = self.query_input.text().strip()
        if hasattr(self, "repo_type_combo"):
            self.settings["repo_type"] = self.repo_type_combo.currentText()
        if hasattr(self, "folder_input"):
            self.settings["download_dir"] = self.folder_input.text().strip()
        if hasattr(self, "revision_input"):
            self.settings["revision"] = self.revision_input.text().strip() or "main"
        if hasattr(self, "use_token_check"):
            self.settings["use_token"] = self.use_token_check.isChecked()
        if hasattr(self, "theme_combo"):
            self.settings["theme"] = self.theme_combo.currentText()
        if hasattr(self, "language_combo"):
            self.settings["language"] = self.language_combo.currentText()
        if hasattr(self, "file_filter_combo"):
            self.settings["file_filter"] = self.file_filter_combo.currentText()
        if hasattr(self, "job_status_filter_combo"):
            self.settings["job_status_filter"] = self.current_job_status_filter()
        self.settings["jobs_sort_column"] = self.jobs_sort_column
        self.settings["jobs_sort_ascending"] = self.jobs_sort_ascending
        if hasattr(self, "retry_spin"):
            self.settings["retries"] = self.retry_spin.value()
        if hasattr(self, "timeout_spin"):
            self.settings["timeout"] = self.timeout_spin.value()
        if hasattr(self, "main_splitter"):
            self.settings["splitter_sizes"] = self.main_splitter.sizes()
        if hasattr(self, "jobs_table"):
            self.settings["jobs_column_widths"] = [
                self.jobs_table.columnWidth(column)
                for column in range(self.jobs_table.columnCount())
            ]
        if self.isVisible():
            self.settings["window_geometry"] = bytes(self.saveGeometry()).hex()
        save_settings(self.settings)

    def _restore_window_state(self):
        geometry = self.settings.get("window_geometry")
        if geometry:
            try:
                self.restoreGeometry(QByteArray.fromHex(geometry.encode("ascii")))
            except Exception:
                pass

    def _apply_theme(self, theme: str):
        self.settings["theme"] = theme
        app = QApplication.instance()
        if theme == "Dark":
            app.setStyleSheet(
                """
                QWidget { background: #202124; color: #f1f3f4; }
                QLineEdit, QComboBox, QSpinBox, QTableWidget {
                    background: #2b2c30; color: #f1f3f4; border: 1px solid #5f6368;
                }
                QHeaderView::section { background: #303134; color: #f1f3f4; padding: 5px; }
                QPushButton { background: #3c4043; color: #f1f3f4; border: 1px solid #5f6368; padding: 6px 10px; }
                QPushButton:disabled { color: #8a8d91; }
                QGroupBox { border: 1px solid #5f6368; margin-top: 8px; padding-top: 8px; }
                QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
                """
            )
        elif theme == "Light":
            app.setStyleSheet("")
            app.setPalette(QApplication.style().standardPalette())
        else:
            app.setStyleSheet("")
            app.setPalette(QApplication.style().standardPalette())
        self._persist()

    def show_error(self, message: str):
        QMessageBox.critical(self, APP_NAME, message)

    def closeEvent(self, event):
        self._persist()
        super().closeEvent(event)

    def save_download_folder(self):
        folder = self.folder_input.text().strip()
        if not folder:
            return
        self.folder_input.setText(str(Path(folder).expanduser()))
        self._persist()

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            self.t("choose_download_folder"),
            self.folder_input.text(),
        )
        if folder:
            self.folder_input.setText(folder)
            self.refresh_local_states()
            self.save_download_folder()

    def open_current_download_folder(self):
        self.save_download_folder()
        self.open_folder(Path(self.folder_input.text()))

    def open_folder(self, folder: Path):
        try:
            folder = folder.expanduser()
            folder.mkdir(parents=True, exist_ok=True)
            os.startfile(str(folder))
        except OSError as exc:
            self.show_error(str(exc))

    def token_arg(self):
        return True if self.use_token_check.isChecked() else None

    def repo_type(self) -> str:
        return self.repo_type_combo.currentText()

    def revision(self) -> str:
        return self.revision_input.text().strip() or "main"

    def base_dir_for_repo(self, repo_id: str) -> Path:
        self.save_download_folder()
        return Path(self.folder_input.text()).expanduser() / repo_id

    def search_repo(self):
        query = self.query_input.text().strip()
        if not query:
            self.show_error(self.t("enter_repo"))
            return
        self._persist()
        self.repo_combo.clear()
        self.repo_combo.setVisible(False)
        self.files_table.setRowCount(0)
        self.current_files = []
        self.checked_file_names.clear()
        self.selection_initialized = False
        self.current_repo_id = ""
        self.download_selected_btn.setEnabled(False)
        self.download_visible_btn.setEnabled(False)
        self.signals.status.emit("Searching...")

        if "/" in query:
            self.load_repo(query)
            return

        selected_repo_type = self.repo_type()

        def work():
            try:
                if selected_repo_type == "dataset":
                    results = [item.id for item in self.api.list_datasets(search=query, limit=50)]
                elif selected_repo_type == "space":
                    results = [item.id for item in self.api.list_spaces(search=query, limit=50)]
                else:
                    results = [item.modelId for item in self.api.list_models(search=query, limit=50)]
                if not results:
                    raise RuntimeError("No repositories found.")
                self.signals.search_results.emit(results)
            except Exception as exc:
                self.signals.error.emit(str(exc))
                self.signals.status.emit("Search failed")

        threading.Thread(target=work, daemon=True).start()

    def _show_search_results(self, repo_ids: List[str]):
        self.repo_combo.addItem("Select repository...")
        self.repo_combo.addItems(repo_ids)
        self.repo_combo.setCurrentIndex(0)
        self.repo_combo.setVisible(True)
        self.signals.status.emit(f"Found {len(repo_ids)} repositories")

    def _repo_combo_changed(self, index: int):
        if index <= 0:
            return
        self.load_repo(self.repo_combo.currentText())

    def load_repo(self, repo_id: str):
        self.current_repo_id = repo_id
        self.repo_label.setText(f"Repository: {repo_id}")
        self.signals.status.emit("Loading repository files...")
        selected_repo_type = self.repo_type()
        selected_revision = self.revision()
        selected_token = self.token_arg()

        def work():
            try:
                files: List[RepoFile] = []
                for entry in self.api.list_repo_tree(
                    repo_id=repo_id,
                    repo_type=repo_type_arg(selected_repo_type),
                    revision=selected_revision,
                    recursive=True,
                    expand=True,
                    token=selected_token,
                ):
                    if not hasattr(entry, "size"):
                        continue
                    files.append(RepoFile(name=entry.path, size=entry.size or 0))
                files.sort(key=lambda item: item.name.lower())
                self.signals.files_ready.emit([asdict(item) for item in files])
            except Exception as exc:
                self.signals.error.emit(str(exc))
                self.signals.status.emit("Failed to load repository")

        threading.Thread(target=work, daemon=True).start()

    def _set_repo_files(self, files: List[Dict]):
        self.current_files = [RepoFile(**item) for item in files]
        self.checked_file_names.clear()
        self.selection_initialized = False
        self.refresh_local_states()
        self.download_selected_btn.setEnabled(True)
        self.download_visible_btn.setEnabled(True)
        total = sum(item.size for item in self.current_files)
        self.signals.status.emit(f"{len(self.current_files)} files, {human_size(total)}")

    def refresh_local_states(self):
        if not self.current_repo_id:
            return
        self._remember_checked_files()
        base_dir = self.base_dir_for_repo(self.current_repo_id)
        for item in self.current_files:
            state, size = local_state(base_dir / item.name, item.size)
            item.local_state = state
            item.local_size = size
        if not self.selection_initialized:
            self.checked_file_names = {
                item.name for item in self.current_files if item.local_state != "Ready"
            }
            self.selection_initialized = True
        self._refresh_files_table(preserve_current=False)

    def _filter_accepts(self, item: RepoFile) -> bool:
        pattern = FILTERS.get(self.file_filter_combo.currentText())
        if pattern is None:
            return True
        name = item.name.lower()
        return any(name.endswith(suffix.lower()) for suffix in pattern)

    def visible_files(self) -> List[RepoFile]:
        return [item for item in self.current_files if self._filter_accepts(item)]

    def _remember_checked_files(self):
        if not self.selection_initialized:
            return
        for row in range(self.files_table.rowCount()):
            check_item = self.files_table.item(row, 0)
            name_item = self.files_table.item(row, 1)
            if not check_item or not name_item:
                continue
            if check_item.checkState() == Qt.Checked:
                self.checked_file_names.add(name_item.text())
            else:
                self.checked_file_names.discard(name_item.text())

    def select_all_visible_files(self):
        for item in self.visible_files():
            self.checked_file_names.add(item.name)
        self.selection_initialized = True
        self._refresh_files_table(preserve_current=False)

    def clear_visible_selection(self):
        for item in self.visible_files():
            self.checked_file_names.discard(item.name)
        self.selection_initialized = True
        self._refresh_files_table(preserve_current=False)

    def _refresh_files_table(self, preserve_current: bool = True):
        if preserve_current:
            self._remember_checked_files()
        visible = self.visible_files()
        self.files_table.setRowCount(0)
        for row, item in enumerate(visible):
            self.files_table.insertRow(row)
            check_item = QTableWidgetItem()
            check_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            check_item.setCheckState(Qt.Checked if item.name in self.checked_file_names else Qt.Unchecked)
            self.files_table.setItem(row, 0, check_item)
            self.files_table.setItem(row, 1, self.table_item(item.name))
            size_item = QTableWidgetItem(human_size(item.size))
            size_item.setTextAlignment(Qt.AlignCenter)
            size_item.setToolTip(size_item.text())
            self.files_table.setItem(row, 2, size_item)
            local_text = item.local_state
            if item.local_state in {"Partial", "Mismatch", "Present"}:
                local_text += f" ({human_size(item.local_size)})"
            local_item = QTableWidgetItem(local_text)
            local_item.setTextAlignment(Qt.AlignCenter)
            local_item.setToolTip(local_text)
            self.files_table.setItem(row, 3, local_item)
        self.files_table.resizeColumnToContents(0)
        self.files_table.resizeColumnToContents(2)
        self.files_table.resizeColumnToContents(3)

    def table_item(self, text: str, tooltip: Optional[str] = None) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setToolTip(tooltip if tooltip is not None else text)
        return item

    def checked_files(self) -> List[RepoFile]:
        self._remember_checked_files()
        by_name = {item.name: item for item in self.visible_files()}
        selected: List[RepoFile] = []
        for row in range(self.files_table.rowCount()):
            check_item = self.files_table.item(row, 0)
            name_item = self.files_table.item(row, 1)
            if check_item and name_item and check_item.checkState() == Qt.Checked:
                file = by_name.get(name_item.text())
                if file:
                    selected.append(file)
        return selected

    def download_checked_files(self):
        files = self.checked_files()
        if not files:
            self.show_error(self.t("check_file"))
            return
        self.enqueue_job(files, "files")

    def download_visible_files(self):
        files = self.visible_files()
        if not files:
            self.show_error(self.t("no_visible_files"))
            return
        self.enqueue_job(files, "repo" if len(files) == len(self.current_files) else "files")

    def _matching_repo_job(self, repo_id: str, repo_type: str, revision: str, base_dir: Path) -> Optional[Dict]:
        base_dir_text = str(base_dir)
        for job in self.jobs:
            if (
                job.get("repo_id") == repo_id
                and job.get("repo_type", "model") == repo_type
                and job.get("revision", "main") == revision
                and job.get("base_dir", "") == base_dir_text
            ):
                return job
        return None

    def _merge_files_into_job(self, job: Dict, files: List[RepoFile], scope: str) -> bool:
        existing_names = {item.get("name", "") for item in job.get("files", [])}
        progress_names = {item.get("name", "") for item in job.get("file_progress", [])}
        added = False
        for file in files:
            if file.name in existing_names:
                continue
            job.setdefault("files", []).append(asdict(file))
            if file.name not in progress_names:
                job.setdefault("file_progress", []).append(
                    {
                        "name": file.name,
                        "status": "Waiting",
                        "progress": 0,
                        "bytes_done": 0,
                        "bytes_total": file.size,
                        "speed": 0,
                    }
                )
            existing_names.add(file.name)
            added = True

        if not added:
            return False

        job["scope"] = "repo" if scope == "repo" or job.get("scope") == "repo" else "files"
        job["use_token"] = self.use_token_check.isChecked()
        job["retries"] = self.retry_spin.value()
        job["timeout"] = self.timeout_spin.value()
        job["error"] = ""
        job["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if self.active_job_id != job.get("id"):
            job["status"] = STATUS_QUEUED
            job["current_file"] = ""
            job["speed"] = 0
        self._recalculate_job_totals(job)
        return True

    def _recalculate_job_totals(self, job: Dict):
        files = {item.get("name", ""): int(item.get("size") or 0) for item in job.get("files", [])}
        total = sum(files.values())
        done = 0
        for state in self._job_file_progress(job):
            name = state.get("name", "")
            expected = files.get(name, int(state.get("bytes_total") or 0))
            bytes_done = int(state.get("bytes_done") or 0)
            if state.get("status") in {"Ready", STATUS_VERIFIED}:
                done += expected or bytes_done
            else:
                done += min(bytes_done, expected) if expected else bytes_done
        job["bytes_total"] = total
        job["bytes_done"] = min(done, total) if total else done
        job["progress"] = int(job["bytes_done"] * 100 / total) if total else 0

    def enqueue_job(self, files: List[RepoFile], scope: str):
        if not self.current_repo_id:
            self.show_error(self.t("load_repo_first"))
            return
        base_dir = self.base_dir_for_repo(self.current_repo_id)
        existing_job = self._matching_repo_job(
            self.current_repo_id,
            self.repo_type(),
            self.revision(),
            base_dir,
        )
        if existing_job:
            if self._merge_files_into_job(existing_job, files, scope):
                self.expanded_jobs.add(existing_job["id"])
                self.jobs = [existing_job] + [
                    job for job in self.jobs if job.get("id") != existing_job.get("id")
                ]
                self._persist()
                self._refresh_jobs_table()
                self._start_next_queued_job()
            return
        job = {
            "id": str(uuid.uuid4()),
            "repo_id": self.current_repo_id,
            "repo_type": self.repo_type(),
            "revision": self.revision(),
            "use_token": self.use_token_check.isChecked(),
            "retries": self.retry_spin.value(),
            "timeout": self.timeout_spin.value(),
            "scope": scope,
            "files": [asdict(item) for item in files],
            "file_progress": [
                {
                    "name": item.name,
                    "status": "Waiting",
                    "progress": 0,
                    "bytes_done": 0,
                    "bytes_total": item.size,
                    "speed": 0,
                }
                for item in files
            ],
            "base_dir": str(base_dir),
            "status": STATUS_QUEUED,
            "progress": 0,
            "bytes_done": 0,
            "bytes_total": sum(item.size for item in files),
            "speed": 0,
            "current_file": "",
            "error": "",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.jobs.insert(0, job)
        self._persist()
        self._refresh_jobs_table()
        self._start_next_queued_job()

    def _find_job(self, job_id: str) -> Optional[Dict]:
        return next((job for job in self.jobs if job.get("id") == job_id), None)

    def _update_job(self, job_id: str, patch: Dict):
        job = self._find_job(job_id)
        if not job:
            return
        job.update(patch)
        job["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        progress = int(job.get("progress") or 0)
        should_save = (
            patch.get("status") is not None
            or patch.get("error") is not None
            or self._last_saved_progress.get(job_id) != progress
        )
        if should_save:
            self._last_saved_progress[job_id] = progress
            self._persist()
        self._refresh_jobs_table()

    def _refresh_jobs_table(self):
        self.jobs_table.setRowCount(0)
        row = 0
        for job in self.visible_jobs():
            self.jobs_table.insertRow(row)
            is_expanded = job.get("id") in self.expanded_jobs
            expand_label = "-" if is_expanded else "+"
            values = [
                expand_label if (job.get("file_progress") or job.get("files")) else "",
                job.get("repo_id", ""),
                job.get("status", ""),
                "",
                job.get("current_file", ""),
                f"{human_size(int(job.get('bytes_done') or 0))} / {human_size(int(job.get('bytes_total') or 0))}",
                human_speed(float(job.get("speed") or 0)),
                job.get("base_dir", ""),
            ]
            for col, value in enumerate(values):
                item = self.table_item(value)
                item.setData(Qt.UserRole, job.get("id", ""))
                if col in {0, 2, 5}:
                    item.setTextAlignment(Qt.AlignCenter)
                if col == 0:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                    item.setToolTip(expand_label)
                if col == 1:
                    item.setForeground(QBrush(Qt.blue))
                    font = item.font()
                    font.setUnderline(True)
                    item.setFont(font)
                    item.setToolTip(f"{value}\n{self.t('open_repo_tooltip')}")
                self.jobs_table.setItem(row, col, item)

            progress = QProgressBar()
            progress.setRange(0, 100)
            progress.setValue(int(job.get("progress") or 0))
            progress.setFormat("%p%")
            progress.setToolTip(f"{int(job.get('progress') or 0)}%")
            progress.setStyleSheet(self.progress_style(job.get("status", ""), int(job.get("progress") or 0)))
            self.jobs_table.setCellWidget(row, 3, progress)

            actions = QWidget()
            actions.setMinimumWidth(JOBS_COLUMN_MIN_WIDTHS[8])
            actions_layout = QHBoxLayout()
            actions_layout.setContentsMargins(4, 0, 4, 0)
            actions_layout.setSpacing(4)
            resume_btn = QPushButton(self.t("resume"))
            cancel_btn = QPushButton(self.t("cancel"))
            open_btn = QPushButton(self.t("open"))
            remove_btn = QPushButton(self.t("remove"))
            for button in (resume_btn, cancel_btn, open_btn, remove_btn):
                button.setMinimumWidth(88)
                button.setToolTip(button.text())
            active = self.active_job_id == job.get("id")
            resume_btn.setEnabled(not active and job.get("status") != STATUS_VERIFIED)
            cancel_btn.setEnabled(active)
            remove_btn.setEnabled(not active)

            resume_btn.clicked.connect(lambda checked=False, jid=job["id"]: self.resume_job(jid))
            cancel_btn.clicked.connect(lambda checked=False, jid=job["id"]: self.cancel_job(jid))
            open_btn.clicked.connect(lambda checked=False, folder=job.get("base_dir", ""): self.open_folder(Path(folder)))
            remove_btn.clicked.connect(lambda checked=False, jid=job["id"]: self.remove_job(jid))

            actions_layout.addWidget(resume_btn)
            actions_layout.addWidget(cancel_btn)
            actions_layout.addWidget(open_btn)
            actions_layout.addWidget(remove_btn)
            actions.setLayout(actions_layout)
            self.jobs_table.setCellWidget(row, 8, actions)
            row += 1

            if is_expanded:
                for file_state in self._job_file_progress(job):
                    self._insert_file_progress_row(row, file_state)
                    row += 1

        for column in range(self.jobs_table.columnCount()):
            self.jobs_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.Interactive)

    def visible_jobs(self) -> List[Dict]:
        jobs = [job for job in self.jobs if self.job_matches_status_filter(job)]
        if self.jobs_sort_column:
            jobs.sort(
                key=lambda job: (self.job_sort_key(job), job.get("updated_at", "")),
                reverse=not self.jobs_sort_ascending,
            )
        return jobs

    def job_matches_status_filter(self, job: Dict) -> bool:
        selected = self.current_job_status_filter()
        status = job.get("status", "")
        progress = int(job.get("progress") or 0)
        if selected == "all":
            return True
        if selected == "downloading":
            return status in {STATUS_DOWNLOADING, STATUS_QUEUED}
        if selected == "downloaded":
            return progress >= 100 or status == STATUS_VERIFIED
        if selected == "error":
            return status == STATUS_FAILED
        if selected == "verified":
            return status == STATUS_VERIFIED
        return True

    def status_sort_key(self, job: Dict) -> int:
        order = {
            STATUS_FAILED: 0,
            STATUS_CANCELED: 1,
            STATUS_INTERRUPTED: 2,
            STATUS_QUEUED: 3,
            STATUS_DOWNLOADING: 4,
            STATUS_VERIFIED: 5,
        }
        return order.get(job.get("status", ""), 99)

    def job_sort_key(self, job: Dict):
        if self.jobs_sort_column == "status":
            return self.status_sort_key(job)
        if self.jobs_sort_column == "progress":
            return int(job.get("progress") or 0)
        if self.jobs_sort_column == "size":
            return int(job.get("bytes_total") or 0)
        return job.get("updated_at", "")

    def progress_style(self, status: str, progress: int) -> str:
        if status == STATUS_FAILED:
            color = "#d93025"
        elif progress >= 100 or status == STATUS_VERIFIED:
            color = "#188038"
        elif status in {STATUS_CANCELED, STATUS_INTERRUPTED}:
            color = "#f9ab00"
        else:
            color = "#1a73e8"
        return (
            "QProgressBar { text-align: center; } "
            f"QProgressBar::chunk {{ background-color: {color}; }}"
        )

    def _jobs_table_clicked(self, row: int, column: int):
        item = self.jobs_table.item(row, column)
        if not item:
            return
        job_id = item.data(Qt.UserRole)
        if not job_id:
            return
        job = self._find_job(job_id)
        if not job:
            return
        if column == 0:
            if job.get("file_progress") or job.get("files"):
                self.toggle_job_expanded(job_id)
            return
        if column != 1:
            return
        QDesktopServices.openUrl(
            QUrl(make_repo_url(job.get("repo_id", ""), job.get("repo_type", "model")))
        )

    def _job_file_progress(self, job: Dict) -> List[Dict]:
        progress = job.get("file_progress")
        if progress:
            return progress
        return [
            {
                "name": item.get("name", ""),
                "status": "Waiting",
                "progress": 0,
                "bytes_done": 0,
                "bytes_total": item.get("size", 0),
            }
            for item in job.get("files", [])
        ]

    def _insert_file_progress_row(self, row: int, file_state: Dict):
        self.jobs_table.insertRow(row)
        file_name = file_state.get("name", "")
        expand_item = self.table_item("")
        expand_item.setData(Qt.UserRole, "")
        self.jobs_table.setItem(row, 0, expand_item)
        name_item = self.table_item(f"  - {file_name}", file_name)
        name_item.setData(Qt.UserRole, "")
        self.jobs_table.setItem(row, 1, name_item)
        status_item = self.table_item(file_state.get("status", ""))
        status_item.setTextAlignment(Qt.AlignCenter)
        self.jobs_table.setItem(row, 2, status_item)

        progress = QProgressBar()
        progress.setRange(0, 100)
        file_progress = int(file_state.get("progress") or 0)
        progress.setValue(file_progress)
        progress.setFormat("%p%")
        progress.setToolTip(f"{file_progress}%")
        progress.setStyleSheet(self.progress_style(file_state.get("status", ""), file_progress))
        self.jobs_table.setCellWidget(row, 3, progress)

        self.jobs_table.setItem(row, 4, self.table_item(file_name))
        size_text = (
            f"{human_size(int(file_state.get('bytes_done') or 0))} / "
            f"{human_size(int(file_state.get('bytes_total') or 0))}"
        )
        size_item = self.table_item(size_text)
        size_item.setTextAlignment(Qt.AlignCenter)
        self.jobs_table.setItem(row, 5, size_item)
        speed_item = self.table_item(human_speed(float(file_state.get("speed") or 0)))
        speed_item.setTextAlignment(Qt.AlignCenter)
        self.jobs_table.setItem(row, 6, speed_item)
        self.jobs_table.setItem(row, 7, self.table_item(""))
        self.jobs_table.setItem(row, 8, self.table_item(""))

    def toggle_job_expanded(self, job_id: str):
        if job_id in self.expanded_jobs:
            self.expanded_jobs.remove(job_id)
        else:
            self.expanded_jobs.add(job_id)
        self._persist()
        self._refresh_jobs_table()

    def resume_job(self, job_id: str):
        job = self._find_job(job_id)
        if not job:
            return
        job["status"] = STATUS_QUEUED
        job["error"] = ""
        self._persist()
        self._refresh_jobs_table()
        self._start_next_queued_job()

    def cancel_job(self, job_id: str):
        if self.active_job_id == job_id:
            self.cancel_event.set()

    def remove_job(self, job_id: str):
        self.jobs = [job for job in self.jobs if job.get("id") != job_id]
        self._persist()
        self._refresh_jobs_table()

    def _start_next_queued_job(self):
        if self.active_job_id:
            return
        job = next((item for item in reversed(self.jobs) if item.get("status") == STATUS_QUEUED), None)
        if not job:
            return
        self._start_job(job)

    def _start_job(self, job: Dict):
        job_id = job["id"]
        self.active_job_id = job_id
        self.cancel_event = threading.Event()
        self.signals.job_update.emit(job_id, {"status": STATUS_DOWNLOADING, "progress": 0, "error": ""})

        def work():
            try:
                self._download_job(job_id)
            except Exception as exc:
                self.signals.job_update.emit(
                    job_id,
                    {"status": STATUS_FAILED, "error": str(exc), "speed": 0},
                )
            finally:
                self.signals.job_finished.emit(job_id)

        threading.Thread(target=work, daemon=True).start()

    def _download_job(self, job_id: str):
        job = self._find_job(job_id)
        if not job:
            return

        headers = auth_headers(bool(job.get("use_token", False)))
        base_dir = Path(job["base_dir"])
        files = [RepoFile(**item) for item in job.get("files", [])]
        total_bytes = sum(item.size for item in files)
        completed_before = 0
        retries = int(job.get("retries", 2))
        timeout = int(job.get("timeout", 30))
        speed_state = {"last_time": time.time(), "last_done": None}

        for index, item in enumerate(files, start=1):
            path = base_dir / item.name
            state, size = local_state(path, item.size)
            if state == "Ready":
                completed_before += item.size
                self.signals.job_update.emit(
                    job_id,
                    {
                        "file_progress": self._patched_file_progress(
                            job,
                            item.name,
                            {
                                "status": "Ready",
                                "progress": 100,
                                "bytes_done": item.size,
                                "bytes_total": item.size,
                                "speed": 0,
                            },
                        )
                    },
                )
                self._emit_job_progress(job_id, completed_before, total_bytes, 0)
                continue

            self.signals.job_update.emit(
                job_id,
                {
                    "status": STATUS_DOWNLOADING,
                    "current_file": f"{item.name} ({index}/{len(files)})",
                    "file_progress": self._patched_file_progress(
                        job,
                        item.name,
                        {
                            "status": "Downloading",
                            "progress": 0,
                            "bytes_done": 0,
                            "bytes_total": item.size,
                            "speed": 0,
                        },
                    ),
                },
            )

            url = make_file_url(
                job["repo_id"],
                item.name,
                job.get("repo_type", "model"),
                job.get("revision", "main"),
            )

            last_error = None
            for attempt in range(retries + 1):
                try:
                    def on_file_progress(done: int, total: int):
                        effective_total = item.size or total
                        now = time.time()
                        total_done = completed_before + min(done, effective_total)
                        elapsed = max(now - speed_state["last_time"], 0.001)
                        if speed_state["last_done"] is None:
                            speed = 0
                        else:
                            speed = max(total_done - speed_state["last_done"], 0) / elapsed
                        speed_state["last_time"] = now
                        speed_state["last_done"] = total_done
                        file_pct = int(done * 100 / effective_total) if effective_total else 0
                        self.signals.job_update.emit(
                            job_id,
                            {
                                "file_progress": self._patched_file_progress(
                                    job,
                                    item.name,
                                    {
                                        "status": "Downloading",
                                        "progress": max(0, min(100, file_pct)),
                                        "bytes_done": min(done, effective_total),
                                        "bytes_total": effective_total,
                                        "speed": speed,
                                    },
                                )
                            },
                        )
                        self._emit_job_progress(
                            job_id,
                            total_done,
                            total_bytes,
                            speed,
                        )

                    stream_file(
                        url=url,
                        dest_path=path,
                        expected_size=item.size,
                        headers=headers,
                        timeout=timeout,
                        cancel_event=self.cancel_event,
                        progress_callback=on_file_progress,
                    )
                    last_error = None
                    break
                except DownloadCanceled:
                    self.signals.job_update.emit(
                        job_id,
                            {
                                "status": STATUS_CANCELED,
                                "speed": 0,
                                "file_progress": self._patched_file_progress(
                                    job,
                                    item.name,
                                    {"status": STATUS_CANCELED, "speed": 0},
                                ),
                            },
                    )
                    return
                except Exception as exc:
                    last_error = exc
                    if attempt < retries:
                        time.sleep(1 + attempt)

            if last_error:
                self.signals.job_update.emit(
                    job_id,
                    {
                        "status": STATUS_FAILED,
                        "error": str(last_error),
                        "speed": 0,
                        "file_progress": self._patched_file_progress(
                            job,
                            item.name,
                            {"status": STATUS_FAILED, "speed": 0},
                        ),
                    },
                )
                return

            completed_before += item.size or path.stat().st_size
            self.signals.job_update.emit(
                job_id,
                {
                    "file_progress": self._patched_file_progress(
                        job,
                        item.name,
                        {
                            "status": STATUS_VERIFIED,
                            "progress": 100,
                            "bytes_done": item.size or path.stat().st_size,
                            "bytes_total": item.size or path.stat().st_size,
                            "speed": 0,
                        },
                    )
                },
            )
            self._emit_job_progress(job_id, completed_before, total_bytes, 0)

        latest_job = self._find_job(job_id)
        latest_files = latest_job.get("files", []) if latest_job else []
        if len(latest_files) > len(files):
            if latest_job:
                self._recalculate_job_totals(latest_job)
            self.signals.job_update.emit(
                job_id,
                {
                    "status": STATUS_QUEUED,
                    "speed": 0,
                    "current_file": "",
                },
            )
            return

        self.signals.job_update.emit(
            job_id,
            {
                "status": STATUS_VERIFIED,
                "progress": 100,
                "bytes_done": total_bytes,
                "bytes_total": total_bytes,
                "speed": 0,
                "current_file": "",
            },
        )

    def _patched_file_progress(self, job: Dict, file_name: str, patch: Dict) -> List[Dict]:
        progress = [dict(item) for item in self._job_file_progress(job)]
        target = next((item for item in progress if item.get("name") == file_name), None)
        if target is None:
            target = {"name": file_name, "status": "Waiting", "progress": 0, "bytes_done": 0, "bytes_total": 0}
            progress.append(target)
        target.update(patch)
        job["file_progress"] = progress
        return progress

    def _emit_job_progress(self, job_id: str, done: int, total: int, speed: float = 0):
        progress = int(done * 100 / total) if total else 100
        self.signals.job_update.emit(
            job_id,
            {
                "progress": max(0, min(100, progress)),
                "bytes_done": min(done, total) if total else done,
                "bytes_total": total,
                "speed": speed,
            },
        )

    def _job_finished(self, job_id: str):
        if self.active_job_id == job_id:
            self.active_job_id = None
        self.refresh_local_states()
        self._refresh_jobs_table()
        self._start_next_queued_job()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

