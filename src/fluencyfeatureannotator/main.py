import os
from pathlib import Path
from traceback import format_exc
from typing import Callable, List

import flet as ft
import pandas as pd
from fluency_feature_annotator import FluencyFeatureAnnotator, TextGrid, Turn, save_grid, save_turn

NO_FILE_SELECTED_TEXT = ft.Text("・ No files are selected.", theme_style=ft.TextThemeStyle.LABEL_LARGE)
ERROR_SELECTED_TEXT = ft.Text(
    "While processing files, unexpected error(s) occurred.\nPlease try it again or share the error message.",
    theme_style=ft.TextThemeStyle.LABEL_LARGE,
    color=ft.colors.ERROR
)
UPLOAD_DIR = Path("uploads")
RESULT_CSV_PATH = Path("results/result.csv")

os.environ["FLET_SECRET_KEY"] = os.urandom(12).hex()

class SelectedFileContainer(ft.Container):
    def __init__(self):
        super().__init__()

        self.width = 500
        self.height = 300
        self.margin = 10
        self.padding = 10
        self.border = ft.border.all(5, color=ft.colors.PRIMARY)
        self.border_radius = ft.border_radius.all(30)
        self.bgcolor = ft.colors.WHITE10

        self.selected_file_list = ft.ListView(
            expand=1,
            spacing=10,
            padding=20,
            auto_scroll=False,
            controls=[NO_FILE_SELECTED_TEXT]
        )

        self.content = self.selected_file_list

class FileSelectionWarningBanner(ft.Banner):
    def __init__(self, on_click: Callable):
        content = ft.Text(
            value="Select wav and txt file pairs!",
            color=ft.colors.BLACK,
        )
        actions = [
            ft.TextButton(text="Close", style=ft.ButtonStyle(color=ft.colors.PRIMARY), on_click=on_click)
        ]

        super().__init__(content=content, actions=actions)

        self.bgcolor = ft.colors.AMBER_100
        self.leading = ft.Icon(ft.icons.WARNING_AMBER_ROUNDED, color=ft.colors.AMBER, size=40)

class GeneralErrorBanner(ft.Banner):
    def __init__(self, on_click: Callable):
        content = ft.Text(
            value="Unexpected Error",
            color=ft.colors.BLACK,
        )
        actions = [
            ft.TextButton(text="Close", style=ft.ButtonStyle(color=ft.colors.PRIMARY), on_click=on_click)
        ]

        super().__init__(content=content, actions=actions)

        self.bgcolor = ft.colors.ERROR_CONTAINER
        self.leading = ft.Icon(ft.icons.WARNING, color=ft.colors.ERROR, size=40)

    def update_error_message(self, error_message: str) -> None:
        content = ft.Text(
            value=error_message,
            color=ft.colors.BLACK,
            overflow=ft.TextOverflow.VISIBLE
        )

        self.content = content

class UploadedFileProgressbar(ft.Column):
    def __init__(self):
        super().__init__()

        self.file_name = ft.Text("", theme_style=ft.TextThemeStyle.LABEL_LARGE)
        self.progress_bar = ft.ProgressBar(value=0, width=300)

        self.controls = [self.file_name, self.progress_bar]

    def update_bar(self, text: str, rate: float) -> None:
        self.file_name.value = text
        self.progress_bar.value = rate

        self.update()


class WavTxtFilePicker(ft.FilePicker):
    def __init__(
        self,
        selected_file_container: SelectedFileContainer,
        uploaded_file_progressbar: UploadedFileProgressbar
    ):
        super().__init__()

        self.on_result = self.pick_file_results
        self.on_upload = self.upload_file_progress
        self.picked_wav_path_list = []
        self.picked_txt_path_list = []

        self.selected_file_container = selected_file_container
        self.uploaded_file_progressbar = uploaded_file_progressbar
        self.file_selection_warning_banner = FileSelectionWarningBanner(on_click=self.close_warning_banner)
        self.general_error_banner = GeneralErrorBanner(on_click=self.close_error_banner)

    def close_warning_banner(self, e):
        self.page.close(self.file_selection_warning_banner)

    def close_error_banner(self, e):
        self.selected_file_container.selected_file_list.controls.append(
            NO_FILE_SELECTED_TEXT
        )
        self.page.update()
        self.page.close(self.general_error_banner)

    def is_wav_txt_path_pair_picked(
        self,
        picked_wav_path_list: List[Path],
        picked_txt_path_lsit: List[Path]
    ) -> bool:
        if len(picked_wav_path_list) != len(picked_txt_path_lsit):
            return False

        picked_wav_filename = []
        picked_txt_filename = []
        for picked_wav_path, picked_txt_path in zip(picked_wav_path_list, picked_txt_path_lsit):
            picked_wav_filename.append(picked_wav_path.stem)
            picked_txt_filename.append(picked_txt_path.stem)

        picked_wav_filename = sorted(picked_wav_filename)
        picked_txt_filename = sorted(picked_txt_filename)

        return picked_wav_filename == picked_txt_filename

    def is_file_selected(self, e: ft.FilePickerResultEvent) -> None:
        if e is None:
            # ファイルが選択されなかった場合
            self.selected_file_container.selected_file_list.controls = [
                NO_FILE_SELECTED_TEXT
            ]
            self.page.update()
            return False

        return True

    def update_picked_files(self, e: ft.FilePickerResultEvent) -> None:
        picked_wav_path_list = []
        picked_txt_path_list = []

        for file in e.files:
            file_name = file.name
            if file_name.endswith(".wav"):
                picked_wav_path_list.append(UPLOAD_DIR / file_name)
            elif file_name.endswith(".txt"):
                picked_txt_path_list.append(UPLOAD_DIR / file_name)

        if not self.is_wav_txt_path_pair_picked(picked_wav_path_list, picked_txt_path_list):
            self.page.open(self.file_selection_warning_banner)
            self.selected_file_container.selected_file_list.controls = [
                NO_FILE_SELECTED_TEXT
            ]
            self.page.update()
            return False

        self.picked_wav_path_list = picked_wav_path_list
        self.picked_txt_path_list = picked_txt_path_list

        showing_file_list = []
        for idx, wav_path in enumerate(picked_wav_path_list):
            showing_file_list.append(ft.Text(
                f"・ [{idx:03}] {wav_path.name} - {wav_path.stem}.txt",
                theme_style=ft.TextThemeStyle.LABEL_LARGE
            ))
        showing_file_list.append(self.uploaded_file_progressbar)
        self.selected_file_container.selected_file_list.controls = showing_file_list
        self.page.update()

        return True

    def handle_exception(self, error_message: str) -> None:
        self.general_error_banner.update_error_message(error_message)
        self.picked_wav_path_list = []
        self.picked_txt_path_list = []
        self.selected_file_container.selected_file_list.controls = [
            ERROR_SELECTED_TEXT
        ]
        self.page.open(self.general_error_banner)
        self.page.update()

    def pick_file_results(self, e: ft.FilePickerResultEvent) -> None:
        try:
            if not self.is_file_selected(e):
                return

            if not self.update_picked_files(e):
                return
        except Exception:
            error_message = format_exc()
            self.handle_exception(error_message)

    def upload_file_progress(self, e: ft.FilePickerUploadEvent) -> None:
        message = f"Uploading {e.file_name}... [{int(e.progress * 100)}%]"

        if e.progress == 1:
            message = f"Upload completed! [{int(e.progress * 100)}%]"

        print(message)

        self.uploaded_file_progressbar.update_bar(message, e.progress)

        self.update()


class WavTxtFileManager(ft.Column):
    def __init__(self, annotator: FluencyFeatureAnnotator):
        super().__init__()

        self.annotator = annotator

        self.selected_file_container = SelectedFileContainer()
        self.uploaded_progresbar = UploadedFileProgressbar()
        self.pick_file_dialog = WavTxtFilePicker(
            self.selected_file_container,
            self.uploaded_progresbar
        )

        self.file_selection_warning_banner = FileSelectionWarningBanner(on_click=self.close_warning_banner)
        self.general_error_banner = GeneralErrorBanner(on_click=self.close_error_banner)

        self.progress_ring = ft.Container(
            width = 500,
            height = 300,
            margin = 10,
            padding = 20,
            border = ft.border.all(5, color=ft.colors.PRIMARY),
            border_radius = ft.border_radius.all(30),
            bgcolor = ft.colors.with_opacity(opacity=0.8, color="#eeeeee"),
            content=ft.Row(
                controls=[
                    ft.ProgressRing(width=20, height=20, stroke_width=4),
                    ft.Text("Annotating...", size=20)
                ],
                alignment=ft.MainAxisAlignment.CENTER
            ),
            visible=False
        )

        self.select_button = ft.ElevatedButton(
            text="① Select wav & txt files",
            icon=ft.icons.FOLDER_OPEN,
            on_click=lambda _: self.pick_file_dialog.pick_files(
                allow_multiple=True,
                allowed_extensions=["wav", "txt"]
            ),
            width=300
        )

        self.upload_button = ft.ElevatedButton(
            text="② Upload wav & txt files",
            icon=ft.icons.UPLOAD_FILE,
            on_click=self.upload_files,
            width=300
        )

        self.annotate_button = ft.FilledButton(
            text="③ Annotate fluency features",
            icon=ft.icons.MULTITRACK_AUDIO_ROUNDED,
            on_click=lambda _: self.annotate(
                self.pick_file_dialog.picked_wav_path_list,
                self.pick_file_dialog.picked_txt_path_list
            ),
            width=300
        )

        self.controls = [
            ft.Stack(controls=[
                self.pick_file_dialog,
                self.select_button
            ]),
            self.upload_button,
            ft.Stack(controls=[
                self.selected_file_container,
                self.progress_ring
            ]),
            self.annotate_button
        ]

    def close_warning_banner(self, e):
        self.page.close(self.file_selection_warning_banner)

    def close_error_banner(self, e) -> None:
        self.selected_file_container.selected_file_list.controls.append(
            NO_FILE_SELECTED_TEXT
        )
        self.page.update()
        self.page.close(self.general_error_banner)
        self.enable_control()

    def upload_files(self, e) -> None:
        uf = []
        txt_path_list = self.pick_file_dialog.picked_txt_path_list
        wav_path_list = self.pick_file_dialog.picked_wav_path_list
        for txt_path, wav_path in zip(txt_path_list, wav_path_list):
            txt_filename = txt_path.name
            wav_filename = wav_path.name

            uf.append(
                ft.FilePickerUploadFile(
                    txt_filename,
                    upload_url=self.page.get_upload_url(txt_filename, 600)
                )
            )
            uf.append(
                ft.FilePickerUploadFile(
                    wav_filename,
                    upload_url=self.page.get_upload_url(wav_filename, 600)
                )
            )

        print(uf)
        self.pick_file_dialog.upload(uf)

    def save_results(
        self,
        save_csv_path: str,
        picked_wav_file_path_list: List[Path],
        turn_list: List[Turn],
        grid_list: List[TextGrid],
        measure_list: List[List[float]],
        measure_names: List[str]
    ) -> None:
        save_csv_path = Path(save_csv_path)

        for turn, grid, wav_path in zip(turn_list, grid_list, picked_wav_file_path_list):
            save_txt_path = save_csv_path.parent / f"{wav_path.stem}_annotated.txt"
            save_grid_path = save_csv_path.parent / f"{wav_path.stem}.TextGrid"

            save_turn(turn, save_txt_path)
            save_grid(grid, save_grid_path)

        df_measures = pd.DataFrame(measure_list, columns=measure_names)
        df_measures.to_csv(save_csv_path, index=False)

    def is_files_picked(
        self,
        picked_wav_file_path_list: List[Path],
        picked_txt_file_path_list: List[Path]
    ) -> bool:
        if len(picked_wav_file_path_list) == 0 or len(picked_txt_file_path_list) == 0:
            self.page.open(self.file_selection_warning_banner)
            return False
        return True

    def disable_control(self):
        self.select_button.disabled = True
        self.select_button.style = ft.ButtonStyle(color="#eeeeee")

        self.upload_button.disabled = True
        self.upload_button.style = ft.ButtonStyle(color="#eeeeee")

        self.annotate_button.disabled = True
        self.annotate_button.style = ft.ButtonStyle(bgcolor="#eeeeee")

        self.progress_ring.visible = True

        self.update()

    def enable_control(self):
        self.select_button.disabled = False
        self.select_button.style = ft.ButtonStyle(color=ft.colors.PRIMARY)

        self.upload_button.disabled = False
        self.upload_button.style = ft.ButtonStyle(color=ft.colors.PRIMARY)

        self.annotate_button.disabled = False
        self.annotate_button.style = ft.ButtonStyle(bgcolor=ft.colors.PRIMARY)

        self.progress_ring.visible = False

        self.update()

    def handle_exception(self, error_message: str) -> None:
        self.general_error_banner.update_error_message(error_message)
        self.pick_file_dialog.picked_wav_path_list = []
        self.pick_file_dialog.picked_txt_path_list = []
        self.selected_file_container.selected_file_list.controls = [
            ERROR_SELECTED_TEXT
        ]
        self.page.open(self.general_error_banner)
        self.page.update()

    def annotate(
        self,
        picked_wav_file_path_list: List[Path],
        picked_txt_file_path_list: List[Path]
    ) -> None:
        if not self.is_files_picked(picked_wav_file_path_list, picked_txt_file_path_list):
            return

        self.disable_control()

        try:
            turn_list, grid_list = self.annotator.annotate(
                picked_wav_file_path_list,
                picked_txt_file_path_list
            )

            measure_list, measure_names = self.annotator.extract(picked_wav_file_path_list, turn_list, grid_list)

            self.save_results(
                RESULT_CSV_PATH,
                picked_wav_file_path_list,
                turn_list,
                grid_list,
                measure_list,
                measure_names
            )
        except Exception:
            error_message = format_exc()
            self.handle_exception(error_message)
        else:
            self.selected_file_container.selected_file_list.controls.append(
                ft.Text(
                    "・ Annotation finished!",
                    theme_style=ft.TextThemeStyle.LABEL_LARGE,
                    color=ft.colors.LIGHT_GREEN
                )
            )
            self.selected_file_container.selected_file_list.controls.append(
                ft.Text(
                    "・ See Donwloads folder",
                    theme_style=ft.TextThemeStyle.LABEL_LARGE,
                    color=ft.colors.LIGHT_GREEN
                )
            )

            self.enable_control()

class AnnotatorLoadingProgressBar(ft.Row):
    def __init__(self):
        super().__init__()

        loading_progress_bar = ft.Column(
            controls=[
                ft.Text(
                    "Loading Fluency Feature Annotator...",
                    theme_style=ft.TextThemeStyle.TITLE_LARGE,
                    text_align=ft.TextAlign.CENTER
                ),
                ft.ProgressBar(width=650, bgcolor="#eeeeee")
            ],
            alignment=ft.MainAxisAlignment.CENTER
        )

        self.controls = [
            loading_progress_bar
        ]
        self.alignment = ft.MainAxisAlignment.CENTER

def main(page: ft.Page):
    page.scroll = "adaptive"

    general_error_banner = GeneralErrorBanner(
        lambda e: page.close(general_error_banner)
    )

    annotator_loading_progress_bar = AnnotatorLoadingProgressBar()
    page.add(annotator_loading_progress_bar)

    try:
        annotator = FluencyFeatureAnnotator()
    except Exception:
        error_message = format_exc()
        general_error_banner.update_error_message(error_message)
        page.open(general_error_banner)
    else:
        page.remove(annotator_loading_progress_bar)
        page.add(WavTxtFileManager(annotator))


app = ft.app(main, upload_dir="uploads")
