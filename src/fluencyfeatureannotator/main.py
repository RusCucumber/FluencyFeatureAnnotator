from pathlib import Path
from typing import Callable, List

import flet as ft
import pandas as pd
from fluency_feature_annotator import FluencyFeatureAnnotator, TextGrid, Turn, save_grid, save_turn

NO_FILE_SELECTED_TEXT = ft.Text("・ No files are selected.", theme_style=ft.TextThemeStyle.LABEL_LARGE)

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

class WavTxtFilePicker(ft.FilePicker):
    def __init__(
        self,
        selected_file_container: SelectedFileContainer
    ):
        super().__init__()

        self.on_result = self.pick_file_results
        self.picked_wav_path_list = []
        self.picked_txt_path_list = []

        self.selected_file_container = selected_file_container
        self.file_selection_warning_banner = FileSelectionWarningBanner(on_click=self.close_banner)

    def close_banner(self, e):
        self.page.close(self.file_selection_warning_banner)

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

    def pick_file_results(self, e: ft.FilePickerResultEvent) -> None:
        if e is None:
            # ファイルが選択されなかった場合
            self.selected_file_container.selected_file_list.controls = [
                NO_FILE_SELECTED_TEXT
            ]
            self.page.update()
            return

        picked_wav_path_list = []
        picked_txt_path_list = []

        for file in e.files:
            file_path = Path(file.path)
            if file_path.suffix == ".wav":
                picked_wav_path_list.append(file_path)
            elif file_path.suffix == ".txt":
                picked_txt_path_list.append(file_path)

        if not self.is_wav_txt_path_pair_picked(picked_wav_path_list, picked_txt_path_list):
            self.page.open(self.file_selection_warning_banner)
            self.selected_file_container.selected_file_list.controls = [
                NO_FILE_SELECTED_TEXT
            ]
            self.page.update()
            return

        self.picked_wav_path_list = picked_wav_path_list
        self.picked_txt_path_list = picked_txt_path_list

        showing_file_list = []
        for idx, wav_path in enumerate(picked_wav_path_list):
            showing_file_list.append(ft.Text(
                f"・ [{idx:03}] {wav_path.name} - {wav_path.stem}.txt",
                theme_style=ft.TextThemeStyle.LABEL_LARGE
            ))
        self.selected_file_container.selected_file_list.controls = showing_file_list
        self.page.update()

class WavTxtFileManager(ft.Column):
    def __init__(self, annotator: FluencyFeatureAnnotator):
        super().__init__()

        self.annotator = annotator

        self.selected_file_container = SelectedFileContainer()
        self.pick_file_dialog = WavTxtFilePicker(self.selected_file_container)

        self.save_file_dialog = ft.FilePicker(
            on_result= lambda e: self.annotate(
                e,
                self.pick_file_dialog.picked_wav_path_list,
                self.pick_file_dialog.picked_txt_path_list
            )
        )

        self.progress_ring = ft.ProgressRing()
        self.progress_ring.visible = False

        self.select_button = ft.ElevatedButton(
            text="Select wav & txt files",
            icon=ft.icons.UPLOAD_FILE,
            on_click=lambda _: self.pick_file_dialog.pick_files(
                allow_multiple=True,
                allowed_extensions=["wav", "txt"]
            ),
            width=300
        )

        self.annotate_button = ft.FilledButton(
            text="Annotate fluency features",
            icon=ft.icons.MULTITRACK_AUDIO_ROUNDED,
            on_click=lambda _: self.save_file_dialog.save_file(
                file_name="result.csv",
                allowed_extensions=["csv"]
            ),
            width=300
        )

        self.controls = [
            ft.Stack(controls=[
                self.pick_file_dialog,
                self.select_button
            ]),
            self.selected_file_container,
            ft.Stack(controls=[
                self.save_file_dialog,
                self.annotate_button
            ]),
            self.progress_ring
        ]

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

    def annotate(
        self,
        e: ft.FilePickerResultEvent,
        picked_wav_file_path_list: List[Path],
        picked_txt_file_path_list: List[Path]
    ) -> None:
        self.annotate_button.disabled = True
        self.progress_ring.visible = True
        self.update()

        turn_list, grid_list = self.annotator.annotate(
            picked_wav_file_path_list,
            picked_txt_file_path_list
        )

        measure_list, measure_names = self.annotator.extract(turn_list, grid_list)

        self.save_results(
            e.path,
            picked_wav_file_path_list,
            turn_list,
            grid_list,
            measure_list,
            measure_names
        )

        self.annotate_button.disabled = False
        self.progress_ring.visible = False
        self.update()

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
    annotator_loading_progress_bar = AnnotatorLoadingProgressBar()
    page.add(annotator_loading_progress_bar)

    annotator = FluencyFeatureAnnotator()

    page.remove(annotator_loading_progress_bar)
    page.add(WavTxtFileManager(annotator))


ft.app(main)
