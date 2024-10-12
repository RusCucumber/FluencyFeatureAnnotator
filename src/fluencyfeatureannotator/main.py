from pathlib import Path
from typing import List, Optional

import flet as ft
import pandas as pd
from fluency_feature_annotator import FluencyFeatureAnnotator, TextGrid, Turn, save_grid, save_turn


class WavTxtFilePicker(ft.FilePicker):
    def __init__(self):
        super().__init__()

        self.on_result = self.pick_file_results
        self.picked_wav_path_list = []
        self.picked_txt_path_list = []

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
            # MEMO: Error 処理などを後ほど追加?
            return

        picked_wav_path_list = []
        picked_txt_path_list = []

        for file in e.files:
            file_path = Path(file.path)
            if file_path.suffix == ".wav":
                picked_wav_path_list.append(file_path)
            elif file_path.suffix == ".txt":
                picked_txt_path_list.append(file_path)

        if self.is_wav_txt_path_pair_picked(picked_wav_path_list, picked_txt_path_list):
            self.picked_wav_path_list = picked_wav_path_list
            self.picked_txt_path_list = picked_txt_path_list

class FileHandlingProgressBar(ft.Column):
    def __init__(
            self,
            initial_message: str =None,
            initial_progress: float =0.0,
            bar_width: int =400
        ) -> None:
        super().__init__()

        self.dialog = ft.Text(value=initial_message)
        self.progress_bar = ft.ProgressBar(value=initial_progress, width=bar_width)

        self.controls = [
            self.dialog,
            self.progress_bar
        ]

    def update_value(self, message: Optional[str] =None, progress: Optional[float] =None) -> None:
        if message:
            self.dialog.value = message
        if progress:
            self.progress_bar.value = progress

        super().update()

class WavTxtFileManager(ft.Column):
    def __init__(self, annotator: FluencyFeatureAnnotator):
        super().__init__()

        self.annotator = annotator

        self.pick_file_dialog = WavTxtFilePicker()

        self.save_file_dialog = ft.FilePicker(
            on_result= lambda e: self.annotate(
                e,
                self.pick_file_dialog.picked_wav_path_list,
                self.pick_file_dialog.picked_txt_path_list
            )
        )

        self.progress_bar = FileHandlingProgressBar()

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
            ft.Stack(controls=[
                self.save_file_dialog,
                self.annotate_button
            ]),
            self.progress_bar
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


def main(page: ft.Page):
    annotator = FluencyFeatureAnnotator()
    page.add(WavTxtFileManager(annotator))


ft.app(main)
