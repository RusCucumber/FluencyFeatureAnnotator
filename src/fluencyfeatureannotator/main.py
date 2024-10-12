from pathlib import Path
from typing import List

import flet as ft
from fluency_feature_annotator import FluencyFeatureAnnotator


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
                file_name="result"
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
            ])
        ]

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

        # TODO: save grid, pruned/unpruned txt, & UF measures
        return

def main(page: ft.Page):
    annotator = FluencyFeatureAnnotator()
    page.add(WavTxtFileManager(annotator))


ft.app(main)
