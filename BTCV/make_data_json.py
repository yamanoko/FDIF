import json
import os
from pathlib import Path


def create_btcv_dataset_json():
    """
    BTCVデータセットのデータ配置を表すJSONファイルを作成する
    """
    # ベースディレクトリの設定（スクリプトファイルの場所を基準にする）
    base_dir = Path(__file__).parent.resolve()
    raw_data_dir = base_dir / "RawData"
    training_img_dir = raw_data_dir / "Training" / "img"
    training_label_dir = raw_data_dir / "Training" / "label"
    testing_img_dir = raw_data_dir / "Testing" / "img"

    # データセットの基本情報
    dataset_info = {
        "description": "btcv yucheng",
        "labels": {
            "0": "background",
            "1": "spleen",
            "2": "rkid",
            "3": "lkid",
            "4": "gall",
            "5": "eso",
            "6": "liver",
            "7": "sto",
            "8": "aorta",
            "9": "IVC",
            "10": "veins",
            "11": "pancreas",
            "12": "rad",
            "13": "lad",
        },
        "licence": "yt",
        "modality": {"0": "CT"},
        "name": "btcv",
        "reference": "Vanderbilt University",
        "release": "1.0 06/08/2015",
        "tensorImageSize": "3D",
    }

    # テストデータのパス取得
    test_images = []
    if testing_img_dir.exists():
        test_files = sorted(
            [f for f in os.listdir(testing_img_dir) if f.endswith(".nii.gz")]
        )
        test_images = [str((testing_img_dir / file).resolve()) for file in test_files]

    # トレーニングデータのパス取得
    training_data = []
    validation_data = []

    if training_img_dir.exists() and training_label_dir.exists():
        # 画像ファイルを取得してソート
        img_files = sorted(
            [f for f in os.listdir(training_img_dir) if f.endswith(".nii.gz")]
        )

        # 各画像ファイルに対応するラベルファイルがあるかチェック
        for img_file in img_files:
            # img0001.nii.gz -> label0001.nii.gz
            label_file = img_file.replace("img", "label")
            label_path = training_label_dir / label_file

            if label_path.exists():
                data_entry = {
                    "image": str((training_img_dir / img_file).resolve()),
                    "label": str((training_label_dir / label_file).resolve()),
                }

                # img0035以降をvalidationに、それ以前をtrainingに分類
                # ファイル名から番号を取得
                img_number = int(img_file.split("img")[1].split(".")[0])

                if img_number >= 35:
                    validation_data.append(data_entry)
                else:
                    training_data.append(data_entry)

    # 最終的なデータセット構造
    dataset_info.update(
        {
            "numTest": len(test_images),
            "numTraining": len(training_data),
            "test": test_images,
            "training": training_data,
            "validation": validation_data,
        }
    )

    # JSONファイルとして出力
    output_file = base_dir / "dataset.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    print(f"Dataset JSON file created: {output_file.resolve()}")
    print(f"Number of training samples: {len(training_data)}")
    print(f"Number of validation samples: {len(validation_data)}")
    print(f"Number of test samples: {len(test_images)}")

    return dataset_info


if __name__ == "__main__":
    create_btcv_dataset_json()
