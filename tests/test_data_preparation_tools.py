import io
import json
import sys
from dataclasses import replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DATA = REPO_ROOT / "tools" / "data"
if str(TOOLS_DATA) not in sys.path:
    sys.path.insert(0, str(TOOLS_DATA))

import dataset_utils
import prepare_ref_adv_dataset
import prepare_refcoco_multi_dataset
import refexp_dataset_audit
from filter_refexp_train_overlap import TRAIN_FILES


def refexp_record(image_id: int, ann_id: int) -> dict:
    return {
        "image_id": image_id,
        "ann_id": ann_id,
        "sentences": [{"sent": f"object {ann_id}"}],
        "captions": [f"object {ann_id}"],
        "bbox": [0.0, 0.0, 10.0, 10.0],
        "raw_anns": {"bbox": [0.0, 0.0, 10.0, 10.0]},
    }


def small_refcoco_targets_and_rows():
    targets = []
    rows = {}
    next_safe_image = 100
    for target in prepare_refcoco_multi_dataset.TARGETS:
        if target.training:
            target_rows = [
                refexp_record(1, next_safe_image),
                refexp_record(next_safe_image, next_safe_image + 1),
            ]
            next_safe_image += 10
        else:
            target_rows = [refexp_record(1, next_safe_image)]
            next_safe_image += 10
        targets.append(replace(target, expected_rows=len(target_rows)))
        rows[(target.dataset, target.split)] = target_rows
    return tuple(targets), rows


def test_refcoco_preparation_filters_all_evaluation_images_and_audits(tmp_path):
    targets, source_rows = small_refcoco_targets_and_rows()

    def fetch_rows(dataset, split, **_kwargs):
        return source_rows[(dataset, split)]

    manifest_path = tmp_path / "results" / "overlap.json"
    manifest = prepare_refcoco_multi_dataset.prepare_refcoco_family(
        tmp_path / "data",
        manifest_path,
        targets=targets,
        fetch_rows=fetch_rows,
    )

    assert manifest["hypothesis_pass"] is True
    assert manifest["evaluation_unique_images_union"] == 1
    for relative_path in TRAIN_FILES.values():
        records = dataset_utils.load_json(tmp_path / "data" / relative_path)
        assert len(records) == 1
        assert records[0]["image_id"] != 1

    audit = refexp_dataset_audit.run_audit(tmp_path / "data")
    assert audit["hypothesis_pass"] is True
    assert audit["summary"]["remaining_overlap_images"] == 0


def test_refcoco_audit_rejects_remaining_evaluation_image(tmp_path):
    targets, source_rows = small_refcoco_targets_and_rows()

    def fetch_rows(dataset, split, **_kwargs):
        return source_rows[(dataset, split)]

    data_root = tmp_path / "data"
    prepare_refcoco_multi_dataset.prepare_refcoco_family(
        data_root,
        tmp_path / "results" / "overlap.json",
        targets=targets,
        fetch_rows=fetch_rows,
    )
    train_path = data_root / "refcoco" / "refcoco" / "train.json"
    records = dataset_utils.load_json(train_path)
    records.append(refexp_record(1, 999))
    dataset_utils.write_json(train_path, records)

    audit = refexp_dataset_audit.run_audit(data_root)
    assert audit["hypothesis_pass"] is False
    assert audit["summary"]["remaining_overlap_images"] == 1


def test_ref_adv_conversion_writes_images_annotations_and_license(tmp_path):
    source = [
        {
            "image": {"src": "https://example.test/image.jpg"},
            "normal_caption": "the center screen",
            "solution": [10, 20, 110, 120],
            "width": 640,
            "height": 480,
            "distractors": "5",
            "image_source": "coco_val2017",
            "human_authored": True,
            "file_name": "source.jpg",
            "use_negation": False,
            "row_idx": 7,
            "image_file": "0007.jpg",
        }
    ]

    def fetch_rows(_dataset, _split, **_kwargs):
        return source

    def download(_url, destination, *, force=False):
        assert force is False
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"image")

    records, manifest = prepare_ref_adv_dataset.prepare_ref_adv(
        tmp_path / "data",
        expected_rows=1,
        fetch_rows=fetch_rows,
        download=download,
    )

    assert records[0]["image_path"] == "ref_adv/images/0007.jpg"
    assert records[0]["raw_anns"]["bbox"] == [10.0, 20.0, 100.0, 100.0]
    assert manifest["source"]["license"] == "CC-BY-4.0"
    assert (tmp_path / "data" / "ref_adv" / "images" / "0007.jpg").is_file()
    assert (tmp_path / "data" / "ref_adv" / "ref_adv_s.json").is_file()


def test_hugging_face_rows_client_paginates(monkeypatch):
    pages = [
        {"rows": [{"row": {"id": 1}}, {"row": {"id": 2}}]},
        {"rows": [{"row": {"id": 3}}]},
    ]

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    def urlopen(_request, timeout):
        assert timeout == 60
        return Response(json.dumps(pages.pop(0)).encode())

    monkeypatch.setattr(dataset_utils.urllib.request, "urlopen", urlopen)
    rows = dataset_utils.fetch_hf_rows(
        "owner/dataset",
        "train",
        chunk_size=2,
        sleep_sec=0,
    )
    assert rows == [{"id": 1}, {"id": 2}, {"id": 3}]
