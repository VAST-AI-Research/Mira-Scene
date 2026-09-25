"""Run both CLI entry points against self-contained, small archive fixtures."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mira extract ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / "downloaded data"
        self.destination = self.root / "extracted data"
        self.package.mkdir()
        (self.package / "manifests").mkdir()
        (self.package / "README.md").write_text("Fixture dataset\n")
        (self.package / "manifests/integrity_summary.json").write_text(
            json.dumps({"status": "PASS"})
        )
        self.payload = {
            "objaverse_outpaint/meshes/object/mesh_normalized.ply": b"mesh",
            "objaverse_outpaint/valid_scenes/object_000/scene.png": b"rgb",
            "objaverse_outpaint/valid_scenes/object_000/mask.png": b"mask",
            "objaverse_outpaint/valid_scenes/object_000/depth.npy": b"depth",
            "3dfront/renderings/scene/view.hdf5": b"hdf5",
            "3dfront/view_samples/view/depth.npy": b"depth",
            "3dfront/poses/scene_scene_state.json": b"{}",
            "3dfront/models/model/raw_model.obj": b"mesh",
        }
        for subset in ("objaverse_outpaint", "3dfront"):
            (self.package / subset / "shards").mkdir(parents=True)
        with gzip.open(self.package / "objaverse_outpaint/metadata.jsonl.gz", "wt") as f:
            f.write(json.dumps({"obj_id": "object", "view": 0,
                               "mesh_path": "objaverse_outpaint/meshes/object/mesh_normalized.ply"}) + "\n")
        self.front = {"scene_results": [{"scene_id": "scene", "obj_keys": ["x|model"]}],
                      "results": [{"scene_id": "scene", "id": "view",
                                   "view_relpath": "view.hdf5", "valid_object_indices": [0]}]}
        (self.package / "3dfront/preprocess_train.json").write_text(json.dumps(self.front))
        self.build_archives()

    def build_archives(self):
        shards = []
        for subset in ("objaverse_outpaint", "3dfront"):
            archive = self.package / subset / "shards/part-00000.tar.gz"
            rows = []
            with tarfile.open(archive, "w:gz") as tar:
                for name, data in self.payload.items():
                    if not name.startswith(subset + "/"):
                        continue
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
                    rows.append({"archive": name, "size": len(data),
                                 "sha256": hashlib.sha256(data).hexdigest()})
            members = self.package / "manifests" / (subset + ".jsonl.gz")
            with gzip.open(members, "wt") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            shards.append({"path": str(archive.relative_to(self.package)),
                           "members": str(members.relative_to(self.package)),
                           "sha256": digest(archive), "members_sha256": digest(members)})
        (self.package / "manifests/shards.json").write_text(json.dumps(shards))

    def run_cli(self, script, *extra, success=True):
        # The current working directory is deliberately outside the code tree.
        result = subprocess.run(
            [sys.executable, str(TOOLS / script), str(self.package),
             str(self.destination), *extra], cwd=self.root,
            text=True, capture_output=True,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def assert_payload(self):
        for name, data in self.payload.items():
            self.assertEqual((self.destination / name).read_bytes(), data)

    def test_full_extract_and_resume(self):
        self.run_cli("extract.py")
        self.assert_payload()
        target = self.destination / next(iter(self.payload))
        before = target.stat().st_mtime_ns
        self.run_cli("extract.py")
        self.assertEqual(target.stat().st_mtime_ns, before)
        self.assert_payload()

    def test_subset_then_other_subset(self):
        self.run_cli("extract.py", "--subset", "3dfront")
        self.assertFalse((self.destination / "objaverse_outpaint").exists())
        self.run_cli("extract.py", "--subset", "objaverse_outpaint")
        self.assert_payload()

    def test_sample_extract_and_reject_nonempty_destination(self):
        self.run_cli("extract_sample.py", "--views-per-subset", "1")
        self.assert_payload()
        report = json.loads((self.destination / "SAMPLE_MANIFEST.json").read_text())
        self.assertEqual(report["views"], {"3dfront": 1, "objaverse_outpaint": 1})
        self.assertEqual(report["payload_files"], len(self.payload))
        self.assertEqual(json.loads((self.destination / "3dfront/preprocess_train.json").read_text()), self.front)
        result = self.run_cli("extract_sample.py", success=False)
        self.assertIn("empty destination", result.stderr)

    def test_corrupted_archive_rejected_by_both_scripts(self):
        archive = self.package / "objaverse_outpaint/shards/part-00000.tar.gz"
        archive.write_bytes(archive.read_bytes() + b"corrupt")
        for script in ("extract.py", "extract_sample.py"):
            self.destination = self.root / script
            result = self.run_cli(script, success=False)
            self.assertTrue("checksum" in result.stderr or "Archive corrupted" in result.stderr)

    def test_resume_rejects_modified_payload(self):
        self.run_cli("extract.py")
        (self.destination / next(iter(self.payload))).write_bytes(b"changed")
        self.assertIn("Existing file differs", self.run_cli("extract.py", success=False).stderr)

    def test_rejects_escaping_member(self):
        self.payload["objaverse_outpaint/../../escaped"] = b"unsafe"
        self.build_archives()
        self.assertIn("Unsafe archive path", self.run_cli("extract.py", success=False).stderr)
        self.assertFalse((self.root / "escaped").exists())

    def test_rejects_escaping_manifest_path(self):
        manifest = self.package / "manifests/shards.json"
        shards = json.loads(manifest.read_text())
        shards[0]["members"] = "../outside.jsonl.gz"
        manifest.write_text(json.dumps(shards))
        for script in ("extract.py", "extract_sample.py"):
            self.destination = self.root / script
            self.assertIn("Unsafe", self.run_cli(script, success=False).stderr)


if __name__ == "__main__":
    unittest.main()
