"""Rendering contracts checked against an existing saved trial, when available."""

import json

import numpy as np
import pytest
from matplotlib.text import Annotation
from PIL import Image

from kalman_hungarian_object_tracking import explained_animation_v2 as animation
from kalman_hungarian_object_tracking.io import ROOT, sha256


@pytest.fixture(scope="module")
def existing_replay():
    directory = ROOT / "outputs/benchmark"
    if not (directory / "metadata.json").exists():
        pytest.skip("existing benchmark archive is not available; no data are generated")
    return animation.reconstruct_saved_trial(directory, seed=100)


def test_rendered_coordinates_labels_and_fixed_axes(existing_replay, monkeypatch):
    replay = existing_replay
    limits = animation.fixed_limits(replay)
    monkeypatch.setattr(animation, "_picture", lambda figure: figure)
    for frame_number in animation.DISPLAY_FRAMES:
        frame = replay.frames[frame_number]
        _, associations = animation.visible_associations(replay, frame_number)
        for phase in range(4):
            figure = animation.render_stage(replay, frame_number, phase, limits)
            try:
                axis = figure.axes[0]
                assert axis.get_xlim() == limits[0]
                assert axis.get_ylim() == limits[1]
                labels = [text for text in axis.texts if isinstance(text, Annotation)]
                if phase == 0:
                    assert not labels and not axis.patches and not axis.lines
                    continue
                estimates = frame.posteriors if phase == 3 else frame.priors
                assert {label.get_text() for label in labels} == {"ID 4", "ID 2"}
                for label in labels:
                    identity = int(label.get_text().split()[1])
                    np.testing.assert_array_equal(label.xy, estimates[identity].mean)
                figure.canvas.draw()
                renderer = figure.canvas.get_renderer()
                boxes = [label.get_bbox_patch().get_window_extent(renderer) for label in labels]
                assert not boxes[0].overlaps(boxes[1])
                boundary = axis.get_window_extent()
                for box in boxes:
                    assert boundary.contains(box.x0, box.y0)
                    assert boundary.contains(box.x1, box.y1)
                if phase == 2:
                    assert len(axis.lines) == len(associations) == 2
                    for line, association in zip(axis.lines, associations, strict=True):
                        expected = np.array(
                            [
                                frame.priors[association.track_id].mean,
                                [association.x, association.y],
                            ]
                        )
                        np.testing.assert_array_equal(line.get_xydata(), expected)
                else:
                    assert not axis.lines
                # The two ellipse artists (fill and border) share each actual state mean.
                for identity, artists in zip(
                    animation.IDENTITIES, (axis.patches[:2], axis.patches[2:]), strict=True
                ):
                    assert len(artists) == 2
                    for ellipse in artists:
                        np.testing.assert_array_equal(ellipse.center, estimates[identity].mean)
            finally:
                animation.plt.close(figure)


def test_compact_gif_and_original_preserved(existing_replay, output_tmp, monkeypatch):
    original = ROOT / "outputs/figures/kalman_hungarian_explained.gif"
    before = sha256(original) if original.exists() else None
    monkeypatch.setattr(animation, "reconstruct_saved_trial", lambda *a, **k: existing_replay)
    gif = output_tmp / "explained_v2.gif"
    review = output_tmp / "review"
    manifest = animation.render_explained_animation_v2(output=gif, review_dir=review)
    assert manifest["displayed_frames"] == (56, 60, 64)
    assert manifest["benchmark_rerun"] is False
    assert manifest["parameters_changed"] is False
    with Image.open(gif) as movie:
        assert movie.size == (900, 1200) and movie.n_frames == 13
        duration = 0
        for index, record in enumerate(manifest["slides"]):
            movie.seek(index)
            movie.load()
            assert movie.info["duration"] == record["duration_ms"]
            duration += movie.info["duration"]
        assert 15000 <= duration <= 20000
        assert duration == manifest["duration_ms"] == 16600
    if before is not None:
        assert sha256(original) == before == manifest["original_gif_sha256"]
    with Image.open(review / "previews/frame_060_stage_3_mobile.png") as preview:
        assert preview.size == (360, 480)
    saved = json.loads((review / "manifest.json").read_text())
    assert all(record["axis_limits"] == saved["axis_limits"] for record in saved["slides"][1:])


def test_original_destination_rejected():
    with pytest.raises(ValueError, match="preserve"):
        animation.render_explained_animation_v2(
            output="outputs/figures/kalman_hungarian_explained.gif"
        )
