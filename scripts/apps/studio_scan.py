"""Viewpoint Studio 의 Scan 패널 — Isaac 이 찍은 depth 스캔을 확인하고, 다시 복원하고, 내보낸다.

스캔(계획·촬영)과 기본 복원은 isaac_pipeline 의 "Scan Object" 패널이 한다. 여기는 그 결과
data/{object}/scan/{run}/ 을 보는 곳이다:
  보기     복원과 같은 판정(core.reconstruction.segment)으로 나눈 점 — 물체로 남긴 점(프레임별 색),
           장애물이라 지운 점(빨강), ROI 밖이라 지운 점(회색) — 과 카메라·ROI·복원 메시·정답
  재복원   파라미터를 바꿔 core.reconstruction.pipeline.reconstruct 를 다시 돌린다(재촬영 없이)
  내보내기 복원 메시를 정답 물체 좌표의 mesh/recon.obj 로 → Target 메시로 골라 바로 Generate

좌표: 스캔은 world(로봇 베이스) 좌표다. 스튜디오는 ``/scene`` 프레임에 물체 배치 회전 R_p 만
입히고 자식은 물체 좌표로 그린다. 그래서 스캔은 자기 루트 ``/scan`` 에 T = [R_p|0]·inv(T_s) 를
걸고 자식을 world 좌표 그대로 둔다 — T_s 는 스캔 당시의 물체 pose(계획에 남긴 값). 기즈모로 옮긴
물체를 찍었어도 화면의 물체와 겹친다.

이 패널의 노드는 ``studio.layers`` 에 넣지 않는다 — 스튜디오는 hops 슬라이더·그래프 빌드마다
레이어를 지우고 다시 그리는데, 수십만 점을 그때마다 다시 보낼 이유가 없다. 물체가 바뀔 때만
스스로 지운다. 불러오기는 자기 스레드(버전 토큰으로 낡은 결과 폐기)에서, 재복원은 스튜디오의
``generating`` 잠금을 같이 쓰는 워커에서 돈다.
"""

from __future__ import annotations

import colorsys
import contextlib
import json
import threading
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from common import config
from core.reconstruction import sim, storage
from core.reconstruction.pipeline import FusionParams, reconstruct, recon_report
from core.reconstruction.segment import OBJECT, OBSTACLE, OUTSIDE, classify_pixels

NONE = "(none)"
OBSTACLE_RGB = (230, 70, 70)       # 씬 장애물이라 지운 점
OUTSIDE_RGB = (130, 130, 130)      # ROI 밖이라 지운 점
RECON_RGB = (90, 200, 120)
GT_RGB = (200, 200, 200)
POINT_STRIDE = 4                    # depth 픽셀 간격 — 1224×1024 / 16 ≈ 7.8만 점/장
KEPT_POINTS_TOTAL = 300_000         # 브라우저로 보내는 점의 총량 상한
REMOVED_POINTS_TOTAL = 100_000
POINT_SIZE = 0.0015


def _frame_colors(n: int) -> np.ndarray:
    return np.array([[int(255 * c) for c in colorsys.hsv_to_rgb(i / max(n, 1), 0.75, 0.95)]
                     for i in range(n)], dtype=np.uint8)


def _wxyz(R: np.ndarray) -> tuple:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return (float(w), float(x), float(y), float(z))


class ScanPanel:
    def __init__(self, studio):
        self.studio = studio
        self._lock = threading.RLock()
        self._version = 0                       # run 을 새로 불러올 때마다 +1 — 낡은 스레드 결과 폐기
        self._handles: dict[str, list] = {k: [] for k in
                                          ("root", "kept", "obstacle", "outside", "camera",
                                           "roi", "gt", "recon")}
        self.obj: str | None = None
        self.run: str | None = None
        self.plan: storage.ScanPlan | None = None
        self.n_frames = 0
        self._margin_drawn: float | None = None   # 점을 나눌 때 쓴 장애물 여유(mm)

    # ------------------------------------------------------------------ GUI
    def build_gui(self, g) -> None:
        d = FusionParams()
        with g.add_folder("Scan (도면 없는 복원)"):
            self.run_dd = g.add_dropdown("Run", options=[NONE], initial_value=NONE,
                                         hint="data/{object}/scan/{run} — Isaac 의 Scan Object 가 만든다")
            self.recon_dd = g.add_dropdown("Recon", options=[NONE], initial_value=NONE,
                                           hint="recon{_tag}.ply — 기본 복원은 태그 없음")
            self.status_md = g.add_markdown("")
            self.sl_frame = g.add_slider("Frame (−1 = all)", min=-1, max=0, step=1, initial_value=-1)
            self.cb_kept = g.add_checkbox("Object points (kept)", initial_value=True)
            self.cb_obstacle = g.add_checkbox("Removed: obstacles (red)", initial_value=True)
            self.cb_outside = g.add_checkbox("Removed: outside ROI (gray)", initial_value=False)
            self.cb_camera = g.add_checkbox("Cameras", initial_value=True)
            self.cb_roi = g.add_checkbox("ROI sphere", initial_value=False)
            self.cb_gt = g.add_checkbox("Ground truth (source.obj)", initial_value=False)
            self.cb_recon = g.add_checkbox("Reconstruction", initial_value=True)
            with g.add_folder("Reconstruct again"):
                self.nb_voxel = g.add_number("Voxel (mm)", initial_value=d.voxel_mm, min=0.2, max=5.0,
                                             step=0.1)
                self.nb_trunc = g.add_number("Trunc (× voxel)", initial_value=d.trunc_voxels, min=1.0,
                                             max=10.0, step=0.5)
                self.nb_margin = g.add_number("Obstacle margin (mm)", initial_value=d.obstacle_margin_mm,
                                              min=0.0, max=20.0, step=0.5,
                                              hint="씬 장애물 상자를 이만큼 부풀려 그 안의 점을 지운다")
                self.nb_smooth = g.add_number("Smooth iters", initial_value=d.smooth_iters, min=0,
                                              max=100, step=1)
                self.nb_faces = g.add_number("Target faces (0 = all)", initial_value=d.target_faces,
                                             min=0, max=500_000, step=1000)
                self.cb_icp = g.add_checkbox("ICP pose refine", initial_value=False)
                self.tb_tag = g.add_text("Tag", initial_value="",
                                         hint="저장 이름 recon_{tag}.ply — 비우면 기본 recon.ply 를 덮는다")
                self.btn_recon = g.add_button("Reconstruct")
                self.recon_md = g.add_markdown("")
            self.btn_export = g.add_button(
                "Export recon.obj", hint="정답 물체 좌표로 data/{object}/mesh/recon.obj — Target 메시로 고른다")
            self.export_md = g.add_markdown("")

        self.run_dd.on_update(lambda _: self._on_run_change())
        self.recon_dd.on_update(lambda _: self._on_recon_change())
        self.sl_frame.on_update(lambda _: self._apply_visibility())
        for cb in (self.cb_kept, self.cb_obstacle, self.cb_outside, self.cb_camera, self.cb_roi,
                   self.cb_gt, self.cb_recon):
            cb.on_update(lambda _: self._apply_visibility())
        self.btn_recon.on_click(lambda _: self._on_reconstruct())
        self.btn_export.on_click(self._on_export_click)
        self.set_enabled(True)

    def set_enabled(self, enabled: bool) -> None:
        """스튜디오 ``_set_buttons`` 가 함께 부른다."""
        has_run = self.plan is not None
        for button, extra in ((self.btn_recon, has_run),
                              (self.btn_export, has_run and self.recon_dd.value != NONE)):
            with contextlib.suppress(Exception):
                button.disabled = not (enabled and extra)

    # ------------------------------------------------------------ studio hooks
    def on_object_change(self, obj: str) -> None:
        with self._lock:
            self.obj = obj
            runs = storage.list_runs(obj, self.studio.data_root)
            self.run_dd.options = [NONE, *runs]
            # 같은 값으로 두면 on_update 가 안 온다 — 직접 부른다.
            target = runs[0] if runs else NONE
            if self.run_dd.value != target:
                self.run_dd.value = target          # → _on_run_change
            else:
                self._on_run_change()

    # ------------------------------------------------------------ loading
    def _scan_dir(self) -> Path:
        return Path(self.studio.data_root) / self.obj / "scan" / self.run

    def _clear(self, keys=None) -> None:
        for key in keys or list(self._handles):
            layer = self._handles[key]
            while layer:
                with contextlib.suppress(Exception):
                    layer.pop().remove()

    def _on_run_change(self) -> None:
        with self._lock:
            self._version += 1
            version = self._version
            # 자식부터 지운다 — 부모 프레임을 먼저 지우면 클라이언트가 자식을 같이 버린다.
            self._clear([k for k in self._handles if k != "root"])
            self._clear(["root"])
            self.plan, self.n_frames = None, 0
            self.run = self.run_dd.value if self.run_dd.value != NONE else None
            self.recon_dd.options = [NONE]
            self.recon_dd.value = NONE
            self.export_md.content = ""
            self.recon_md.content = ""
            if self.run is None:
                self.status_md.content = ("스캔 없음 — Isaac Pipeline 의 **Scan Object** 로 찍으면 "
                                          "여기 나타난다." if self.obj else "")
                self.set_enabled(not self.studio.generating)
                return
            self.status_md.content = f"⏳ `{self.run}` 불러오는 중…"
        threading.Thread(target=self._load, args=(version, self.obj, self.run), daemon=True).start()

    def _stale(self, version: int) -> bool:
        return version != self._version

    def _load(self, version: int, obj: str, run: str) -> None:
        try:
            scan = Path(self.studio.data_root) / obj / "scan" / run
            plan = storage.load_plan(scan)
            names, poses = storage.load_frame_poses(scan)
            tags = storage.list_recon_tags(scan)
            with self._lock:
                if self._stale(version):
                    return
                self.plan, self.n_frames = plan, len(names)
                self._draw_root(obj, plan)
                self._draw_cameras(names, poses, plan)
                self._draw_roi(plan)
                self._draw_ground_truth(obj, plan)
                self.sl_frame.max = max(len(names) - 1, 0)
                self.sl_frame.value = -1
                self.recon_dd.options = [NONE] + [t or "(default)" for t in tags]
                self.recon_dd.value = ("(default)" if "" in tags else (tags[0] if tags else NONE))
                self.status_md.content = self._status_text(plan, len(names), classifying=True)
                self.set_enabled(not self.studio.generating)
                self._apply_visibility()
            self._classify_and_draw(version, scan, plan, float(self.nb_margin.value))
            with self._lock:
                if not self._stale(version):
                    self.status_md.content = self._status_text(plan, len(names))
        except Exception as exc:  # noqa: BLE001
            if not self._stale(version):
                self.status_md.content = f"**Error:** `{run}` 를 못 읽었다 — {str(exc)[:300]}"
            print(f"[scan] load error: {exc}")

    def _status_text(self, plan, n_frames: int, classifying: bool = False) -> str:
        meta = plan.meta
        bits = [f"**{n_frames}** frames · ROI r {plan.roi_radius * 1000:.0f} mm · "
                f"standoff {plan.standoff * 1000:.0f} mm"]
        if plan.object_pose is None:
            bits.append("옛 계획 — 장애물·물체 pose 는 지금 씬 설정에서")
        elif meta.get("object_pose_source") == "cli":
            bits.append("물체 pose: Isaac 스테이지")
        if meta.get("scene") and meta["scene"] != config.ACTIVE_SCENE and plan.obstacles is None:
            bits.append(f"⚠️ 스캔 씬 `{meta['scene']}` ≠ 지금 `{config.ACTIVE_SCENE}`")
        if classifying:
            bits.append("⏳ 점 분류 중…")
        return " · ".join(bits)

    # ------------------------------------------------------------ drawing
    def _scan_root_transform(self, obj: str, plan) -> tuple:
        """/scan 루트의 (wxyz, position) — T = [R_p|0]·inv(T_s). 모듈 docstring 참고."""
        config.apply_object_placement(obj)
        R_p = sim.world_T_object()[:3, :3]                    # 스튜디오 /scene 이 쓰는 회전
        T_s = sim.world_T_object(sim.scan_object_pose(plan))  # 스캔 당시 pose (없으면 지금 배치)
        R = R_p @ T_s[:3, :3].T
        return _wxyz(R), tuple(float(v) for v in -R @ T_s[:3, 3])

    def _draw_root(self, obj: str, plan) -> None:
        wxyz, position = self._scan_root_transform(obj, plan)
        self._handles["root"].append(self.studio.server.scene.add_frame(
            "/scan", show_axes=False, wxyz=wxyz, position=position))

    def _draw_cameras(self, names, poses, plan) -> None:
        srv, spec = self.studio.server, plan.camera
        colors = _frame_colors(len(names))
        for i, (name, T) in enumerate(zip(names, poses)):
            self._handles["camera"].append(srv.scene.add_camera_frustum(
                f"/scan/frames/{name}/camera", fov=np.radians(spec.vfov_deg),
                aspect=spec.width / spec.height, scale=0.03,
                color=tuple(int(c) for c in colors[i]), wxyz=_wxyz(T[:3, :3]),
                position=tuple(float(v) for v in T[:3, 3])))

    def _draw_roi(self, plan) -> None:
        import trimesh

        sphere = trimesh.creation.icosphere(subdivisions=2, radius=float(plan.roi_radius))
        self._handles["roi"].append(self.studio.server.scene.add_mesh_simple(
            "/scan/roi", vertices=(sphere.vertices + plan.roi_center).astype(np.float32),
            faces=sphere.faces.astype(np.int32), color=(80, 160, 255), opacity=0.08))

    def _draw_ground_truth(self, obj: str, plan) -> None:
        try:
            gt = sim.ground_truth_mesh(obj, sim.scan_object_pose(plan))
        except Exception as exc:  # noqa: BLE001 — 정답이 없으면 안 그릴 뿐
            print(f"[scan] ground truth unavailable: {exc}")
            return
        self._handles["gt"].append(self.studio.server.scene.add_mesh_simple(
            "/scan/ground_truth", vertices=gt.vertices.astype(np.float32),
            faces=gt.faces.astype(np.int32), color=GT_RGB, opacity=0.35))

    def _draw_recon(self, mesh) -> None:
        self._clear(["recon"])
        if mesh is None:
            return
        self._handles["recon"].append(self.studio.server.scene.add_mesh_simple(
            "/scan/reconstruction", vertices=mesh.vertices.astype(np.float32),
            faces=mesh.faces.astype(np.int32), color=RECON_RGB, opacity=1.0))
        self._apply_visibility()

    def _classify_and_draw(self, version: int, scan: Path, plan, margin_mm: float) -> None:
        """프레임별로 depth 를 풀어 복원과 같은 판정으로 나누고 그린다(장마다 바로 보낸다)."""
        files = sorted(storage.frames_dir(scan).glob("frame_*.npz"))
        n = max(len(files), 1)
        caps = {"kept": KEPT_POINTS_TOTAL // n, "obstacle": REMOVED_POINTS_TOTAL // n,
                "outside": REMOVED_POINTS_TOTAL // n}
        colors = _frame_colors(len(files))
        rng = np.random.default_rng(0)
        new = {"kept": [], "obstacle": [], "outside": []}
        for i, f in enumerate(files):
            if self._stale(version):
                return
            with np.load(f) as d:
                pts, labels = classify_pixels(d["depth"], d["K"], d["world_T_cam"], plan.camera,
                                              plan.roi_center, plan.roi_radius, margin_mm / 1000.0,
                                              stride=POINT_STRIDE, obstacles=plan.obstacles)
            for key, lab, rgb in (("kept", OBJECT, colors[i]), ("obstacle", OBSTACLE, OBSTACLE_RGB),
                                  ("outside", OUTSIDE, OUTSIDE_RGB)):
                p = pts[labels == lab]
                if len(p) > caps[key]:
                    p = p[rng.choice(len(p), caps[key], replace=False)]
                if len(p):
                    new[key].append((i, self.studio.server.scene.add_point_cloud(
                        f"/scan/frames/{f.stem}/{key}", points=p.astype(np.float32), colors=rgb,
                        point_size=POINT_SIZE)))
        with self._lock:
            if self._stale(version):
                for items in new.values():
                    for _, h in items:
                        with contextlib.suppress(Exception):
                            h.remove()
                return
            self._clear(["kept", "obstacle", "outside"])
            for key, items in new.items():
                self._handles[key] = [h for _, h in items]
                for i, h in items:
                    h._scan_frame = i               # 프레임 슬라이더용
            self._margin_drawn = margin_mm
            self._apply_visibility()

    def _apply_visibility(self) -> None:
        toggles = {"kept": self.cb_kept, "obstacle": self.cb_obstacle, "outside": self.cb_outside,
                   "camera": self.cb_camera, "roi": self.cb_roi, "gt": self.cb_gt,
                   "recon": self.cb_recon}
        frame = int(self.sl_frame.value)
        for key, cb in toggles.items():
            for i, h in enumerate(self._handles[key]):
                idx = getattr(h, "_scan_frame", i) if key in ("kept", "obstacle", "outside", "camera") else None
                with contextlib.suppress(Exception):
                    h.visible = bool(cb.value) and (idx is None or frame < 0 or idx == frame)

    # ------------------------------------------------------------ recon tag
    def _current_tag(self) -> str | None:
        v = self.recon_dd.value
        return None if v == NONE else ("" if v == "(default)" else v)

    def _on_recon_change(self) -> None:
        tag = self._current_tag()
        self.set_enabled(not self.studio.generating)
        if self.run is None or tag is None:
            self._draw_recon(None)
            return
        scan = self._scan_dir()
        mesh = storage.load_recon(scan, tag)
        self._draw_recon(mesh)
        report_path = scan / f"{storage.recon_stem(tag)}.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            self._fill_params(report)
            self.recon_md.content = self._report_text(report)

    def _fill_params(self, report: dict) -> None:
        for widget, key in ((self.nb_voxel, "voxel_mm"), (self.nb_trunc, "trunc_voxels"),
                            (self.nb_margin, "obstacle_margin_mm"), (self.nb_smooth, "smooth_iters"),
                            (self.nb_faces, "target_faces")):
            if key in report:
                widget.value = report[key]
        self.cb_icp.value = bool(report.get("icp", False))

    @staticmethod
    def _report_text(report: dict) -> str:
        text = (f"{report.get('faces', '?')} faces · voxel {report.get('voxel_mm')} mm"
                + (" · ICP" if report.get("icp") else ""))
        e = report.get("eval")
        if e:
            text += (f"  \n정답 대비: 정확도 중앙값 **{e['accuracy_mm']['median']:.2f} mm** · "
                     f"p95 {e['accuracy_mm']['p95']:.2f} mm · ≤1mm **{e['completeness']['within_1mm'] * 100:.1f}%** · "
                     f"법선 중앙값 {e['normal_deg']['median']:.1f}°")
        return text

    # ------------------------------------------------------------ actions
    def _run(self, status_md, busy_text: str, fn) -> None:
        """스튜디오와 같은 방식의 워커 — 바쁨 표시, 버튼 잠금, 물체가 바뀌면 결과 폐기."""
        st = self.studio
        if st.generating:
            return
        st.generating = True
        st._set_buttons(False)
        status_md.content = busy_text
        obj, run = self.obj, self.run

        def work():
            try:
                fn(obj, run)
            except Exception as exc:  # noqa: BLE001
                status_md.content = f"**Error:** {str(exc)[:300]}"
                print(f"[scan] error: {exc}")
            finally:
                st.generating = False
                st._set_buttons(True)

        threading.Thread(target=work, daemon=True).start()

    def _changed(self, obj: str, run: str, status_md) -> bool:
        if obj != self.obj or run != self.run:
            status_md.content = "**Discarded** — 작업 중 Object/Run 이 바뀌었다."
            return True
        return False

    def _on_reconstruct(self) -> None:
        if self.plan is None:
            return
        params = FusionParams(voxel_mm=float(self.nb_voxel.value), trunc_voxels=float(self.nb_trunc.value),
                              obstacle_margin_mm=float(self.nb_margin.value),
                              smooth_iters=int(self.nb_smooth.value), target_faces=int(self.nb_faces.value))
        icp = bool(self.cb_icp.value)
        tag = self.tb_tag.value.strip()
        if tag and not all(c.isalnum() or c in "_.-" for c in tag):
            self.recon_md.content = "**Tag** 는 영문·숫자·`_ . -` 만."
            return

        def work(obj, run):
            scan = Path(self.studio.data_root) / obj / "scan" / run
            plan = storage.load_plan(scan)
            frames = storage.load_frames(scan)
            lines: list[str] = []

            def log(msg):
                lines.append(str(msg))
                self.recon_md.content = "⏳ " + "  \n".join(lines[-3:])

            result = reconstruct(frames, plan, params, icp=icp, log=log)
            report = recon_report(params, result, icp)
            try:
                report.update(sim.eval_report(result.mesh, sim.ground_truth_mesh(obj, sim.scan_object_pose(plan))))
            except Exception as exc:  # noqa: BLE001 — 정답이 없으면 평가 없이 저장한다
                print(f"[scan] eval skipped: {exc}")
            if self._changed(obj, run, self.recon_md):
                return
            storage.save_recon(scan, result.mesh, report, tag=tag)
            with self._lock:
                tags = storage.list_recon_tags(scan)
                self.recon_dd.options = [NONE] + [t or "(default)" for t in tags]
                self.recon_dd.value = tag or "(default)"   # → _on_recon_change 가 그리고 수치를 채운다
                self._on_recon_change()
            self.recon_md.content = "**Saved** · " + self._report_text(report)
            # 점 분류를 이번 여유로 다시 — 화면의 지운 점이 방금 복원과 같아야 한다.
            if self._margin_drawn != params.obstacle_margin_mm:
                self._classify_and_draw(self._version, scan, plan, params.obstacle_margin_mm)

        self._run(self.recon_md, "⏳ 복원 중…", work)

    def _on_export_click(self, event) -> None:
        tag = self._current_tag()
        if self.plan is None or tag is None:
            self.export_md.content = "내보낼 복원이 없다."
            return
        out = Path(self.studio.data_root) / self.obj / "mesh" / "recon.obj"
        client = getattr(event, "client", None)
        if not out.exists() or client is None:
            self._start_export(tag)
            return
        with client.gui.add_modal("recon.obj 를 덮어씁니다") as modal:
            client.gui.add_markdown(
                f"`data/{self.obj}/mesh/recon.obj` 가 이미 있다. 지금 고른 복원(`{self.recon_dd.value}`)"
                f"으로 덮어쓴다.")
            ok = client.gui.add_button("덮어쓰기", color="red")
            cancel = client.gui.add_button("취소")

            @ok.on_click
            def _(_):
                modal.close()
                self._start_export(tag)

            @cancel.on_click
            def _(_):
                modal.close()

    def _start_export(self, tag: str) -> None:
        def work(obj, run):
            scan = Path(self.studio.data_root) / obj / "scan" / run
            plan = storage.load_plan(scan)
            mesh = storage.load_recon(scan, tag)
            if mesh is None:
                raise FileNotFoundError(f"{storage.recon_stem(tag)}.ply 가 없다")
            out, bbox = sim.export_gt_frame(mesh, obj, sim.scan_object_pose(plan), "recon.obj")
            if self._changed(obj, run, self.export_md):
                return
            self._adopt_export(obj, out.name)
            self.export_md.content = (f"**Exported** → `data/{obj}/mesh/{out.name}` · Target 메시로 "
                                      f"골랐다 — 이제 **Generate**.  \n```\n{bbox}\n```")

        self._run(self.export_md, "⏳ 내보내는 중…", work)

    def _adopt_export(self, obj: str, name: str) -> None:
        """내보낸 메시를 Target 으로 고른다. 같은 파일 이름의 캐시는 버린다(옛 기하가 남는다)."""
        st = self.studio
        for cache in (st.mesh_cache, st.surface_cache, st.adjacency_cache):
            for key in [k for k in cache if isinstance(k, tuple) and k and k[0] == obj and name in k]:
                cache.pop(key, None)
        st._refresh_mesh_options()
        if name in (st.dd_mesh.options or ()):
            st.dd_mesh.value = name          # → _on_mesh_source_change 가 재질 칸을 물체 기본값으로 채운다
            st.tb_material.value = ""        # 복원 메시는 재질이 없다 — 남겨 두면 재질 매칭에서 실패한다
