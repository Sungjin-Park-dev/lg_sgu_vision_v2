"""Viewpoint Studio 의 Defects 패널 — 스크래치를 배치하고, viewpoint 옆에 그리고, 어느 카메라가
보는지 센 뒤, 원하면 Isaac 용 ``source.usd`` 에 새긴다.

만드는 일은 ``setup/apply_scratch_normal.py`` 가 한다(CLI 와 같은 함수). 이 패널은 그 함수들을
부르고 결과를 화면에 올린다. 스크래치는 언제나 ``source.obj`` 좌표계로 들고 있고, 그릴 때와
가시성을 셀 때만 스튜디오 좌표로 옮긴다(``core.defects.geometry.FrameAlignment``).

바쁨 표시는 스튜디오의 ``generating`` 과 ``_set_buttons`` 를 그대로 쓴다. Blender 가 도는 동안
Generate 도 막히지만, 레이어 목록을 두 스레드가 동시에 건드리는 일은 없다.
"""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path

import numpy as np
from common import config
from core.defects import geometry as dg
from core.viewpoint import visibility

# 화면에는 흠집 모양만 그린다 — 판정(온전/부분/놓침)은 패널 목록의 점 색으로 읽는다.
STATE_DOT = {None: "⚪", "seen": "🟢", "split": "🟠", "missed": "🔴"}
PREVIEW_MESH_RGB = (180, 180, 180)
FALLBACK_RGB = (60, 60, 60)          # 흠집 PNG 가 없을 때 모양 대신 긋는 선
OCCLUSION_TOLERANCE_MM = 1.0          # ViewpointGenParams 기본값과 같다


class DefectsPanel:
    def __init__(self, studio):
        self.studio = studio
        try:
            from setup import apply_scratch_normal as tool
        except Exception as exc:  # noqa: BLE001 — 스크래치 도구가 없어도 스튜디오는 떠야 한다
            tool, self.tool_error = None, str(exc)
        else:
            self.tool_error = None
        self.tool = tool
        self.blender = tool.find_blender() if tool else None
        try:
            self.library_ok = bool(tool and tool.scratch_library())
        except Exception:  # noqa: BLE001
            self.library_ok = False

        self.scratches: list[dict] = []     # source.obj 좌표, SPEC_KEYS 만
        self.origin = "none"                 # none | file | planned
        self._version = 0                    # scratches 가 바뀔 때마다 +1 — 캐시 키
        self._surface_cache: dict[str, tuple] = {}
        self._source_full_cache: dict[str, object] = {}
        self._geom, self._geom_key = None, None
        self._look_cache: dict[tuple, dict] = {}
        self._align, self._align_key = None, None
        self.vis, self._vis_key = None, None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ GUI
    def build_gui(self, g) -> None:
        with g.add_folder("Defects (스크래치)"):
            self.state_md = g.add_markdown("스크래치 없음")
            self.nb_count = g.add_number("개수", initial_value=3, min=1, max=20, step=1)
            self.nb_seed = g.add_number("Seed", initial_value=0, min=0, max=1_000_000, step=1,
                                        hint="같은 seed 면 같은 배치 — CLI --random N --seed S 와 같다")
            self.nb_len_lo = g.add_number("길이 최소 (mm)", initial_value=20, min=2, max=200, step=1)
            self.nb_len_hi = g.add_number("길이 최대 (mm)", initial_value=30, min=2, max=200, step=1)
            self.nb_strength = g.add_number("Strength", initial_value=1.0, min=0.1, max=3.0,
                                            step=0.1, hint="홈 깊이 — 1.0 = 원본 노멀맵 그대로")
            self.btn_place = g.add_button("배치", hint="계획 + 화면 표시만 — 파일은 쓰지 않는다")
            self.btn_load = g.add_button("저장본 불러오기",
                                         hint="data/{object}/mesh/scratches.json")
            self.btn_check = g.add_button(
                "가시성 검사", hint="지금 화면의 viewpoint 로 — 각 스크래치를 한 장에 온전히 담는 "
                                   "카메라가 몇 대인가")
            self.vis_md = g.add_markdown("")
            with g.add_folder("USD 적용"):
                self.target_md = g.add_markdown("")
                self.cb_refresh_backup = g.add_checkbox(
                    "백업 갱신", initial_value=False,
                    hint="켜면 지금 source.usd 로 source_prev.usd 를 덮는다 — 이미 스크래치가 "
                         "들어간 USD 라면 원본 백업이 사라진다")
                self.btn_apply = g.add_button("USD에 적용", color="red",
                                              hint="Blender 로 source.usd 를 다시 만든다")
                self.apply_md = g.add_markdown("")

        self.btn_place.on_click(lambda _: self._on_place())
        self.btn_load.on_click(lambda _: self._on_load())
        self.btn_check.on_click(lambda _: self._on_check())
        self.btn_apply.on_click(self._on_apply_click)
        if self.tool is None:
            self.state_md.content = f"**스크래치 도구를 불러오지 못했다:** {self.tool_error}"
        self.set_enabled(True)

    def set_enabled(self, enabled: bool) -> None:
        """스튜디오 ``_set_buttons`` 가 함께 부른다. 조건이 안 되는 버튼은 켜지지 않는다."""
        usable = enabled and self.tool is not None
        for button, extra in ((self.btn_place, self.library_ok), (self.btn_load, True),
                              (self.btn_check, True), (self.btn_apply, self.blender is not None)):
            with contextlib.suppress(Exception):      # 스튜디오 _set_buttons 와 같은 관용
                button.disabled = not (usable and extra)

    # ------------------------------------------------------------ studio hooks
    def on_object_change(self, obj: str) -> None:
        """물체가 바뀌었다 — 이전 물체의 것은 전부 버리고, 저장본이 있으면 불러와 보인다."""
        with self._lock:
            self._set_scratches([], "none")
            self._align, self._align_key = None, None
            spec_path = self._spec_path(obj)
            if self.tool is not None and spec_path.exists():
                try:
                    self._set_scratches(self.tool.load_spec(spec_path), "file")
                except Exception as exc:  # noqa: BLE001
                    self.state_md.content = f"**Error:** scratches.json 읽기 실패 — {exc}"
            self.vis_md.content = ""
            self.apply_md.content = ""
            self._refresh_target_md(obj)
            self._refresh_state_md()
            self.draw()

    def on_viewpoints_changed(self) -> None:
        """Generate / h5 로드 뒤 — 색을 새 viewpoint 로 다시 칠한다(같은 스레드에서 동기로)."""
        if not self.scratches:
            return
        try:
            self._check(self.studio.object_dd.value)
        except Exception as exc:  # noqa: BLE001
            self.vis_md.content = f"**Error:** {exc}"
            print(f"[defects] check error: {exc}")
        self.draw()

    # --------------------------------------------------------------- actions
    def _run(self, status_md, busy_text: str, fn) -> None:
        """스튜디오와 같은 방식의 워커 — 바쁨 표시, 버튼 잠금, 물체가 바뀌면 결과 폐기."""
        st = self.studio
        if st.generating:
            return
        st.generating = True
        st._set_buttons(False)
        status_md.content = busy_text
        obj = st.object_dd.value

        def work():
            try:
                fn(obj)
            except Exception as exc:  # noqa: BLE001
                status_md.content = f"**Error:** {str(exc)[:300]}"
                print(f"[defects] error: {exc}")
            finally:
                st.generating = False
                st._set_buttons(True)

        threading.Thread(target=work, daemon=True).start()

    def _changed(self, obj: str, status_md) -> bool:
        if obj != self.studio.object_dd.value:
            status_md.content = (f"**Discarded** — 작업 중 Object 가 `{obj}` → "
                                 f"`{self.studio.object_dd.value}` 로 바뀌었습니다.")
            return True
        return False

    def _on_place(self) -> None:
        lo, hi = sorted((float(self.nb_len_lo.value), float(self.nb_len_hi.value)))
        opts = self.tool.PlanOptions(count=int(self.nb_count.value), seed=int(self.nb_seed.value),
                                     length_mm=(lo, hi), strength=float(self.nb_strength.value))
        self._run(self.state_md, "⏳ 배치 중…", lambda obj: self._place(obj, opts))

    def _place(self, obj: str, opts) -> None:
        warn = ""
        try:
            scratches = self.tool.plan_scratches(obj, opts, surface=self._surface(obj),
                                                 data_root=self.studio.data_root,
                                                 log=lambda *_: None)
        except self.tool.ScratchError as exc:
            if not exc.placed:
                raise
            scratches = exc.placed
            warn = (f"\n\n⚠ {len(scratches)}/{opts.count}개만 배치됐다 — 매끈한 자리가 모자란다. "
                    f"길이를 줄이세요.")
        if self._changed(obj, self.state_md):
            return
        with self._lock:
            self._set_scratches(scratches, "planned")
            self._check_if_possible(obj)
            self._refresh_state_md(warn)
            self.draw()

    def _on_load(self) -> None:
        def load(obj):
            path = self._spec_path(obj)
            if not path.exists():
                self.state_md.content = f"저장본 없음 — `{path.relative_to(self.studio.data_root.parent)}`"
                return
            scratches = self.tool.load_spec(path)
            if self._changed(obj, self.state_md):
                return
            with self._lock:
                self._set_scratches(scratches, "file")
                self._check_if_possible(obj)
                self._refresh_state_md()
                self.draw()
        self._run(self.state_md, "⏳ 불러오는 중…", load)

    def _on_check(self) -> None:
        def check(obj):
            if not self.scratches:
                self.vis_md.content = "스크래치가 없다 — 먼저 **배치** 하거나 저장본을 불러오세요."
                return
            with self._lock:
                self._check(obj)
                self.draw()
        self._run(self.vis_md, "⏳ 가시성 계산 중…", check)

    def _on_apply_click(self, event) -> None:
        if not self.scratches:
            self.apply_md.content = "적용할 스크래치가 없다 — 먼저 **배치** 하세요."
            return
        obj = self.studio.object_dd.value
        client = getattr(event, "client", None)
        if client is None:
            self._start_apply()
            return
        backup = self._spec_path(obj).with_name("source_prev.usd")
        with client.gui.add_modal("source.usd 를 다시 만듭니다") as modal:
            client.gui.add_markdown(
                f"`data/{obj}/mesh/source.usd` 를 스크래치 **{len(self.scratches)}개**로 "
                f"다시 만든다 (Isaac 이 읽는 파일).\n\n"
                f"백업 `source_prev.usd`: "
                + ("**지금 USD 로 덮어쓴다**" if self.cb_refresh_backup.value else
                   ("그대로 둔다" if backup.exists() else "새로 만든다")))
            ok = client.gui.add_button("적용", color="red")
            cancel = client.gui.add_button("취소")

            @ok.on_click
            def _(_):
                modal.close()
                self._start_apply()

            @cancel.on_click
            def _(_):
                modal.close()

    def _start_apply(self) -> None:
        opts = self.tool.ApplyOptions(force_backup=bool(self.cb_refresh_backup.value),
                                      blender=self.blender, timeout_s=600)
        scratches = [dict(s) for s in self.scratches]
        self._run(self.apply_md, "⏳ Blender 실행 중…",
                  lambda obj: self._apply(obj, scratches, opts))

    def _apply(self, obj: str, scratches: list, opts) -> None:
        recent: list[str] = []

        def on_line(line: str) -> None:
            recent.append(line.replace("[scratch]", "").strip())
            self.apply_md.content = "⏳ Blender …\n\n" + "\n\n".join(f"`{x}`" for x in recent[-3:])

        result = self.tool.apply_scratches(obj, scratches, opts, data_root=self.studio.data_root,
                                           log=lambda *_: None, on_line=on_line)
        patches = [x.split(":", 1)[1].strip() for x in result.lines if "faces, area ratio" in x]
        self.apply_md.content = (
            f"**Done** · `source.usd` · 스크래치 {len(scratches)}개 · 백업 "
            + ("갱신" if result.backup_written and opts.force_backup else
               "새로 만듦" if result.backup_written else "기존 유지")
            + ("\n\n" + "\n\n".join(f"S{i}: {p}" for i, p in enumerate(patches)) if patches else ""))
        if obj == self.studio.object_dd.value:
            with self._lock:
                self.origin = "file"
                self._refresh_state_md()
                self._refresh_target_md(obj)

    # --------------------------------------------------------------- state
    def _set_scratches(self, scratches: list, origin: str) -> None:
        keys = self.tool.SPEC_KEYS if self.tool else ()
        self.scratches = [{k: s[k] for k in keys} for s in scratches]
        self.origin = origin
        self._version += 1
        self.vis, self._vis_key = None, None

    def _spec_path(self, obj: str) -> Path:
        return Path(self.studio.data_root) / obj / "mesh" / "scratches.json"

    def _refresh_state_md(self, suffix: str = "") -> None:
        n = len(self.scratches)
        if n == 0:
            text = "스크래치 없음"
        elif self.origin == "file":
            text = f"저장본 `scratches.json` · {n}개"
        else:
            text = f"배치 {n}개 · **USD 미적용**"
        if self._align is not None and not self._align.ok:
            text += f"\n\n⚠ {self._align.note} — 화면에 그리지 않았다. Mesh source 를 " \
                    f"source.obj 로 바꾸거나 Align 을 확인하세요."
        self.state_md.content = text + suffix

    def _refresh_target_md(self, obj: str) -> None:
        mesh_dir = Path(self.studio.data_root) / obj / "mesh"
        backup = "있음" if (mesh_dir / "source_prev.usd").exists() else "없음"
        text = f"`data/{obj}/mesh/source.usd` · 백업 `source_prev.usd` {backup}"
        if self.blender is None:
            text += "\n\n**Blender 없음** (`$BLENDER` 또는 PATH) — 배치·표시·가시성만 가능"
        self.target_md.content = text

    # ------------------------------------------------------------ geometry
    def _surface(self, obj: str):
        """스크래치가 앉는 면(target.ply 우선) — 배치와 표면 투영이 같이 쓴다."""
        if obj not in self._surface_cache:
            self._surface_cache[obj] = self.tool.load_surface(obj, data_root=self.studio.data_root)
        return self._surface_cache[obj]

    def _source_full(self, obj: str):
        if obj not in self._source_full_cache:
            import trimesh
            self._source_full_cache[obj] = trimesh.load(
                str(Path(self.studio.data_root) / obj / "mesh" / "source.obj"), force="mesh")
        return self._source_full_cache[obj]

    def _look(self, s: dict) -> dict:
        """흠집의 실제 모양 — 음영 텍스처와 크기. Blender 단계와 같은 식으로 스탬프 크기를 잰다.

        PNG 가 없으면(다른 머신에서 연 scratches.json) 모양 없이 길이로만 크기를 잡는다.
        """
        key = (s["png"], round(float(s["length_mm"]), 4), round(float(s["strength"]), 4))
        if key not in self._look_cache:
            span = float(s["length_mm"]) / 1000.0 * self.tool.SPAN_FACTOR
            size = 256
            try:
                rgb, alpha, dst = self.tool.composite_normal_array(
                    Path(s["png"]), size=size, length_px=size / self.tool.SPAN_FACTOR,
                    strength=float(s["strength"]))
                image = dg.shade_normal_map(rgb, alpha)
                stamp_w, stamp_h = dst[0] / size * span, dst[1] / size * span
            except Exception:  # noqa: BLE001
                image, stamp_w, stamp_h = None, float(s["length_mm"]) / 1000.0, 0.003
            self._look_cache[key] = {
                "image": image, "span": span,
                # Blender 단계의 패치 크기와 같다 — 흠집 자신의 상자에 여유를 붙인 것.
                "half_len": stamp_w * 0.5 * 1.25,
                "half_wid": max(stamp_h * 0.5 * 1.5, stamp_w * 0.06)}
        return self._look_cache[key]

    def _geometry(self, obj: str) -> list[dict]:
        key = (obj, self._version)
        if self._geom_key != key:
            target, _ = self._surface(obj)
            geom = []
            for s in self.scratches:
                look = self._look(s)
                samples, normals = dg.scratch_samples(target, s)
                decal = polyline = None
                if look["image"] is not None:
                    decal = dg.decal_patch(target, s, look["half_len"], look["half_wid"],
                                           look["span"])
                if decal is None or not len(decal[1]):
                    # 모양을 못 그리면(PNG 없음·면 밖) 흠집 자리를 가는 선으로라도 남긴다 —
                    # 안 그리면 스크래치가 있는지조차 안 보인다.
                    decal, polyline = None, dg.scratch_polyline(target, s)
                geom.append({"samples": samples, "normals": normals,
                             "decal": decal, "polyline": polyline, "image": look["image"]})
            self._geom, self._geom_key = geom, key
        return self._geom

    def _alignment(self, obj: str) -> dg.FrameAlignment:
        """source.obj → 지금 화면 좌표. 화면이 source.obj 그 자체면 이동 0."""
        st = self.studio
        if st.data is None:
            return dg.IDENTITY                  # 미리보기: 이 패널이 source.obj 를 직접 그린다
        source = (Path(st.data_root) / obj / "mesh" / "source.obj").resolve()
        input_mesh = st.data.get("input_mesh")
        if input_mesh and Path(str(input_mesh)).resolve() == source:
            return dg.IDENTITY
        if st.scene_full_mesh is None:
            return dg.FrameAlignment(np.zeros(3), float("nan"), float("nan"), True,
                                     "좌표계 미검증 (화면 메시 없음)")
        key = (obj, id(st.scene_full_mesh))
        if self._align_key != key:
            scene_target = next((v[1] for v in st.mesh_cache.values()
                                 if v[0] is st.scene_full_mesh), None)
            self._align = dg.align_source_to_scene(
                self._source_full(obj), st.scene_full_mesh,
                source_target=self._surface(obj)[0], scene_target=scene_target)
            self._align_key = key
        return self._align

    # ----------------------------------------------------------- visibility
    def _views(self):
        """지금 화면의 viewpoint 와 **생성 당시** 판정 조건. 없으면 None."""
        st = self.studio
        if st.last is not None:
            surface, p = st.last["surface"], st.last["params"]
            extras = surface.get("extras") or {}
            fov = extras.get("effective_fov_mm")
            if fov is None or not len(np.asarray(fov)):
                fov = min(float(p["fov_width_mm"]), float(p["fov_height_mm"]))
            spec = visibility.SensorSpec(
                working_distance_mm=float(p["working_distance_mm"]),
                max_incidence_deg=float(p["max_incidence_deg"]),
                depth_of_field_mm=float(p["dof_mm"]),
                occlusion_tolerance_mm=OCCLUSION_TOLERANCE_MM)
            frames = visibility.ViewFrames.from_extras(extras)
            return (np.asarray(surface["positions"]), np.asarray(surface["normals"]),
                    spec, fov, frames, "사각 프레임" if frames is not None else "원 근사")
        if st.data is not None and st.data.get("n"):
            d = st.data
            fov = min(float(d.get("fov_w_mm") or 50.0), float(d.get("fov_h_mm") or 50.0))
            spec = visibility.SensorSpec(working_distance_mm=float(d["wd_m"]) * 1000.0,
                                         occlusion_tolerance_mm=OCCLUSION_TOLERANCE_MM)
            return (np.asarray(d["positions"]), np.asarray(d["normals"]), spec, fov, None,
                    "h5 — 입사각·DoF 미적용")
        return None

    def _vis_state_key(self, obj: str):
        st = self.studio
        return (obj, self._version, id(st.last) if st.last is not None else id(st.data))

    def _check_if_possible(self, obj: str) -> None:
        if self._views() is not None:
            self._check(obj)
        else:
            self.vis_md.content = "viewpoint 가 없다 — **Generate viewpoints** 뒤에 자동으로 검사된다."

    def _check(self, obj: str) -> None:
        views = self._views()
        if views is None or not self.scratches:
            self.vis, self._vis_key = None, None
            self.vis_md.content = ("viewpoint 가 없다 — **Generate viewpoints** 뒤에 자동으로 "
                                   "검사된다." if self.scratches else "")
            return
        align = self._alignment(obj)
        if not align.ok:
            self.vis, self._vis_key = None, None
            self._refresh_state_md()
            self.vis_md.content = f"⚠ {align.note} — 검사하지 않았다."
            return
        positions, normals, spec, fov, frames, how = views
        geom = self._geometry(obj)
        samples = [(align.to_scene(g["samples"]), g["normals"]) for g in geom]
        self.vis = dg.scratch_visibility(samples, positions, normals,
                                         self.studio.scene_full_mesh, spec,
                                         fov_mm=fov, frames=frames)
        self._vis_key = self._vis_state_key(obj)

        states = [v.state for v in self.vis]
        lines = [(f"**가시성** · 온전히 {states.count('seen')}/{len(states)} · "
                  f"부분 {states.count('split')} · 놓침 {states.count('missed')} — "
                  f"vp {len(positions)} · {how}")]
        for i, (s, v) in enumerate(zip(self.scratches, self.vis)):
            lines.append(f"{STATE_DOT[v.state]} S{i} · {s['length_mm']:.0f}mm · "
                         f"온전 {v.n_full} · 부분 {v.n_partial} · 점 {v.points_seen}/{v.k}")
        if self.studio.scene_full_mesh is None:
            lines.append("_(화면 메시 없음 — 가림은 검사하지 않았다)_")
        self.vis_md.content = "\n\n".join(lines)

    # ----------------------------------------------------------------- draw
    def draw(self) -> None:
        """스크래치를 그린다. ``_build_scene`` 이 레이어를 비운 뒤에도 불린다(재계산 없이 캐시로)."""
        st = self.studio
        with self._lock:
            # defects 는 Scratches 토글이 숨기고 보이는 것, defects_base 는 viewpoint 가 없을 때
            # 깔아 둔 물체(토글 없음). 둘 다 이름이 고정이라, 다시 그리기 전에 반드시 걷어낸다 —
            # 같은 이름으로 새로 올린 뒤 옛 핸들을 지우면 새 노드가 지워진다.
            for key in ("defects", "defects_base"):
                layer = st.layers[key]
                while layer:
                    layer.pop().remove()
            handles = st.layers["defects"]
            if not self.scratches or self.tool is None:
                return
            obj = st.object_dd.value
            if st.data is None:
                self._draw_preview_base(obj)
            align = self._alignment(obj)
            if not align.ok:
                self._refresh_state_md()
                return
            srv = st.server
            for i, g in enumerate(self._geometry(obj)):
                # 흠집 모양만 — 노멀맵을 비스듬한 빛으로 한 번 칠한 텍스처를 표면 격자에 입힌다.
                # 테두리·번호는 두지 않는다: 모양을 가리고, 판정은 패널 목록이 이미 말한다.
                if g["decal"] is not None:
                    handles.append(srv.scene.add_mesh_trimesh(
                        f"/scene/defects/s{i}/decal", self._decal_mesh(g, align)))
                else:
                    handles.append(srv.scene.add_spline_catmull_rom(
                        f"/scene/defects/s{i}/line", positions=align.to_scene(g["polyline"]),
                        color=FALLBACK_RGB, line_width=2.0))
            visible = bool(getattr(st, "cb_defects", None) is None or st.cb_defects.value)
            for h in handles:
                h.visible = visible

    @staticmethod
    def _decal_mesh(g: dict, align):
        import trimesh
        from PIL import Image

        vertices, faces, uv = g["decal"]
        material = trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.fromarray(g["image"], "RGBA"), alphaMode="BLEND",
            doubleSided=True, metallicFactor=0.0, roughnessFactor=1.0)
        return trimesh.Trimesh(align.to_scene(vertices), faces, process=False,
                               visual=trimesh.visual.TextureVisuals(uv=uv, material=material))

    def _draw_preview_base(self, obj: str) -> None:
        """화면에 viewpoint 가 아직 없을 때 — 물체만 source.obj 로 깔아 스크래치가 뜰 자리를 만든다.

        ``defects_base`` 레이어에 넣는다: 토글이 걸리지 않고, 다음 ``_build_scene`` 이 레이어를
        비울 때 같이 걷힌다.
        """
        st = self.studio
        config.apply_object_placement(obj)
        frame = st.server.scene.add_frame(
            "/scene", show_axes=False,
            wxyz=np.asarray(config.TARGET_OBJECT["rotation"], dtype=np.float64),
            position=(0.0, 0.0, 0.0))
        mesh = self._source_full(obj)
        handle = st.server.scene.add_mesh_simple(
            "/scene/mesh", vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces),
            color=PREVIEW_MESH_RGB, opacity=0.25, side="double")
        handle.visible = bool(st.cb_mesh.value)
        # 부모(frame)를 먼저, 자식(메시)을 나중에 넣는다 — 걷어낼 때 뒤에서부터 pop 하므로
        # 자식이 먼저 걷힌다.
        st.layers["defects_base"].extend([frame, handle])
