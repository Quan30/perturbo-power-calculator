"""Select an existing reference or fit one from server-local single-cell data."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import streamlit as st

from services.baseline_catalog import (
    baseline_catalog_signature,
    discover_baselines,
    source_signature,
    validate_baseline,
)
from services.baseline_jobs import list_jobs, read_job, start_fit, job_label, job_source_name
from services.dataset_labels import dataset_label
from services.custom_data import (
    CustomDataMapping,
    control_cell_mask,
    custom_mapping_to_options,
    validate_custom_data_file,
    check_custom_count_sample,
)
from services.uploads import save_uploaded_dataset
from services.custom_metadata import inspect_custom_metadata
from services.file_selection import choose_file, file_label
from services.interface import show_navigation, show_baseline_summary, show_analysis_support, show_next_analysis, reset_setup

APP_ROOT = Path(__file__).resolve().parent.parent


@st.cache_data(show_spinner=False)
def _discover_baselines_cached(root: str, signature: tuple) -> list[Path]:
    return discover_baselines(Path(root))


st.set_page_config(page_title="Data Setup", layout="wide")
st.title("Data Setup")
st.write("Choose a fitted PerTurbo reference baseline, or fit one from your own single-cell data. Analysis 3 uses a separate repeated-screen simulation artifact.")
active_summary = st.container()
if message := st.session_state.pop("gpu_activation_message", None):
    st.success(message)
st.button('Reset this setup', on_click=reset_setup, help='Clear selections and validation in this session. Saved files, fit jobs, and running workers are preserved.')
show_navigation()
st.markdown("**1. Choose a dataset → 2. Reuse or fit a baseline → 3. Activate it → 4. Open an analysis.**")
with st.expander("How to fit a baseline and find the output"):
    st.write("Bundled Gasperini and Replogle baselines are in baselines/. For custom data, select Compute from raw data, upload or select a file, map its fields, validate, then start a fit. Use the completed job's baseline to activate it.")
    st.caption("Fitting a new baseline requires the PerTurbo model package; see the README section *Fit a baseline from your own data*.")

def _dataset_changed():
    if st.session_state.get('setup_dataset') == 'Custom':
        st.session_state['setup_baseline_source'] = 'Compute from raw data'


dataset = st.selectbox(
    "Dataset type",
    ["Gasperini", "Replogle", "Custom"],
    format_func=dataset_label,
    key='setup_dataset',
    on_change=_dataset_changed,
    help="Select the raw-data schema and baseline defaults to use.",
)
show_analysis_support(dataset)
with st.expander("Advanced: server data folder"):
    root = Path(st.text_input("Server data folder", key="setup_root", value=os.environ.get("CORTADO_DATA_ROOT", str(APP_ROOT / "baselines")), help="Folder where fitted baseline files and background fit jobs are stored.")).expanduser()
if not root.is_dir():
    st.info(f"Create `{root}` or choose an existing server folder.")

choice = st.radio("Baseline source", key="setup_baseline_source", options=["Use a fitted baseline", "Compute from raw data"], horizontal=True, help="Reuse existing fitted parameters or start a new reference-model fit from single-cell data.")
if choice == "Use a fitted baseline":
    files = _discover_baselines_cached(str(root), baseline_catalog_signature(root))
    if files:
        selected = st.selectbox("Available baseline files", key="setup_existing_baseline", options=files, format_func=file_label, help="Choose a validated baseline parameter file discovered in the server folder.")
        candidate = Path(choose_file("Baseline parameters", str(selected), {".npz"}, APP_ROOT / "baselines" / "uploads", key="setup_baseline"))
    else:
        st.info("No valid baseline_params.npz files were found in this folder.")
        candidate = Path(choose_file("Baseline parameters", "", {".npz"}, APP_ROOT / "baselines" / "uploads", key="setup_baseline"))
    if st.button("Use this baseline", type="primary"):
        try:
            info = validate_baseline(candidate)
            st.session_state["gpu_baseline_path"] = str(info.path)
            st.session_state["gpu_dataset"] = dataset
            st.success(f"Selected {info.n_genes:,} genes from {info.path.name}.")
        except Exception as exc:
            st.error(f"Cannot use this baseline: {exc}")
else:
    if importlib.util.find_spec("cortado") is None:
        st.warning(
            "Fitting a new baseline needs the PerTurbo model package (`cortado`), which is not installed "
            "in this environment. All three analyses work with the included Gasperini and Replogle "
            "baselines. See the README section *Fit a baseline from your own data*."
        )
    if dataset == 'Custom':
        st.caption('Custom input can be AnnData (.h5ad) or MuData (.h5mu). Select the count matrix and metadata fields from your file below.')
    else:
        st.caption("Raw Gasperini input is `.h5mu`; the supplied Replogle workflow uses `.h5ad`. Large datasets are faster to access by server path than through your browser.")
    if dataset == "Custom":
        st.subheader("Step 1 · Select custom AnnData / MuData")
        st.info(
            "Upload or select an external AnnData/MuData, declare its fields, validate it, "
            "then fit a control-only PerTurbo baseline in the background."
        )
        st.caption(
            "Required: raw counts in X or a layer, a perturbation column with identifiable "
            "negative controls, and unique stable gene IDs. Use a server path for multi-GB files."
        )
        mode = st.radio(
            "Where is the raw file?",
            ["On this server", "Upload from my computer"],
            horizontal=True,
            key="setup_custom_location",
            help="This app is configured for uploads up to 2 GB; server paths are recommended for larger datasets.",
        )
        source = None
        if mode == "On this server":
            local_files = sorted({p for folder in (APP_ROOT / "data" / "uploads", APP_ROOT / "data" / "examples", root) if folder.is_dir() for pattern in ("*.h5ad", "*.h5mu") for p in folder.glob(pattern)})
            available = st.selectbox("Available single-cell files", key="setup_custom_file", options=[None, *local_files], format_func=lambda p: "Choose a data file…" if p is None else file_label(p))
            source = available
            with st.expander("Advanced: use another server file"):
                custom_path_text = st.text_input("Custom .h5ad or .h5mu file on server", value="", key="setup_custom_path", help="For large files stored elsewhere on this server.")
            if custom_path_text.strip():
                source = Path(custom_path_text).expanduser()
        else:
            uploaded = st.file_uploader(
                "Upload a custom .h5ad or .h5mu",
                type=["h5ad", "h5mu"],
                key="setup_custom_upload",
                help="The upload is saved to data/uploads before validation. The configured limit is 2 GB.",
            )
            if uploaded is not None:
                upload_key = (uploaded.name, uploaded.size, getattr(uploaded, "file_id", None))
                if st.button("Save uploaded file to server"):
                    try:
                        saved = save_uploaded_dataset(
                            uploaded,
                            APP_ROOT / "data" / "uploads",
                            allowed_suffixes={".h5ad", ".h5mu"},
                            prefix="custom",
                        )
                        st.session_state["gpu_custom_saved_upload"] = upload_key
                        st.session_state["gpu_custom_upload_path"] = str(saved)
                        st.success(f"Saved {uploaded.name}; it is ready for validation.")
                    except Exception as exc:
                        st.error(f"Could not save uploaded file: {exc}")
                saved_path = Path(st.session_state.get("gpu_custom_upload_path", ""))
                if (
                    st.session_state.get("gpu_custom_saved_upload") == upload_key
                    and saved_path.is_file()
                ):
                    source = saved_path
                    st.caption(f"Selected file: {uploaded.name}")
        mapping = None
        if source is not None and source.is_file():
            try:
                @st.cache_data(show_spinner=False)
                def load_metadata(path, signature):
                    return inspect_custom_metadata(path)
                metadata = load_metadata(str(source), source_signature(source))
                file_key = str(source_signature(source))
                if source.suffix.lower() == ".h5mu":
                    modalities = list(metadata)
                    preferred = next((m for m in ("rna", "gene", "transcriptome") if m in modalities), modalities[0])
                    rna_modality = st.selectbox("RNA modality (MuData only)", modalities, index=modalities.index(preferred), key=f"modality_{file_key}")
                else:
                    rna_modality = None
                fields = metadata[rna_modality or "AnnData"]
                field_key = f"{file_key}_{rna_modality}"
                st.caption(f"{fields['n_cells']:,} cells × {fields['n_genes']:,} genes. Choose the fields below; expression matrices are not loaded for this preview.")
                matrices = ([None] if fields["has_x"] else []) + fields["layers"]
                if not matrices:
                    raise ValueError("No expression matrix was found")
                st.subheader("Step 2 · Map counts and metadata")
                selected_matrix = st.selectbox("Raw-count matrix", matrices, format_func=lambda v: "X (main matrix)" if v is None else f"Layer: {v}", index=matrices.index("counts") if "counts" in matrices else 0, key=f"matrix_{field_key}", help="Choose unnormalized integer counts. Layers are listed by name; X is the main matrix.")
                obs_columns = list(fields["obs"].columns)
                if not obs_columns:
                    raise ValueError("No cell annotations were found in the selected modality")
                preferred_column = next((c for c in ("perturbation", "guide", "gene_transcript", "guide_assignment") if c in obs_columns), obs_columns[0])
                perturbation_column = st.selectbox("Perturbation column in obs", obs_columns, index=obs_columns.index(preferred_column), key=f"perturbation_{field_key}")
                labels = fields["obs"][perturbation_column].dropna().astype(str)
                counts = labels.value_counts()
                control_labels = sorted(counts.index.tolist())
                if not control_labels:
                    raise ValueError("The chosen perturbation column contains no labels")
                rule = st.radio("Control matching rule", ["Exact label", "Contains text", "Regular expression"], index=1, horizontal=True, key=f"rule_{field_key}_{perturbation_column}")
                control_match = {"Exact label": "exact", "Contains text": "contains", "Regular expression": "regex"}[rule]
                if control_match == "exact":
                    preferred_control = next((v for v in ("non-targeting", "NTC", "control", "CONTROL") if v in control_labels), control_labels[0])
                    control_value = st.selectbox("Negative-control value", control_labels, index=control_labels.index(preferred_control), key=f"control_{field_key}_{perturbation_column}", format_func=lambda v: f"{v} ({int(counts[v]):,} cells)")
                else:
                    control_value = st.text_input("Negative-control expression", value="", placeholder="non-targeting" if control_match == "contains" else r"^\d+_non-targeting", key=f"expression_{control_match}_{field_key}_{perturbation_column}", help="Matching is case-sensitive. Contains text uses literal text; a regular expression searches each label. Use ^ and $ to match a whole label.")
                mask = None
                try:
                    mask = control_cell_mask(fields["obs"][perturbation_column], control_value, control_match)
                except ValueError as exc:
                    if control_value:
                        st.error(str(exc))
                    else:
                        st.info("Enter an expression to identify negative-control cells.")
                if mask is not None:
                    selected_labels = fields["obs"].loc[mask, perturbation_column].astype(str).value_counts()
                    st.caption(f"{int(mask.sum()):,} control cells selected across {len(selected_labels):,} labels: {int(mask.sum()):,} of {fields['n_cells']:,} cells ({100 * mask.mean():.1f}%). Matching is case-sensitive.")
                    if mask.all():
                        st.warning('Every cell is selected as a control. Confirm that this is a control-only dataset, or narrow the matching expression.')
                    elif 0 < mask.sum() < 100:
                        st.warning('Fewer than 100 control cells are selected. A fit may be unstable; check whether the expression should pool additional control labels.')
                    if not mask.any():
                        st.warning("No cells match this expression. Adjust the expression or choose another column.")
                    else:
                        with st.expander("Preview matched control labels"):
                            st.dataframe(selected_labels.head(50).rename_axis("Matched label").reset_index(name="Cells"), width="stretch")
                            if len(selected_labels) > 50:
                                st.caption("Showing the 50 most frequent matched labels; the total above includes all matches.")
                var_columns = list(fields["var"].columns)
                gene_id_column = st.selectbox("Stable gene-ID column in var (optional)", [None, *var_columns], format_func=lambda v: "Use gene index (var_names)" if v is None else v, key=f"ids_{field_key}")
                symbol_default = next((c for c in ("gene_name", "gene_symbol", "symbol") if c in var_columns), None)
                gene_symbol_column = st.selectbox("Gene-symbol column in var (optional)", [None, *var_columns], index=([None, *var_columns]).index(symbol_default), format_func=lambda v: "Use stable gene IDs" if v is None else v, key=f"symbols_{field_key}")
                total_counts_column = st.selectbox("Total-counts column in obs (optional)", [None, *obs_columns], format_func=lambda v: "Calculate from raw counts" if v is None else v, key=f"totals_{field_key}")
                preview = fields["var"].head(3).copy()
                preview.index.name = "Gene index"
                st.dataframe(preview, width="stretch")
                mapping = CustomDataMapping(control_match=control_match, rna_modality=rna_modality, counts_layer=selected_matrix, perturbation_column=perturbation_column, control_value=control_value, gene_id_column=gene_id_column, gene_symbol_column=gene_symbol_column, total_counts_column=total_counts_column) if mask is not None and mask.any() else None
                if mapping is not None:
                    st.markdown('**Review the mapping before checking**')
                    st.write(f"Source: {file_label(source)} · RNA: {rna_modality or 'AnnData'} · Counts: {selected_matrix or 'X (main matrix)'}")
                    st.write(f"Controls: {perturbation_column} · {rule} · `{control_value}`")
                    st.caption(f"Gene IDs: {gene_id_column or 'var_names'} · Gene labels: {gene_symbol_column or 'stable IDs'} · Total counts: {total_counts_column or 'calculated from raw counts'}")
            except Exception as exc:
                st.error(f"Could not inspect the selected file: {exc}")
        else:
            st.info("Choose a data file to populate the mapping dropdowns.")
        validation_key = None
        if mapping is not None and source is not None and source.is_file():
            try:
                validation_key = (
                    source_signature(source),
                    tuple(sorted(custom_mapping_to_options(mapping).items())),
                )
            except OSError:
                validation_key = None
        if st.session_state.get('gpu_custom_validation_key') is not None and st.session_state.get('gpu_custom_validation_key') != validation_key:
            st.info('The file or mapping has changed. Check again before fitting.')
        if st.button(
            "Check custom data structure", type="primary", disabled=validation_key is None
        ):
            try:
                report = validate_custom_data_file(source, mapping)
                st.session_state["gpu_custom_validation_key"] = validation_key
                st.session_state["gpu_custom_validation_report"] = {
                    "n_cells": report.n_cells,
                    "n_genes": report.n_genes,
                    "n_control_cells": report.n_control_cells,
                    "count_source": report.count_source,
                    "gene_id_source": report.gene_id_source,
                    "warnings": report.warnings,
                }
                st.session_state.pop('gpu_custom_count_sample', None)
                st.session_state.pop('gpu_custom_count_error', None)
                try:
                    with st.spinner('Checking a sample of raw control counts…'):
                        sampled = check_custom_count_sample(source, mapping)
                    st.session_state['gpu_custom_count_sample'] = sampled
                except Exception as exc:
                    st.session_state['gpu_custom_count_error'] = str(exc)
            except Exception as exc:
                st.session_state.pop("gpu_custom_validation_key", None)
                st.session_state.pop("gpu_custom_validation_report", None)
                st.error(f"Custom data is not compatible with the declared mapping: {exc}")

        validated = (
            validation_key is not None
            and st.session_state.get("gpu_custom_validation_key") == validation_key
        )
        if validated:
            report = st.session_state["gpu_custom_validation_report"]
            st.success(
                f"Structure checked: {report['n_cells']:,} cells × {report['n_genes']:,} genes; "
                f"counts from {report['count_source']}; gene IDs from {report['gene_id_source']}."
            )
            for warning in report["warnings"]:
                st.warning(warning)
            if st.session_state.get('gpu_custom_count_error'):
                st.error('Raw-count sample failed: ' + st.session_state['gpu_custom_count_error'])
                st.info('Choose an unnormalized count matrix and check again before fitting.')
            elif st.session_state.get('gpu_custom_count_sample'):
                st.success(f"Raw-count sample checked: {st.session_state.get('gpu_custom_count_sample', 0):,} selected control cells have finite, nonnegative integer counts.")
            else:
                st.info('Raw-count sample not checked yet. Check again before fitting.')
            st.caption('The count check samples up to 100 control cells. Before optimization, the worker checks every control row used in the fit; unsampled rows have not yet been checked.')
            if st.session_state.get('gpu_custom_count_sample') and not st.session_state.get('gpu_custom_count_error'):
                max_cells = st.number_input(
                    "Maximum control cells in baseline fit", key="setup_custom_max",
                    min_value=100,
                    value=10000,
                    step=1000,
                    help="Only cells matching the selected control rule are eligible.",
                )
                steps = st.number_input(
                    "Control fit steps", key="setup_custom_steps", min_value=100, value=2000, step=100,
                    help="Number of optimization steps used for the control/reference fit."
                )
                st.caption(
                    "Before fitting, the worker checks that the selected matrix contains finite, "
                    "nonnegative integer counts. The job continues if this browser page reloads."
                )
                if st.button("Compute custom baseline", type="primary"):
                    try:
                        options = {
                            "max_cells": str(max_cells),
                            "steps": str(steps),
                            **custom_mapping_to_options(mapping),
                        }
                        job = start_fit(
                            Path(source), root / "baseline_jobs", "custom", options
                        )
                        st.session_state["gpu_fit_job_dir"] = str(job.output_path.parent)
                        st.session_state["gpu_job_choice"] = job.output_path.parent
                        st.success("Fit selected below: " + job_label(job.output_path.parent))
                    except Exception as exc:
                        st.error(f"Could not start custom baseline fit: {exc}")
        elif source is not None and not source.is_file():
            st.warning(f"Custom data file was not found: `{source}`")
    else:
        suffix = "h5mu" if dataset == "Gasperini" else "h5ad"
        mode = st.radio("Where is the raw file?", ["On this server", "Upload from my computer"], key="setup_builtin_location", horizontal=True, help="Server paths are recommended for large single-cell datasets.")
        if mode == "On this server":
            raw_files = sorted({p for folder in (APP_ROOT / "data" / "uploads", APP_ROOT / "data" / "examples", root) if folder.is_dir() for p in folder.glob(f"*.{suffix}")})
            source = st.selectbox("Available single-cell files", key="setup_builtin_file", options=[None, *raw_files], format_func=lambda p: "Choose a data file…" if p is None else file_label(p))
            with st.expander("Advanced: use another server file"):
                raw_path = st.text_input(f"Raw .{suffix} file on server", key="setup_builtin_path", value="", help="For large files stored elsewhere on this server.")
            if raw_path.strip():
                source = Path(raw_path).expanduser()
        else:
            uploaded = st.file_uploader("Upload a raw dataset", key="setup_builtin_upload", type=[suffix], help="Large uploads are held by Streamlit; use a server path for multi-GB data.")
            source = None
            if uploaded is not None:
                upload_dir = APP_ROOT / "data" / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                candidate = upload_dir / Path(uploaded.name).name
                upload_key = (uploaded.name, uploaded.size, getattr(uploaded, "file_id", None))
                if st.button("Save uploaded file to server"):
                    staging = candidate.with_suffix(candidate.suffix + ".part")
                    uploaded.seek(0)
                    with staging.open("wb") as dest:
                        while chunk := uploaded.read(8 * 1024 * 1024):
                            dest.write(chunk)
                    staging.replace(candidate)
                    st.session_state["gpu_saved_upload"] = upload_key
                    st.success(f"Saved {candidate.name}; you can now compute its baseline.")
                if st.session_state.get("gpu_saved_upload") == upload_key and candidate.exists():
                    source = candidate
        max_cells = st.number_input("Maximum cells in baseline fit", key="setup_builtin_max", min_value=100, value=10000, step=1000, help="Upper limit on cells used to fit the reference model; lower values run faster but may be less stable.")
        steps = st.number_input("Control fit steps", key=f"setup_builtin_steps_{dataset}", min_value=100, value=2500 if dataset == "Gasperini" else 2000, step=100, help="Number of optimization steps used for the control/reference fit.")
        st.caption("Fits only the reference model. It may take substantial GPU memory and time; the job continues if this browser page reloads.")
        if st.button("Compute baseline", type="primary", disabled=source is None):
            try:
                job = start_fit(Path(source), root / "baseline_jobs", dataset.lower(), {"max_cells": str(max_cells), "steps": str(steps)})
                st.session_state["gpu_fit_job_dir"] = str(job.output_path.parent)
                st.session_state["gpu_job_choice"] = job.output_path.parent
            except Exception as exc:
                st.error(f"Could not start baseline fit: {exc}")

# Saved jobs are discovered on every run, including after a browser refresh.
jobs = list_jobs(root / "baseline_jobs")
if jobs:
    st.subheader("Baseline fit jobs")
    remembered = st.session_state.get("gpu_fit_job_dir")
    selected_index = next((i for i, path in enumerate(jobs) if str(path) == remembered), 0)
    pending = st.session_state.pop("gpu_pending_job_choice", None)
    if pending in jobs:
        st.session_state["gpu_job_choice"] = pending
    if st.session_state.get("gpu_job_choice") not in jobs:
        st.session_state["gpu_job_choice"] = jobs[selected_index]
    # Freeze display labels for this render; workers can change status mid-request.
    job_labels = {path: ("Latest requested in this session · " if str(path) == remembered else "") + job_label(path) for path in jobs}
    selected_job_dir = st.selectbox(
        "Select a fit job",
        jobs,
        index=selected_index,
        format_func=job_labels.__getitem__,
        help="Choose a background fit job to inspect its status, logs, and output.",
        key="gpu_job_choice",
    )
    st.button("Refresh status and log")
    auto_refresh = st.checkbox('Automatically refresh running job (every 5 seconds)', value=True, key='setup_auto_refresh')
    try:
        was_running = read_job(selected_job_dir).status == 'running'
    except (OSError, ValueError, KeyError):
        was_running = False
    if not hasattr(st, 'fragment'):
        st.caption('Automatic refresh needs Streamlit 1.37 or newer. Manual refresh is available below.')

    def render_job_status():
        try:
            job = read_job(selected_job_dir)
            st.markdown(f"**Showing job for: {job_source_name(job)}**")
            st.caption(f"Dataset: {job.dataset.title()} · Status: {job.status.title()}")
            if remembered:
                if str(selected_job_dir) == remembered:
                    st.info("Most recently requested fit in this session. The log below belongs to this selected job.")
                else:
                    st.info("You are viewing an older job, rather than the most recently requested fit in this session.")
                    current = next((path for path in jobs if str(path) == remembered), None)
                    if current is not None and st.button("Show latest job"):
                        st.session_state["gpu_pending_job_choice"] = current
                        st.rerun()
            with st.expander("Job details"):
                st.write("Source file:", str(job.input_path) if job.input_path else "Unavailable for this older job")
                st.write("Started (UTC):", job.started_at or "Unavailable for this older job")
                st.write("Job ID:", selected_job_dir.name)
                if job.options:
                    st.json(job.options)
            if job.status == "complete":
                info = validate_baseline(job.output_path)
                job_dataset = selected_job_dir.name.split("_", 1)[0]
                completed_dataset = {
                    "gasperini": "Gasperini",
                    "replogle": "Replogle",
                    "custom": "Custom",
                }.get(job_dataset, "Custom")
                st.success(f"Baseline ready: {info.n_genes:,} genes · {file_label(info.path)}.")
                if st.button("Use completed baseline", type="primary"):
                    st.session_state["gpu_baseline_path"] = str(info.path)
                    st.session_state["gpu_dataset"] = completed_dataset
                    destination = "Analyses 1–2" if completed_dataset == "Gasperini" else "Analysis 2"
                    st.session_state["gpu_activation_message"] = f"Selected {dataset_label(completed_dataset)} for {destination}."
                    st.rerun()
            elif job.status == "failed":
                st.error(f"Baseline fit failed: {job.error}")
            else:
                st.info("Baseline fit running. Status and log refresh every 5 seconds." if auto_refresh and hasattr(st, "fragment") else "Baseline fit running. Click Refresh status and log to check progress.")
            st.markdown(f"**Fit log — {job_source_name(job)}**")
            if job.log_path.exists():
                # tqdm rewrites its progress line with carriage returns; show the latest update.
                log = job.log_path.read_text(errors="replace").replace("\r", "\n")
                st.code(log[-6000:] or "Worker is starting; check again shortly.")
            else:
                st.info("The worker has not written a log yet. Check again shortly.")
            if was_running and job.status != "running":
                st.rerun()
        except Exception as exc:
            st.warning(f"Could not read job status: {exc}")
    if hasattr(st, 'fragment'):
        st.fragment(run_every=5 if auto_refresh and was_running else None)(render_job_status)()
    else:
        render_job_status()

if st.session_state.get("gpu_baseline_path"):
    show_baseline_summary(st.session_state["gpu_baseline_path"], st.session_state.get("gpu_dataset", "Custom"), container=active_summary)
    show_next_analysis(st.session_state.get("gpu_dataset", "Custom"))
