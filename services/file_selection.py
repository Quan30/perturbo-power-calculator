"""Readable file choices shared by the app's data controls."""
from pathlib import Path

import streamlit as st

from services.uploads import save_uploaded_dataset


def file_label(path: Path) -> str:
    path = Path(path)
    if path.name == 'baseline_params.npz':
        from services.baseline_jobs import job_source_name, read_job
        try:
            job = read_job(path.parent)
            if job.input_path:
                return f'{job_source_name(job)} · {job.dataset.title()} reference · #{path.parent.name[-6:]}'
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return f'Custom reference · {path.parent.name}' if path.parent.name.startswith('custom_') else f'{path.parent.name} · reference'
    return path.name


def choose_file(label: str, default: str, suffixes: set[str], upload_dir: Path, *, key: str, container=None) -> str:
    ui = container if container is not None else st
    selected = default
    if default and Path(default).is_file():
        ui.caption(f'{label}: {file_label(Path(default))}')
    else:
        ui.info(f'Choose a file for {label.lower()}.')
    with ui.expander(f'Change {label.lower()}', expanded=not default or not Path(default).is_file()):
        uploaded = st.file_uploader(f'Upload {label.lower()}', type=sorted(s.lstrip('.') for s in suffixes), key=f'{key}_upload')
        if uploaded is not None:
            signature = (uploaded.name, uploaded.size, getattr(uploaded, 'file_id', None))
            if st.session_state.get(f'{key}_signature') != signature:
                try:
                    saved = save_uploaded_dataset(uploaded, upload_dir, allowed_suffixes=suffixes, prefix=key)
                    st.session_state[f'{key}_saved'] = str(saved)
                    st.session_state[f'{key}_signature'] = signature
                except Exception as exc:
                    st.error(f'Could not read uploaded file: {exc}')
            if st.session_state.get(f'{key}_signature') == signature:
                selected = st.session_state[f'{key}_saved']
        with st.expander('Advanced: server file path'):
            override = st.text_input(f'{label} server path', value=default, key=f'{key}_path_{default}')
            if override != default:
                selected = override
    return selected
