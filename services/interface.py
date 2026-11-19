"""Shared user-facing source summaries and setup navigation."""
from pathlib import Path

import streamlit as st

from services.baseline_catalog import validate_baseline
from services.dataset_labels import dataset_label
from services.file_selection import file_label


def show_baseline_summary(path, dataset, *, n_genes=None, container=None):
    ui = container if container is not None else st
    try:
        info = validate_baseline(Path(path)) if n_genes is None else None
        n_genes = info.n_genes if info is not None else n_genes
        ui.info(f"Active baseline · {dataset_label(dataset)} · {file_label(Path(path))} · {n_genes:,} genes")
        with ui.expander('Baseline source details'):
            st.write('Baseline file:', str(Path(path).expanduser().resolve()))
            st.caption('This is the baseline used for this calculation. Selecting another file changes the results on this page.')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        ui.warning(f'The selected baseline is unavailable or invalid: {exc}')


def show_analysis_support(dataset):
    if dataset == 'Custom':
        st.info('Custom baselines support Analysis 2 with your target-gene list. Analysis 1 requires Gasperini genomic annotations. Analysis 3 requires a separate precomputed simulation file; fitting a baseline does not create simulations.')
    elif dataset == 'Replogle':
        st.caption('Use this baseline in Analysis 2. Analysis 1 uses Gasperini genomic annotations; Analysis 3 uses its own simulation file.')
    else:
        st.caption('Use this baseline in Analyses 1–2 with the required annotations or target list. Analysis 3 uses its own simulation file.')


def reset_setup():
    """Clear session inputs only; saved files and worker processes are preserved."""
    prefixes = ('setup_', 'gpu_', 'modality_', 'matrix_', 'perturbation_', 'rule_', 'control_', 'expression_', 'ids_', 'symbols_', 'totals_', 'program_file_', 'program_baseline_', 'cis_baseline_')
    for key in list(st.session_state):
        if str(key).startswith(prefixes):
            del st.session_state[key]


def show_next_analysis(dataset):
    if dataset == 'Gasperini' and st.button('Continue to Analysis 1'):
        st.switch_page('pages/analysis1_cis_power.py')
    if st.button('Continue to Analysis 2'):
        st.switch_page('pages/analysis2_target_program.py')


def preferred_simulation_dataset(datasets):
    """Prefer Replogle only when its simulations are actually available."""
    return next((d for d in datasets if str(d).lower().startswith('replogle')), datasets[0])


def median_design_value(values):
    """Nearest available design to the median; ties use the smaller design."""
    import statistics
    ordered = sorted(set(values))
    median = statistics.median(ordered)
    return min(ordered, key=lambda value: (abs(value - median), value))


def show_navigation():
    destinations = [('Home', 'app.py'), ('Data Setup', 'pages/0_Data_Setup.py'),
                    ('3 · Screen-wide', 'pages/analysis3_screen_power.py'),
                    ('2 · Target program', 'pages/analysis2_target_program.py'),
                    ('1 · Nearby genes', 'pages/analysis1_cis_power.py')]
    for column, (label, page) in zip(st.columns(len(destinations)), destinations):
        if column.button(label, key='nav_' + page, width="stretch"):
            st.switch_page(page)


def show_method_status():
    st.caption('Analyses 1–2: analytical estimates · Analysis 3: simulated detection frequencies · CRT update pending.')
    with st.expander('Statistical methods and limitations', expanded=False):
        st.write('Analyses 1–2 use a negative-binomial/Wald approximation with calibration where supported. Analysis 3 reports the detection frequencies in the supplied simulation artifact, which does not identify a CRT test. The CRT replacement needs the updated model code or documented CRT results.')
