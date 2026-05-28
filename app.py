import os
import gradio as gr
import matplotlib
matplotlib.use("Agg")

from src.layout import (
    build_sidebar, build_filterbar, build_map_view,
    on_load, on_closure_row_select, on_station_row_select, on_view_switch,
    TOPBAR, FOOTER, GAP_FIX_JS,
)
from src.dashboard import build_data_view

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "style.css")) as _f:
    CSS = _f.read()

with gr.Blocks(
    title="Road-Rail Resilience",
    css=CSS,
    theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
) as demo:

    gr.HTML(TOPBAR)

    with gr.Row(equal_height=False, elem_classes=["rr-shell"], elem_id="rr-shell"):
        view_sel = build_sidebar()

        with gr.Column(scale=1, min_width=0, elem_classes=["rr-main"], elem_id="rr-main"):

            # Filterbar and map view are now wrapped together.
            # This prevents the filterbar from leaving an empty placeholder
            # when the Data Overview tab is selected.
            with gr.Group(visible=True, elem_id="rr-map-area") as map_area:
                filterbar, f_date, f_time, f_type, f_dist, f_dur, f_stn, go_btn = build_filterbar()

                (v1, _map1, _note1, _raw1, _tbl1,
                 _nb_col, _nb_note, _nb_tbl,
                 _pred1, _htbl1, _hfig1, _ttbl1, _tfig1) = build_map_view()

            v2 = build_data_view()

    gr.HTML(FOOTER)
    gr.HTML(GAP_FIX_JS)

    _LOAD_OUTPUTS = [
        _tbl1, _note1, _raw1, _map1,
        _nb_tbl, _nb_note, _nb_col,
        _pred1,
        _htbl1, _hfig1, _ttbl1, _tfig1,
    ]

    # IMPORTANT:
    # This now has only TWO outputs because on_view_switch returns two values.
    view_sel.change(on_view_switch, inputs=view_sel, outputs=[map_area, v2])

    go_btn.click(
        on_load,
        inputs=[f_date, f_time, f_type, f_dist, f_dur, f_stn],
        outputs=_LOAD_OUTPUTS,
    )

    f_stn.change(
        on_load,
        inputs=[f_date, f_time, f_type, f_dist, f_dur, f_stn],
        outputs=_LOAD_OUTPUTS,
    )

    _tbl1.select(
        on_closure_row_select,
        inputs=[_tbl1, _raw1, f_dist, f_stn, f_date, f_time, f_dur],
        outputs=[_nb_tbl, _nb_note, _map1],
    )

    _nb_tbl.select(
        on_station_row_select,
        inputs=[_nb_tbl, f_date, f_time],
        outputs=[_pred1, _htbl1, _hfig1, _ttbl1, _tfig1],
    )


if __name__ == "__main__":
    demo.launch(share=True, show_error=True, server_port=7860)