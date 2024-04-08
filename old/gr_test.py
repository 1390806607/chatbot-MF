import gradio as gr
from modules.utils import get_history_names, get_first_history_name
with gr.Blocks() as demo:
    with gr.Row(elem_id="chuanhu-history-body"):
        with gr.Column(scale=6, elem_id="history-select-wrap"):
            historySelectList = gr.Radio(
                label="从列表中加载对话",
                choices=get_history_names(),
                value=get_first_history_name(),
                # multiselect=False,
                container=False,
                elem_id="history-select-dropdown"
            )


    # load_history_from_file_args = dict(
    #     fn=load_chat_history,
    #     inputs=[current_model, historySelectList],
    #     outputs=[saveFileName, systemPromptTxt, chatbot, single_turn_checkbox, temperature_slider, top_p_slider, n_choices_slider, stop_sequence_txt, max_context_length_slider, max_generation_slider, presence_penalty_slider, frequency_penalty_slider, logit_bias_txt, user_identifier_txt],
    # )
demo.launch()