from threading import Lock

import gradio as gr
from langchain_core.documents import Document

from modules.set_model import setup_model
from modules.video_loader.load_video import load_bilibili_video, load_youtube_video, load_custom_video
from modules.utils import get_summery, get_vector_store, get_answer, get_splits


class ChatWrapper:
    def __init__(self):
        self.lock = Lock()

    def __call__(self, vector_store, query, history, llm):
        self.lock.acquire()
        try:
            if llm is None:
                gr.Warning("模型尚未加载，请检查您的API密钥。")
                return query, history
            if vector_store is None:
                gr.Warning("向量库尚未加载，请检查您的视频输入或等待加载完成。")
                return query, history
            response = get_answer(vector_store, query, history, llm)
            history.append((query, response))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return "", history


chat = ChatWrapper()

# Gradio Web UI setup
with gr.Blocks() as demo:
    with gr.Row():
        gr.HTML("<h2>VideoChat | Powered by Qwen</h2>")
    with gr.Row():
        with gr.Column(scale=1):
            api_key_textbox = gr.Textbox(
                label="API Key",
                placeholder="此处粘贴您的API密钥...",
                lines=1,
                type="password",
            )
            gr.HTML("<p> 没有 API key? 请 <a href='https://dashscope.console.aliyun.com/apiKey'>点击这里</a> 获取。</p>")
            gen_summery_checkbox = gr.Checkbox(label="Generate Summery", info="Options")
            with gr.Tabs():
                with gr.Tab(label="Bilibili"):
                    with gr.Accordion(label="Cookie Settings", open=False):
                        with gr.Group():
                            sessdata_textbox = gr.Textbox(
                                label="SESSDATA",
                                placeholder="此处粘贴您的SESSDATA...",
                                lines=1,
                            )
                            buvid3_textbox = gr.Textbox(
                                label="BUVID3",
                                placeholder="此处粘贴您的BUVID3...",
                                lines=1,
                            )
                            bili_jct_textbox = gr.Textbox(
                                label="BILI_JCT",
                                placeholder="此处粘贴您的BILI_JCT...",
                                lines=1,
                            )

                    with gr.Group():
                        bili_url_textbox = gr.Textbox(
                            label="Video URL",
                            placeholder="此处粘贴您的视频URL...",
                            lines=1
                        )
                        with gr.Row():
                            bili_submit_btn = gr.Button(value="Load")

                with gr.Tab(label="Youtube"):
                    with gr.Group():
                        youtube_url_textbox = gr.Textbox(
                            label="Video URL",
                            placeholder="此处粘贴您的视频URL...",
                            lines=1,
                        )
                        ytb_submit_btn = gr.Button(value="Load")
                with gr.Tab(label="Custom"):
                    with gr.Group():
                        custom_video = gr.Video(label="Video File", sources=["upload"])
                        custom_submit_btn = gr.Button(value="Load")
        with gr.Column(scale=3):
            with gr.Accordion(label="Video Summary", open=False):
                summary_textbox = gr.Textbox(
                    show_label=False,
                    placeholder="此处显示视频概述...",
                    lines=4
                )
            chatbot = gr.Chatbot(height=370)
            with gr.Group():
                with gr.Row():
                    query_textbox = gr.Textbox(
                        show_label=False,
                        autofocus=True,
                        placeholder="请输入您的问题...",
                        lines=1,
                        scale=25
                    )
                    clear_btn = gr.ClearButton([query_textbox], value="Clear", min_width=1)
                    submit_btn = gr.Button(value="Send", min_width=1)
            gr.Examples(
                examples=[
                    "您可以帮我概况一下视频内容吗？",
                    "请帮我提取视频中的关键信息。"
                ],
                inputs=query_textbox,
            )

    history_s = gr.State(value=[])
    llm_s = gr.State(value=None)
    text_s = gr.State(value=Document(""))
    splits_s = gr.State(value=[])
    vector_store_s = gr.State(value=None)

    # setup model
    api_key_textbox.change(
        setup_model,
        inputs=[api_key_textbox],
        outputs=[llm_s],
    )

    splits_args = dict(
        fn=get_splits,
        inputs=[text_s],
        outputs=[splits_s]
    )
    summery_args = dict(
        fn=get_summery,
        inputs=[splits_s, llm_s, gen_summery_checkbox],
        outputs=[summary_textbox]
    )
    vector_args = dict(
        fn=get_vector_store,
        inputs=[splits_s],
        outputs=[vector_store_s]
    )
    bili_submit_btn.click(
        load_bilibili_video,
        inputs=[bili_url_textbox, sessdata_textbox, bili_jct_textbox, buvid3_textbox],
        outputs=[text_s]
    ).then(**splits_args).then(**vector_args).then(**summery_args)
    ytb_submit_btn.click(
        load_youtube_video,
        inputs=[youtube_url_textbox],
        outputs=[text_s]
    ).then(**splits_args).then(**vector_args).then(**summery_args)
    custom_submit_btn.click(
        load_custom_video,
        inputs=[custom_video],
        outputs=[text_s]
    ).then(**splits_args).then(**vector_args).then(**summery_args)

    # chat about the video
    query_args = dict(
        fn=chat,
        inputs=[vector_store_s, query_textbox, history_s, llm_s],
        outputs=[query_textbox, chatbot]
    )
    submit_btn.click(**query_args)
    query_textbox.submit(**query_args)

demo.queue(100).launch(debug=False)
