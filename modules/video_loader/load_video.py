import gradio as gr

from .bilibili_loader import BiliBiliLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document


def load_bilibili_video(url: str, sessdata: str, bili_jct: str, buvid3: str) -> list[Document]:
    try:
        loader = BiliBiliLoader(video_urls=[url], sessdata=sessdata, buvid3=buvid3, bili_jct=bili_jct)
        return loader.load()
    except Exception as e:
        raise gr.Error("Failed to load video from BiliBili." + str(e))


def load_youtube_video(url: str) -> list[Document]:
    try:
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        return loader.load()
    except Exception as e:
        raise gr.Error("Failed to load video from YouTube." + str(e))


def load_custom_video(file):
    # todo: implement custom video loader
    return None
