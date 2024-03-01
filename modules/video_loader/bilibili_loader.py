import json
import re
import warnings
from typing import List, Tuple

import requests
from langchain_core.documents import Document
from bilibili_api import sync, video
from langchain_community.document_loaders.base import BaseLoader

# Pre-compile regular expressions for video ID extraction
BV_PATTERN = re.compile(r"BV\w+")
AV_PATTERN = re.compile(r"av[0-9]+")


class BiliBiliLoader(BaseLoader):
    """
    Loader for fetching transcripts from BiliBili videos.
    """

    def __init__(self, video_urls: List[str], sessdata: str, bili_jct: str, buvid3: str):
        """Initialize with bilibili url.
        Args:
            video_urls (List[str]): List of BiliBili video URLs.
            sessdata (str): SESSDATA cookie value for authentication.
            bili_jct (str): BILI_JCT cookie value for authentication.
            buvid3 (str): BUVI3 cookie value for authentication.
        """
        self.video_urls = video_urls
        self.credential = video.Credential(
            sessdata=sessdata, bili_jct=bili_jct, buvid3=buvid3
        )

    def load(self) -> List[Document]:
        """
        Load and return a list of documents containing video transcripts.
        Returns:
            List[Document]: List of Document objects transcripts and metadata.
        """
        results = []
        for url in self.video_urls:
            transcript, video_info = self._get_bilibili_subs_and_info(url)
            doc = Document(page_content=transcript, metadata=video_info)
            results.append(doc)
        return results

    def _get_bilibili_subs_and_info(self, url: str) -> Tuple[str, dict]:
        """
        Retrieve video information and transcript for a given BiliBili URL.
        Args:
            url (str): BiliBili video URL.
        Returns:
            Tuple[str, dict]: A tuple containing the transcript and video information.
        """
        bvid = BV_PATTERN.search(url)
        if bvid:
            v = video.Video(bvid=bvid.group(), credential=self.credential)
        else:
            aid = AV_PATTERN.search(url)
            if aid:
                v = video.Video(aid=int(aid.group()[2:]), credential=self.credential)
            else:
                raise ValueError(f"Unable to find a valid video ID in URL: {url}")

        video_info = sync(v.get_info())
        video_info.update({"url": url})
        sub = sync(v.get_subtitle(video_info["cid"]))

        # Retrieve and process subtitle content
        sub_list = sub["subtitles"]
        if sub_list:
            sub_url = sub_list[0]["subtitle_url"]
            if not sub_url.startswith("http"):
                sub_url = "https:" + sub_url
            response = requests.get(sub_url)
            if response.status_code == 200:
                raw_sub_titles = json.loads(response.content)["body"]
                raw_transcript = " ".join([c["content"] for c in raw_sub_titles])

                raw_transcript_with_meta_info = (
                    f"Video Title: {video_info['title']}, "
                    f"description: {video_info['desc']}\n\n"
                    f"Transcript: {raw_transcript}"
                )
                return raw_transcript_with_meta_info, video_info
            else:
                warnings.warn(
                    f"Failed to fetch subtitles for {url}. "
                    f"HTTP Status Code: {response.status_code}"
                )
                return "", video_info
        else:
            warnings.warn(
                f"No subtitles found for video: {url}. Returning empty transcript."
            )
            return "", video_info
